"""A bounded LangGraph workflow that reuses the project's RAG tools."""

import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

from rag_book_agent.agent.tools import AgentTools
from rag_book_agent.models import Answer, SearchResult


class ResearchState(TypedDict, total=False):
    question: str
    session_id: str
    force_web: bool
    lane: str
    lane_reason: str
    plan: List[str]
    planner_mode: str
    local_results: List[SearchResult]
    web_results: List[dict]
    web_search_status: str
    routing_reason: str
    evidence_preview: List[dict]
    budget: Dict[str, int]
    trace: List[dict]
    answer: Answer
    next_action: str
    step_count: int
    observation: str


class LangGraphAgent:
    """Explicit graph implementation kept separate from the stable agent workflow."""

    def __init__(self, service):
        self.service = service
        self.tools = AgentTools(service)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ResearchState)
        workflow.add_node("understand", self._complexity_router)
        workflow.add_node("plan", self._plan)
        workflow.add_node("decide_action", self._decide_action)
        workflow.add_node("local_search", self._local_research)
        workflow.add_node("web_search", self._web_research)
        workflow.add_node("observe", self._observe)
        workflow.add_node("evidence_review", self._deduplicate)
        workflow.add_node("synthesize", self._synthesize)
        workflow.add_edge(START, "understand")
        workflow.add_edge("understand", "plan")
        workflow.add_edge("plan", "decide_action")
        workflow.add_conditional_edges(
            "decide_action", self._next_node,
            {"local_search": "local_search", "web_search": "web_search", "finalize": "evidence_review"},
        )
        workflow.add_edge("local_search", "observe")
        workflow.add_edge("web_search", "observe")
        workflow.add_edge("observe", "decide_action")
        workflow.add_edge("evidence_review", "synthesize")
        workflow.add_edge("synthesize", END)
        return workflow.compile()

    def _complexity_router(self, state: ResearchState) -> Dict:
        question = state["question"].strip()
        lower = question.lower()
        deep_terms = ("对比", "比较", "分析", "方案", "规划", "路线", "调研", "趋势", "最新", "今天", "为什么", "优缺点", "影响")
        standard_terms = ("如何", "怎么", "步骤", "原理", "配置", "报错", "部署", "实现", "解释")
        if state.get("force_web") or any(term in question for term in deep_terms):
            lane, reason = "deep", "问题包含分析/比较/时效性需求，采用多步规划与联网车道"
        elif any(term in question for term in standard_terms) or len(question) > self.service.settings.agent_fast_question_max_chars:
            lane, reason = "standard", "常规专业问答，采用混合检索与自适应联网"
        else:
            lane, reason = "fast", "短问题且无复杂意图，采用本地稠密检索极速车道"
        trace = list(state.get("trace", []))
        trace.append({"node": "complexity_router", "lane": lane, "reason": reason})
        # Fast/standard lanes skip the LLM planner, but every downstream node
        # still receives a valid one-item plan.
        return {"lane": lane, "lane_reason": reason, "plan": [question], "step_count": 0,
                "trace": trace}

    @staticmethod
    def _choose_lane(state: ResearchState) -> str:
        return state.get("lane", "standard")

    def _decide_action(self, state: ResearchState) -> Dict:
        steps = state.get("step_count", 0)
        local = state.get("local_results")
        web_done = "web_search_status" in state
        best = max((item.rerank_score for item in (local or [])), default=0.0)
        if steps >= self.service.settings.agent_max_react_steps:
            action, reason = "finalize", "达到工具调用上限，开始综合"
        elif local is None:
            action, reason = "local_search", "先检索本地知识库建立事实基础"
        elif not web_done and (state.get("force_web") or state.get("lane") == "deep" or
                               best < self.service.settings.agent_local_confidence_threshold):
            action, reason = "web_search", "本地证据需要时效信息或外部补充"
        else:
            action, reason = "finalize", "现有证据已足够，停止调用工具"
        trace = list(state.get("trace", []))
        trace.append({"node": "decide_action", "action": action, "reason": reason, "step": steps})
        return {"next_action": action, "routing_reason": reason, "trace": trace}

    @staticmethod
    def _next_node(state: ResearchState) -> str:
        return state.get("next_action", "finalize")

    def _observe(self, state: ResearchState) -> Dict:
        local_count = len(state.get("local_results", []))
        web_count = len(state.get("web_results", []))
        observation = "已获得 %d 条本地证据、%d 条网页证据" % (local_count, web_count)
        trace = list(state.get("trace", []))
        trace.append({"node": "observe", "observation": observation})
        return {"observation": observation, "step_count": state.get("step_count", 0) + 1,
                "trace": trace}

    def _fast_local_research(self, state: ResearchState) -> Dict:
        trace = list(state.get("trace", []))
        try:
            results = self.tools.fast_local_search(state["question"])
            trace.append({"node": "fast_local_research", "results": len(results), "strategy": "dense-only-no-rerank"})
        except Exception as error:
            results = []
            trace.append({"node": "fast_local_research", "error": str(error)[:160]})
        return {"plan": [state["question"]], "local_results": results, "web_results": [],
                "web_search_status": "skipped-fast-lane", "trace": trace}

    def _plan(self, state: ResearchState) -> Dict:
        plan = self._llm_plan(state["question"])
        mode = "llm" if plan else "rule-fallback"
        if not plan:
            parts = [item.strip(" ?。") for item in re.split(r"[，,；;]", state["question"]) if item.strip()]
            plan = (parts or [state["question"]])[: self.service.settings.agent_max_react_steps]
        trace = list(state.get("trace", []))
        trace.append({"node": "plan", "sub_questions": len(plan), "mode": mode})
        return {"plan": plan, "planner_mode": mode, "trace": trace}

    def _llm_plan(self, question: str) -> List[str]:
        settings = self.service.settings
        if not settings.api_key or not settings.api_base or not settings.api_model:
            return []
        prompt = ("将用户问题拆为最多 %d 个独立检索子问题。简单问题保持原样。"
                  "只返回 JSON 字符串数组，不要解释。\n问题：%s" %
                  (settings.agent_max_react_steps, question))
        try:
            response = httpx.post(settings.api_base.rstrip("/") + "/responses",
                                  headers={"Authorization": "Bearer " + settings.api_key,
                                           "Content-Type": "application/json"},
                                  json={"model": settings.api_model, "input": prompt,
                                        "temperature": 0, "max_output_tokens": 180},
                                  timeout=settings.agent_planner_timeout_seconds)
            response.raise_for_status()
            raw = (response.json().get("output_text") or "").strip()
            match = re.search(r"\[[\s\S]*\]", raw)
            values = json.loads(match.group(0) if match else raw)
            if not isinstance(values, list):
                return []
            return [str(item).strip() for item in values if str(item).strip()][:
                settings.agent_max_react_steps]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return []

    def _local_research(self, state: ResearchState) -> Dict:
        results = []
        trace = list(state.get("trace", []))
        for question in state.get("plan") or [state["question"]]:
            try:
                found = self.tools.local_search(question)
            except Exception as error:
                found = []
                trace.append({"node": "local_research", "query": question, "error": str(error)[:160]})
            results.extend(found)
            trace.append({"node": "local_research", "query": question, "results": len(found)})
        return {"local_results": results, "trace": trace}

    def _route_web(self, state: ResearchState) -> Dict:
        results = state.get("local_results", [])
        best = max((item.rerank_score for item in results), default=0.0)
        threshold = self.service.settings.agent_local_confidence_threshold
        if state.get("force_web"):
            reason = "用户强制要求联网"
        elif state.get("lane") == "deep":
            reason = "深度车道启用网页研究与多步综合"
        elif results and best >= threshold:
            reason = "本地最高重排分 %.3f >= %.3f，跳过联网" % (best, threshold)
        else:
            reason = "本地证据不足（最高重排分 %.3f < %.3f）" % (best, threshold)
        trace = list(state.get("trace", []))
        trace.append({"node": "route_web", "decision": "web_research" if state.get("force_web") or best < threshold else "deduplicate", "reason": reason})
        return {"routing_reason": reason, "trace": trace}

    @staticmethod
    def _decide_web_route(state: ResearchState) -> str:
        if state.get("force_web") or state.get("lane") == "deep":
            return "web_research"
        return "deduplicate" if state.get("routing_reason", "").startswith("本地最高") else "web_research"

    def _web_research(self, state: ResearchState) -> Dict:
        results = []
        trace = list(state.get("trace", []))
        def search(question):
            return question, self.tools.web_search(question)

        try:
            with ThreadPoolExecutor(max_workers=max(1, len(state["plan"]))) as pool:
                futures = [pool.submit(search, question) for question in state["plan"]]
                for future in as_completed(futures):
                    try:
                        question, found = future.result()
                    except Exception as error:
                        trace.append({"node": "web_research", "error": str(error)[:160]})
                        continue
                    results.extend(found)
                    trace.append({
                        "node": "web_research", "query": question, "results": len(found),
                        "pipeline": dict(self.service.web_search.last_trace),
                    })
        except Exception as error:
            trace.append({"node": "web_research", "status": "fallback-local", "error": str(error)[:160]})
            return {"web_results": [], "web_search_status": "fallback-local", "trace": trace}
        return {"web_results": results, "web_search_status": "success" if results else "empty-fallback-local", "trace": trace}

    def _deduplicate(self, state: ResearchState) -> Dict:
        local_by_parent = {}
        for result in state.get("local_results", []):
            key = result.chunk.parent_id or result.chunk.id
            old = local_by_parent.get(key)
            if old is None or result.rerank_score > old.rerank_score:
                local_by_parent[key] = result
        web_by_url = {}
        for item in state.get("web_results", []):
            url = item.get("url", "")
            if url and url not in web_by_url:
                web_by_url[url] = item
        local_results, local_used, local_dropped = self._fit_local(list(local_by_parent.values()))
        web_results, web_used, web_dropped = self._fit_web(list(web_by_url.values()))
        preview = [{"kind": "local", "title": item.document_title, "page": item.chunk.page_start} for item in local_results]
        preview += [{"kind": "web", "title": item.get("title", "网页资料"), "url": item.get("url", "")} for item in web_results]
        budget = {"local_used_chars": local_used, "local_dropped": local_dropped, "web_used_chars": web_used, "web_dropped": web_dropped,
                  "two_stage_full_sources": self.service.settings.agent_two_stage_full_sources}
        trace = list(state.get("trace", [])); trace.append({"node": "deduplicate", "local": len(local_results), "web": len(web_results), **budget})
        return {"local_results": local_results, "web_results": web_results, "evidence_preview": preview, "budget": budget, "trace": trace}

    def _fit_local(self, items):
        items.sort(key=lambda item: (item.rerank_score, item.fusion_score), reverse=True)
        chosen, used, limit = [], 0, self.service.settings.agent_local_context_chars
        for index, item in enumerate(items):
            size = len(item.chunk.text)
            if chosen and used + size > limit: continue
            item = self._copy_result(item, item.chunk.text[:limit] if not chosen and size > limit else item.chunk.text)
            size = len(item.chunk.text)
            # Two-stage reading: keep full source text only for top-ranked evidence.
            if index >= self.service.settings.agent_two_stage_full_sources:
                summary = self._extract_summary(item.chunk.text)
                item = self._copy_result(item, "[背景摘要] " + summary)
                size = len(item.chunk.text)
            chosen.append(item); used += size
        return chosen, used, len(items) - len(chosen)

    @staticmethod
    def _extract_summary(text: str, limit: int = 110) -> str:
        clean = re.sub(r"\s+", " ", text).strip()
        if len(clean) <= limit:
            return clean
        sentences = re.split(r"(?<=[。！？!?])", clean)
        summary = ""
        for sentence in sentences:
            if len(summary) + len(sentence) > limit:
                break
            summary += sentence
        return (summary or clean[:limit]).rstrip() + "..."

    @staticmethod
    def _copy_result(result: SearchResult, text: str) -> SearchResult:
        chunk = result.chunk
        copied = type(chunk)(chunk.id, chunk.document_id, text, chunk.heading, chunk.page_start,
                             chunk.page_end, chunk.position, chunk.parent_id, chunk.chunk_type)
        return SearchResult(copied, result.document_title, result.document_path, result.sparse_score,
                            result.dense_score, result.fusion_score, result.rerank_score)

    def _fit_web(self, items):
        chosen, used, limit = [], 0, self.service.settings.agent_web_context_chars
        for item in items:
            size = len(item.get("text", ""))
            if chosen and used + size > limit: continue
            value = dict(item)
            if not chosen and size > limit: value["text"] = value.get("text", "")[:limit]; size = limit
            chosen.append(value); used += size
        return chosen, used, len(items) - len(chosen)

    def _synthesize(self, state: ResearchState) -> Dict:
        answer = self.service._ask(
            state["question"],
            session_id=state["session_id"],
            force_web=state.get("force_web", False),
            precomputed_sources=state.get("local_results", []),
            precomputed_web=state.get("web_results", []),
            agent_mode=True,
            agent_plan=state["plan"],
        )
        answer.mode = "langgraph-" + answer.mode
        trace = list(state.get("trace", []))
        trace.append({"node": "synthesize", "mode": answer.mode})
        return {"answer": answer, "trace": trace}

    def run(self, question: str, session_id: str = "default", force_web: bool = False) -> Answer:
        state = self.graph.invoke({"question": question, "session_id": session_id, "force_web": force_web})
        self._record_final_state(state)
        return state["answer"]

    def stream(self, question: str, session_id: str = "default", force_web: bool = False):
        """Yield actual LangGraph node updates for the web UI."""
        state = {"question": question, "session_id": session_id, "force_web": force_web}
        for update in self.graph.stream(state, stream_mode="updates"):
            node, delta = next(iter(update.items()))
            state.update(delta)
            yield node, delta
        self._record_final_state(state)

    def _record_final_state(self, state: ResearchState) -> None:
        self.service.last_retrieval["agent"] = True
        self.service.last_retrieval["agent_engine"] = "langgraph"
        self.service.last_retrieval["agent_lane"] = state.get("lane", "standard")
        self.service.last_retrieval["agent_lane_reason"] = state.get("lane_reason", "")
        self.service.last_retrieval["agent_web_status"] = state.get("web_search_status", "")
        self.service.last_retrieval["agent_budget"] = state.get("budget", {})
        self.service.last_retrieval["agent_roles"] = ["planner", "researcher", "synthesizer"]
        self.service.last_retrieval["agent_tools"] = list(self.tools.registry.keys())
        self.service.last_retrieval["agent_sub_questions"] = state.get("plan", [])
        self.service.last_retrieval["agent_research"] = state.get("trace", [])
        self.service.last_retrieval["agent_react_steps"] = min(
            len(state.get("plan", [])), self.service.settings.agent_max_react_steps
        )
        self.service.last_retrieval["agent_web_provider"] = self.service.web_search.last_provider
        self.service.last_retrieval["agent_web_pipeline"] = dict(
            self.service.web_search.last_trace
        )
        self.service.storage.add_trace_detail(self.service.last_trace_id, self.service.last_retrieval)
        from rag_book_agent.audit_log import OperationLog
        OperationLog(self.service.settings.database_path.parent.parent).write(
            "LANGGRAPH_TRACE", "%s | %s" % (state["question"], self.service.last_retrieval)
        )

    def draw_png(self) -> bytes:
        """Use LangGraph's official graph renderer for the UI diagram."""
        return self.graph.get_graph().draw_mermaid_png()
