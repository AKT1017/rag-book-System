"""Independent, lightweight multi-agent workflow for broad questions."""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from rag_book_agent.agent.prompts import PLANNER_PROMPT
from rag_book_agent.agent.tools import AgentTools
from rag_book_agent.models import Answer, SearchResult


class AgentOrchestrator:
    """Planner -> researcher -> synthesizer, reusing the existing RAG facade."""

    def __init__(self, service):
        self.service = service
        self.tools = AgentTools(service)

    def run(self, question: str, session_id: str = "default", force_web: bool = False) -> Answer:
        plan = self.plan(question)
        local_results, web_results, research_trace = self.research(plan["sub_questions"])
        # Existing generator remains the single answer boundary, so citations and memory stay consistent.
        answer = self.service._ask(
            question,
            session_id=session_id,
            force_web=force_web,
            precomputed_sources=local_results,
            precomputed_web=web_results,
            agent_mode=True,
            agent_plan=plan["sub_questions"],
        )
        answer.mode = "agent-api" if answer.mode == "api" else "agent-" + answer.mode
        self.service.last_retrieval["agent"] = True
        self.service.last_retrieval["agent_roles"] = ["planner", "researcher", "synthesizer"]
        self.service.last_retrieval["agent_tools"] = list(self.tools.registry.keys())
        self.service.last_retrieval["web_results"] = len(web_results)
        self.service.last_retrieval["agent_research"] = research_trace
        self.service.last_retrieval["agent_sub_questions"] = plan["sub_questions"]
        self.service.last_retrieval["agent_react_steps"] = min(
            len(plan["sub_questions"]), self.service.settings.agent_max_react_steps
        )
        self.service.last_retrieval["agent_web_provider"] = self.service.web_search.last_provider
        return answer

    def plan(self, question: str) -> Dict[str, List[str]]:
        # Deterministic planning is cheap and works when no API key is present.
        parts = [item.strip(" ?。") for item in re.split(r"[，,；;]", question) if item.strip()]
        if len(parts) > 1:
            return {"sub_questions": parts[:3]}
        return {"sub_questions": [question]}

    def research_local(self, questions: List[str]) -> List[SearchResult]:
        unique = {}
        for item in questions:
            for result in self.tools.local_search(item):
                if result.chunk.id not in unique:
                    unique[result.chunk.id] = result
        return list(unique.values())[: self.service.settings.context_top_k]

    def research(self, questions: List[str]):
        questions = questions[: self.service.settings.agent_max_react_steps]
        local_by_id, web_by_url, trace = {}, {}, []
        deadline = time.time() + self.service.settings.agent_timeout_seconds

        def web_work(item):
            started = time.time()
            return item, self.tools.web_search(item), int((time.time() - started) * 1000)

        # SQLite storage is intentionally kept single-threaded.  This preserves the
        # existing connection lifecycle while still allowing independent web workers.
        for item in questions:
            started = time.time()
            try:
                local = self.tools.local_search(item)
                for result in local:
                    key = result.chunk.parent_id or result.chunk.id
                    old = local_by_id.get(key)
                    if old is None or result.rerank_score > old.rerank_score:
                        local_by_id[key] = result
                trace.append({"sub_question": item, "local": len(local), "ms": int((time.time() - started) * 1000)})
            except Exception as exc:
                trace.append({"sub_question": item, "status": "local_error", "error": str(exc)[:160]})

        with ThreadPoolExecutor(max_workers=max(1, len(questions))) as pool:
            futures = [pool.submit(web_work, item) for item in questions]
            for future in as_completed(futures):
                if time.time() >= deadline:
                    break
                try:
                    item, web, elapsed = future.result(timeout=max(1, deadline - time.time()))
                except Exception as exc:
                    trace.append({"status": "error", "error": str(exc)[:160]})
                    continue
                for result in web:
                    url = result.get("url", "")
                    if url and url not in web_by_url:
                        web_by_url[url] = result
                trace.append({"sub_question": item, "web": len(web), "web_ms": elapsed})
        local_results = sorted(local_by_id.values(), key=lambda item: item.rerank_score, reverse=True)
        return local_results[: self.service.settings.context_top_k], list(web_by_url.values())[:5], trace

    def research_web(self, questions: List[str], force_web: bool) -> List[dict]:
        results = []
        deadline = time.time() + self.service.settings.agent_timeout_seconds
        seen = set()
        for item in questions[: self.service.settings.agent_max_react_steps]:
            if time.time() >= deadline:
                break
            try:
                found = self.tools.web_search(item)
            except Exception:
                found = []
            for result in found[:5]:
                url = result.get("url", "")
                if url not in seen:
                    seen.add(url)
                    results.append(result)
        return results[:5]
