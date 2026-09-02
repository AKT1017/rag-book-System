"""A bounded LangGraph workflow that reuses the project's RAG tools."""

from typing import Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from rag_book_agent.agent.orchestrator import AgentOrchestrator
from rag_book_agent.models import Answer, SearchResult


class ResearchState(TypedDict, total=False):
    question: str
    session_id: str
    force_web: bool
    plan: List[str]
    local_results: List[SearchResult]
    web_results: List[dict]
    trace: List[dict]
    answer: Answer


class LangGraphAgent:
    """Explicit graph implementation kept separate from the stable agent workflow."""

    def __init__(self, service):
        self.service = service
        self.orchestrator = AgentOrchestrator(service)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ResearchState)
        workflow.add_node("plan", self._plan)
        workflow.add_node("local_research", self._local_research)
        workflow.add_node("web_research", self._web_research)
        workflow.add_node("deduplicate", self._deduplicate)
        workflow.add_node("synthesize", self._synthesize)
        workflow.add_edge(START, "plan")
        workflow.add_edge("plan", "local_research")
        workflow.add_edge("local_research", "web_research")
        workflow.add_edge("web_research", "deduplicate")
        workflow.add_edge("deduplicate", "synthesize")
        workflow.add_edge("synthesize", END)
        return workflow.compile()

    def _plan(self, state: ResearchState) -> Dict:
        plan = self.orchestrator.plan(state["question"])["sub_questions"]
        return {"plan": plan, "trace": [{"node": "plan", "sub_questions": len(plan)}]}

    def _local_research(self, state: ResearchState) -> Dict:
        results = []
        trace = list(state.get("trace", []))
        for question in state["plan"]:
            found = self.orchestrator.tools.local_search(question)
            results.extend(found)
            trace.append({"node": "local_research", "query": question, "results": len(found)})
        return {"local_results": results, "trace": trace}

    def _web_research(self, state: ResearchState) -> Dict:
        results = []
        trace = list(state.get("trace", []))
        for question in state["plan"]:
            found = self.orchestrator.tools.web_search(question)
            results.extend(found)
            trace.append({"node": "web_research", "query": question, "results": len(found)})
        return {"web_results": results, "trace": trace}

    @staticmethod
    def _deduplicate(state: ResearchState) -> Dict:
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
        trace = list(state.get("trace", []))
        trace.append({"node": "deduplicate", "local": len(local_by_parent), "web": len(web_by_url)})
        return {"local_results": list(local_by_parent.values())[:6], "web_results": list(web_by_url.values())[:5], "trace": trace}

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
        self.service.last_retrieval["agent"] = True
        self.service.last_retrieval["agent_engine"] = "langgraph"
        self.service.last_retrieval["agent_roles"] = ["planner", "researcher", "synthesizer"]
        self.service.last_retrieval["agent_tools"] = list(self.orchestrator.tools.registry.keys())
        self.service.last_retrieval["agent_sub_questions"] = state.get("plan", [])
        self.service.last_retrieval["agent_research"] = state.get("trace", [])
        self.service.last_retrieval["agent_react_steps"] = len(state.get("plan", []))
        self.service.last_retrieval["agent_web_provider"] = self.service.web_search.last_provider
        return state["answer"]

    def draw_png(self) -> bytes:
        """Use LangGraph's official graph renderer for the UI diagram."""
        return self.graph.get_graph().draw_mermaid_png()
