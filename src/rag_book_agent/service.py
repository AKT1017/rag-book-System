import time
from pathlib import Path
from typing import Dict, List, Tuple

from rag_book_agent.chunking import ChapterChunker
from rag_book_agent.config import Settings
from rag_book_agent.generation import AnswerGenerator
from rag_book_agent.ingest import DocumentLoader
from rag_book_agent.memory import ConversationMemory
from rag_book_agent.models import Answer, SearchResult
from rag_book_agent.retrieval import HybridRetriever
from rag_book_agent.storage import Storage
from rag_book_agent.web_search import WebSearch


class RagService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.storage = Storage(settings.database_path)
        self.storage.settings = settings
        self.loader = DocumentLoader()
        self.chunker = ChapterChunker(
            settings.child_chunk_size, settings.chunk_overlap, settings.parent_chunk_size
        )
        self.retriever = HybridRetriever(
            self.storage,
            sparse_top_k=settings.sparse_top_k,
            dense_top_k=settings.dense_top_k,
            rerank_top_k=settings.rerank_top_k,
            rrf_k=settings.rrf_k,
        )
        self.generator = AnswerGenerator(settings)
        self.web_search = WebSearch(settings)
        self.last_trace_id = 0

    def close(self) -> None:
        self.storage.close()

    def ingest(self, input_path: Path) -> Dict[str, int]:
        found = self.loader.find_documents(input_path)
        imported = 0
        skipped = 0
        chunks_added = 0

        for path in found:
            document = self.loader.load(path)
            if self.storage.document_is_current(document.path, document.content_hash):
                skipped += 1
                continue
            chunks = self.chunker.split(document)
            document_id = self.storage.replace_document(document, chunks)
            try:
                from rag_book_agent.vector_store import ChromaVectorStore

                vector_store = ChromaVectorStore(self.settings)
                stored_chunks = self.storage.list_chunks([document_id])
                vector_store.upsert([self.retriever._row_to_chunk(row) for row in stored_chunks])
            except (ImportError, RuntimeError, OSError):
                pass
            imported += 1
            chunks_added += sum(1 for chunk in chunks if chunk.chunk_type == "child")

        return {
            "found": len(found),
            "imported": imported,
            "skipped": skipped,
            "chunks": chunks_added,
        }

    def search(
        self, question: str, limit: int = 6, document_ids: List[int] = None
    ) -> List[SearchResult]:
        return self.retriever.search(question, limit=limit, document_ids=document_ids)

    def ask(self, question: str, session_id: str = "default", force_web: bool = False) -> Answer:
        return self._ask(question, session_id=session_id, force_web=force_web)

    def ask_agent(
        self, question: str, session_id: str = "default", force_web: bool = False
    ) -> Answer:
        from rag_book_agent.agent import AgentOrchestrator

        return AgentOrchestrator(self).run(question, session_id, force_web)

    def ask_for_documents(self, question: str, document_ids: List[int]) -> Answer:
        return self._ask(question, document_ids)

    def _ask(
        self,
        question: str,
        document_ids: List[int] = None,
        session_id: str = "default",
        force_web: bool = False,
        precomputed_sources=None,
        precomputed_web=None,
        agent_mode=False,
        agent_plan=None,
    ) -> Answer:
        start = time.perf_counter()
        route = self.retriever.query_processor.route(question)
        if route.cached_answer:
            answer = Answer(text=route.cached_answer, sources=[], mode="rule-cache")
            self.last_retrieval = {
                "route_layer": route.layer, "route_action": route.action,
                "route_score": round(route.score, 4), "route_reason": route.reason,
            }
            memory = ConversationMemory(self.settings.database_path.parent, session_id)
            memory.add(question, answer.text, [], [])
            self.last_trace_id = self.storage.add_trace(question, answer.text, answer.mode, [], 0)
            return answer
        sources = precomputed_sources
        if sources is None:
            sources = self.search(
                question, limit=self.settings.context_top_k, document_ids=document_ids
            )
        web_results = precomputed_web
        if web_results is None:
            web_results = self.web_search.search(question) if self.settings.web_search_enabled else []
        self.last_web_results = web_results
        memory = ConversationMemory(self.settings.database_path.parent, session_id)
        answer = self.generator.answer(
            question, sources, web_results, memory.context(), force_web, agent_mode, agent_plan
        )
        if force_web and self.settings.deepseek_web_search:
            if not self.settings.api_key:
                web_status = "deepseek-unavailable-no-api-key"
            elif answer.mode == "extractive-fallback":
                web_status = "deepseek-request-failed-local-fallback"
            else:
                web_status = "deepseek-enabled"
        elif self.settings.web_search_enabled and web_results:
            web_status = "local-web-search"
        else:
            web_status = "off"
        self.last_retrieval = {
            "route_layer": route.layer,
            "route_action": route.action,
            "route_score": round(route.score, 4),
            "route_reason": route.reason,
            "sparse": any(source.sparse_score for source in sources),
            "dense": any(source.dense_score for source in sources),
            "rrf": any(source.fusion_score for source in sources),
            "rerank": any(source.rerank_score for source in sources),
            "web_search": web_status not in {"off", "deepseek-unavailable-no-api-key"},
            "web_search_requested": bool(force_web),
            "web_search_status": web_status,
            "web_search_provider": self.web_search.last_provider,
            "web_search_mode": (
                "forced" if force_web else ("auto" if self.settings.deepseek_web_search else "off")
            ),
        }
        saved_sources = [
            {
                "id": source.chunk.id,
                "title": source.document_title,
                "page": source.chunk.page_start,
                "heading": source.chunk.heading,
                "text": source.chunk.text,
                "score": round(source.rerank_score, 4),
            }
            for source in answer.sources
        ]
        memory.add(question, answer.text, saved_sources, web_results)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        source_ids = [item.chunk.id for item in answer.sources if item.chunk.id is not None]
        self.last_trace_id = self.storage.add_trace(
            question, answer.text, answer.mode, source_ids, elapsed_ms
        )
        return answer

    def add_feedback(self, rating: str, note: str = "") -> None:
        if not self.last_trace_id:
            raise ValueError("No query is available for feedback")
        self.storage.add_feedback(self.last_trace_id, rating, note)

    def stats(self) -> Dict[str, int]:
        return self.storage.stats()

    def health(self) -> List[Tuple[str, bool, str]]:
        checks = []
        checks.append(
            ("database", self.settings.database_path.exists(), str(self.settings.database_path))
        )
        fts5 = False
        try:
            self.storage.connection.execute("SELECT count(*) FROM chunks_fts").fetchone()
            fts5 = True
        except Exception:
            pass
        checks.append(("sqlite-fts5", fts5, "full text search"))
        api_ready = bool(self.settings.api_base and self.settings.api_model)
        detail = self.settings.api_model if api_ready else "extractive mode"
        checks.append(("generation", True, detail))
        return checks
