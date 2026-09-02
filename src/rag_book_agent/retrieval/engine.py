import hashlib
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List

from rag_book_agent.models import Chunk, SearchResult
from rag_book_agent.query import QuestionProcessor
from rag_book_agent.retrieval.text import fts_query, tokens
from rag_book_agent.storage import Storage
from rag_book_agent.vector_store import ChromaVectorStore


class HashEmbedding:
    """Small deterministic embedding used before an optional BGE model is installed."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def encode(self, text: str) -> Dict[int, float]:
        counts = Counter(tokens(text))
        vector = defaultdict(float)
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "little")
            index = number % self.dimensions
            sign = 1.0 if number & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm:
            return {key: value / norm for key, value in vector.items()}
        return {}

    @staticmethod
    def similarity(left: Dict[int, float], right: Dict[int, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(key, 0.0) for key, value in left.items())


class LightweightReranker:
    def score(self, question: str, text: str, heading: str = "") -> float:
        query_tokens = set(tokens(question))
        if not query_tokens:
            return 0.0
        text_tokens = set(tokens(text))
        heading_tokens = set(tokens(heading))
        coverage = len(query_tokens & text_tokens) / len(query_tokens)
        heading_coverage = len(query_tokens & heading_tokens) / len(query_tokens)
        exact_bonus = 0.3 if question.lower() in text.lower() else 0.0
        return coverage + 0.35 * heading_coverage + exact_bonus


class CrossEncoderReranker(LightweightReranker):
    def __init__(self, settings):
        self.settings = settings
        self.model = None

    def score(self, question: str, text: str, heading: str = "") -> float:
        if not self.settings.reranker_model:
            return super().score(question, text, heading)
        try:
            if self.model is None:
                from sentence_transformers import CrossEncoder

                self.model = CrossEncoder(
                    self.settings.resolve_model_path(self.settings.reranker_model),
                    local_files_only=self.settings.reranker_local_files_only,
                )
            return float(self.model.predict([(question, heading + "\n" + text)])[0])
        except (ImportError, OSError, RuntimeError, ValueError):
            return super().score(question, text, heading)


class HybridRetriever:
    def __init__(
        self,
        storage: Storage,
        sparse_top_k: int = 30,
        dense_top_k: int = 30,
        rerank_top_k: int = 12,
        rrf_k: int = 60,
    ):
        self.storage = storage
        self.sparse_top_k = sparse_top_k
        self.dense_top_k = dense_top_k
        self.rerank_top_k = rerank_top_k
        self.rrf_k = rrf_k
        self.embedding = HashEmbedding()
        self.reranker = CrossEncoderReranker(storage.settings)
        self.query_processor = QuestionProcessor()
        self.vector_store = None
        self.last_trace = {}
        try:
            self.vector_store = ChromaVectorStore(storage.settings)
        except (ImportError, RuntimeError, OSError):
            self.vector_store = None

    def search(
        self, question: str, limit: int = 6, document_ids: List[int] = None
    ) -> List[SearchResult]:
        question = question.strip()
        if not question:
            return []

        retrieval_question = self.query_processor.process(question)

        all_rows = self.storage.list_chunks(document_ids, active_only=True)
        sparse_rows = self._sparse_search(retrieval_question, all_rows, document_ids)
        dense_rows = self._dense_search(retrieval_question, all_rows, document_ids)

        candidates = {}
        fusion_scores = defaultdict(float)
        sparse_scores = {}
        dense_scores = {}

        for rank, row in enumerate(sparse_rows, start=1):
            chunk_id = int(row["id"])
            candidates[chunk_id] = row
            fusion_scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            sparse_scores[chunk_id] = 1.0 / rank

        for rank, item in enumerate(dense_rows, start=1):
            row, score = item
            chunk_id = int(row["id"])
            candidates[chunk_id] = row
            fusion_scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            dense_scores[chunk_id] = score

        fused_ids = sorted(fusion_scores, key=fusion_scores.get, reverse=True)
        fused_ids = fused_ids[: self.rerank_top_k]
        results = []
        for chunk_id in fused_ids:
            row = candidates[chunk_id]
            chunk = self._row_to_chunk(row)
            if chunk.parent_id:
                parent = self.storage.get_chunk(chunk.parent_id)
                if parent is not None:
                    chunk.text = parent["text"]
            result = SearchResult(
                chunk=chunk,
                document_title=row["document_title"],
                document_path=row["document_path"],
                sparse_score=sparse_scores.get(chunk_id, 0.0),
                dense_score=dense_scores.get(chunk_id, 0.0),
                fusion_score=fusion_scores[chunk_id],
            )
            result.rerank_score = self.reranker.score(retrieval_question, chunk.text, chunk.heading)
            results.append(result)

        results.sort(key=lambda item: (item.rerank_score, item.fusion_score), reverse=True)
        final = results[:limit]
        self.last_trace = {
            "query": question,
            "retrieval_query": retrieval_question,
            "candidate_counts": {"all_active_chunks": len(all_rows), "bm25": len(sparse_rows), "dense": len(dense_rows), "rrf": len(fused_ids), "final": len(final)},
            "bm25_candidates": [int(row["id"]) for row in sparse_rows[:12]],
            "dense_candidates": [{"id": int(row["id"]), "score": round(score, 4)} for row, score in dense_rows[:12]],
            "reranked": [{"id": item.chunk.id, "parent_id": item.chunk.parent_id, "score": round(item.rerank_score, 4), "fusion": round(item.fusion_score, 4)} for item in final],
        }
        return final

    @staticmethod
    def _expand_query(question: str) -> str:
        """Add stable bilingual terms for common Chinese questions over English corpora."""
        terms = {
            "发布": "publish upload release distribution",
            "打包": "build package packaging distribution",
            "安装": "install package",
            "依赖": "dependencies requirements",
            "概念": "concepts terminology glossary",
            "教程": "tutorial guide",
            "构建": "build package",
            "上传": "upload publish",
            "Python包": "Python package",
            "python包": "Python package",
        }
        expanded = [question]
        for source, target in terms.items():
            if source in question:
                expanded.append(target)
        return " ".join(expanded)

    def _sparse_search(self, question: str, all_rows: Iterable, document_ids: List[int]) -> List:
        query = fts_query(question)
        fts_rows = self.storage.search_fts(query, self.sparse_top_k, document_ids) if query else []
        found = {int(row["id"]) for row in fts_rows}

        query_tokens = set(tokens(question))
        fallback = []
        for row in all_rows:
            if int(row["id"]) in found:
                continue
            document_tokens = set(tokens(row["heading"] + " " + row["text"]))
            score = len(query_tokens & document_tokens)
            if score:
                fallback.append((score, row))
        fallback.sort(key=lambda item: item[0], reverse=True)
        fts_rows.extend(row for _, row in fallback[: self.sparse_top_k - len(fts_rows)])
        return fts_rows[: self.sparse_top_k]

    def _dense_search(
        self, question: str, all_rows: Iterable, document_ids: List[int] = None
    ) -> List:
        if self.vector_store is not None:
            try:
                rows = {int(row["id"]): row for row in all_rows}
                found = self.vector_store.search(question, self.dense_top_k)
                return [(rows[chunk_id], score) for chunk_id, score in found if chunk_id in rows]
            except (RuntimeError, ValueError, OSError):
                pass
        query_vector = self.embedding.encode(question)
        scored = []
        for row in all_rows:
            text = row["heading"] + "\n" + row["text"]
            score = self.embedding.similarity(query_vector, self.embedding.encode(text))
            if score > 0:
                scored.append((row, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: self.dense_top_k]

    @staticmethod
    def _row_to_chunk(row) -> Chunk:
        return Chunk(
            id=int(row["id"]),
            document_id=int(row["document_id"]),
            text=row["text"],
            heading=row["heading"],
            page_start=int(row["page_start"]),
            page_end=int(row["page_end"]),
            position=int(row["position"]),
            parent_id=int(row["parent_id"]) if row["parent_id"] is not None else None,
            chunk_type=row["chunk_type"] if "chunk_type" in row.keys() else "child",
        )
