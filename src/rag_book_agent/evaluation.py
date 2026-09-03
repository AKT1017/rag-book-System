"""Deterministic, model-free retrieval evaluation and route ablation."""

import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from rag_book_agent.retrieval import HybridRetriever
from rag_book_agent.storage import Storage


class Evaluator:
    ROUTES = ("hybrid_rerank", "hybrid", "bm25", "dense")

    def __init__(self, storage: Storage, retriever: HybridRetriever):
        self.storage = storage
        self.retriever = retriever

    def run(self, top_k=10, max_questions=None, dataset="all", route="hybrid_rerank"):
        if route not in self.ROUTES:
            raise ValueError("Unsupported evaluation route: %s" % route)
        rows = self.storage.list_golden_questions(dataset)
        if max_questions:
            rows = rows[:max_questions]
        details, metric_rows, latencies = [], [], []
        negative_count = 0
        for row in rows:
            expected = set(json.loads(row["expected_chunk_ids"]))
            if not expected:
                negative_count += 1
                details.append({"question": row["question"], "dataset": row["dataset"],
                                "expected": [], "found": [], "evaluation": "refusal_only"})
                continue
            started = time.perf_counter()
            ranked = self._retrieve(row["question"], route, top_k)[:top_k]
            latencies.append((time.perf_counter() - started) * 1000)
            hit_ids = expected.intersection(ranked)
            recall = len(hit_ids) / len(expected)
            precision = len(hit_ids) / max(1, len(ranked))
            first_rank = next((rank for rank, item in enumerate(ranked, 1)
                               if item in expected), None)
            reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
            dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(ranked, 1)
                      if item in expected)
            ideal_hits = min(len(expected), top_k)
            ideal_dcg = sum(1.0 / math.log2(rank + 1)
                            for rank in range(1, ideal_hits + 1))
            ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
            values = {"recall": recall, "precision": precision,
                      "reciprocal_rank": reciprocal_rank, "ndcg": ndcg,
                      "hit": 1.0 if hit_ids else 0.0}
            metric_rows.append(values)
            details.append({
                "question": row["question"], "dataset": row["dataset"],
                "expected": sorted(expected), "found": ranked,
                "first_relevant_rank": first_rank, "recall": round(recall, 4),
                "precision": round(precision, 4),
                "reciprocal_rank": round(reciprocal_rank, 4),
                "ndcg": round(ndcg, 4), "evaluation": "retrieval",
            })
        return {
            "dataset": dataset, "route": route, "question_count": len(metric_rows),
            "negative_count": negative_count, "sampled": len(rows), "top_k": top_k,
            "recall_at_k": self._mean(metric_rows, "recall"),
            "precision_at_k": self._mean(metric_rows, "precision"),
            "mrr_at_k": self._mean(metric_rows, "reciprocal_rank"),
            "ndcg_at_k": self._mean(metric_rows, "ndcg"),
            "hit_rate_at_k": self._mean(metric_rows, "hit"),
            "latency_ms": {
                "mean": round(statistics.mean(latencies), 2) if latencies else 0.0,
                "p50": round(self._percentile(latencies, 50), 2),
                "p95": round(self._percentile(latencies, 95), 2),
            },
            "details": details,
        }

    def compare(self, top_k=10, max_questions=None, dataset="all"):
        reports = {}
        for route in self.ROUTES:
            report = self.run(top_k, max_questions, dataset, route)
            report.pop("details", None)
            reports[route] = report
        return {"dataset": dataset, "top_k": top_k,
                "sampled": max((item["sampled"] for item in reports.values()), default=0),
                "routes": reports}

    def _retrieve(self, question: str, route: str, top_k: int) -> List[int]:
        query = self.retriever.query_processor.process(question)
        rows = self.storage.list_chunks(active_only=True)
        if route == "hybrid_rerank":
            return [item.chunk.id for item in self.retriever.search(question, limit=top_k)]
        if route == "bm25":
            return [int(row["id"]) for row in
                    self.retriever._sparse_search(query, rows, None)[:top_k]]
        if route == "dense":
            return [int(row["id"]) for row, _ in
                    self.retriever._dense_search(query, rows)[:top_k]]
        scores = defaultdict(float)
        for rank, row in enumerate(self.retriever._sparse_search(query, rows, None), 1):
            scores[int(row["id"])] += 1.0 / (self.retriever.rrf_k + rank)
        for rank, (row, _) in enumerate(self.retriever._dense_search(query, rows), 1):
            scores[int(row["id"])] += 1.0 / (self.retriever.rrf_k + rank)
        return sorted(scores, key=scores.get, reverse=True)[:top_k]

    @staticmethod
    def _mean(rows, key):
        return round(statistics.mean(row[key] for row in rows), 4) if rows else 0.0

    @staticmethod
    def _percentile(values, percentile):
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
        return ordered[index]

    @staticmethod
    def save(report: Dict[str, object], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
