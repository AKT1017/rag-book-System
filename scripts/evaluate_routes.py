"""Compare sparse, dense, hybrid and reranked retrieval on the CMRC set."""

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_book_agent.config import load_settings  # noqa: E402
from rag_book_agent.service import RagService  # noqa: E402


def metrics(rows):
    output = {}
    for k in (5, 10, 20):
        recalls = []
        reciprocal = []
        hits = []
        for expected, found in rows:
            ranked = found[:k]
            hit = any(chunk_id in expected for chunk_id in ranked)
            hits.append(1.0 if hit else 0.0)
            recalls.append(len(expected.intersection(ranked)) / len(expected))
            reciprocal.append(next((1.0 / rank for rank, chunk_id in enumerate(ranked, 1) if chunk_id in expected), 0.0))
        output["recall@%d" % k] = round(statistics.mean(recalls), 4)
        output["mrr@%d" % k] = round(statistics.mean(reciprocal), 4)
        output["hit_rate@%d" % k] = round(statistics.mean(hits), 4)
    return output


def main():
    settings = load_settings(ROOT)
    service = RagService(settings)
    try:
        questions = service.storage.list_golden_questions()
        routes = {"bm25": [], "dense": [], "hybrid": [], "hybrid_rerank": []}
        timings = {name: [] for name in routes}
        for row in questions:
            expected = set(json.loads(row["expected_chunk_ids"]))
            if not expected:
                continue
            query = service.retriever.query_processor.process(row["question"])
            all_rows = service.storage.list_chunks(active_only=True)
            started = time.perf_counter()
            sparse = service.retriever._sparse_search(query, all_rows, None)
            timings["bm25"].append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            dense = service.retriever._dense_search(query, all_rows, None)
            timings["dense"].append((time.perf_counter() - started) * 1000)
            sparse_ids = [int(item["id"]) for item in sparse]
            dense_ids = [int(item[0]["id"]) for item in dense]
            fused = []
            scores = {}
            for rank, chunk_id in enumerate(sparse_ids, 1):
                scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (service.retriever.rrf_k + rank)
            for rank, chunk_id in enumerate(dense_ids, 1):
                scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (service.retriever.rrf_k + rank)
            fused = sorted(scores, key=scores.get, reverse=True)
            routes["bm25"].append((expected, sparse_ids))
            routes["dense"].append((expected, dense_ids))
            routes["hybrid"].append((expected, fused))
            started = time.perf_counter()
            reranked = service.search(row["question"], limit=20)
            timings["hybrid_rerank"].append((time.perf_counter() - started) * 1000)
            routes["hybrid_rerank"].append((expected, [item.chunk.id for item in reranked]))
        latency = {name: round(statistics.median(values), 2) if values else None for name, values in timings.items()}
        report = {"dataset": "CMRC2018", "questions": len(routes["bm25"]), "routes": {name: metrics(rows) for name, rows in routes.items()}, "latency_ms": latency}
        output = ROOT / "data" / "reports" / "route-evaluation.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
