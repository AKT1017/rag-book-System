import json
from pathlib import Path
from typing import Dict

from rag_book_agent.retrieval import HybridRetriever
from rag_book_agent.storage import Storage


class Evaluator:
    def __init__(self, storage: Storage, retriever: HybridRetriever):
        self.storage = storage
        self.retriever = retriever

    def run(self, top_k: int = 10) -> Dict[str, object]:
        rows = self.storage.list_golden_questions()
        details = []
        recall_total = 0.0
        reciprocal_rank_total = 0.0

        for row in rows:
            expected = set(json.loads(row["expected_chunk_ids"]))
            # Negative questions are evaluated separately as refusal tests.
            if not expected:
                details.append(
                    {
                        "question": row["question"],
                        "expected": [],
                        "found": [],
                        "recall": None,
                        "reciprocal_rank": None,
                        "evaluation": "refusal_only",
                    }
                )
                continue
            results = self.retriever.search(row["question"], limit=top_k)
            found_ids = [result.chunk.id for result in results]
            found = expected.intersection(found_ids)
            recall = len(found) / len(expected) if expected else 0.0
            reciprocal_rank = 0.0
            for rank, chunk_id in enumerate(found_ids, start=1):
                if chunk_id in expected:
                    reciprocal_rank = 1.0 / rank
                    break

            recall_total += recall
            reciprocal_rank_total += reciprocal_rank
            details.append(
                {
                    "question": row["question"],
                    "expected": sorted(expected),
                    "found": found_ids,
                    "recall": round(recall, 4),
                    "reciprocal_rank": round(reciprocal_rank, 4),
                }
            )

        count = sum(1 for row in rows if json.loads(row["expected_chunk_ids"]))
        return {
            "question_count": count,
            "recall_at_k": round(recall_total / count, 4) if count else 0.0,
            "mrr_at_k": round(reciprocal_rank_total / count, 4) if count else 0.0,
            "top_k": top_k,
            "details": details,
        }

    @staticmethod
    def save(report: Dict[str, object], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
