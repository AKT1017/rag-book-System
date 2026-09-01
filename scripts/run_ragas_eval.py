"""Run LLM-based RAGAS evaluation in the separate .venv-ragas environment."""

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rag_book_agent.config import load_settings  # noqa: E402
from rag_book_agent.service import RagService  # noqa: E402


def build_dataset(service: RagService, limit: int) -> Dataset:
    rows = service.storage.list_golden_questions()[:limit]
    samples = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for row in rows:
        question = row["question"]
        expected_chunk_ids = json.loads(row["expected_chunk_ids"])
        if not expected_chunk_ids:
            continue
        document_ids = []
        for chunk_id in expected_chunk_ids:
            chunk = service.storage.get_chunk(chunk_id)
            if chunk is not None:
                document_ids.append(int(chunk["document_id"]))
        if not document_ids:
            raise ValueError("Golden question has no valid expected chunk IDs: %s" % question)
        answer = service.ask_for_documents(question, sorted(set(document_ids)))
        samples["question"].append(question)
        samples["answer"].append(answer.text)
        samples["contexts"].append([source.chunk.text for source in answer.sources])
        samples["ground_truth"].append(row["reference_answer"])
    return Dataset.from_dict(samples)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate the local RAG system with RAGAS.")
    parser.add_argument(
        "--limit", type=int, default=1, help="Maximum golden questions to evaluate."
    )
    parser.add_argument(
        "--output", default="data/reports/ragas-latest.json", help="Path relative to the project."
    )
    args = parser.parse_args()

    settings = load_settings(PROJECT_DIR)
    if not settings.api_base or not settings.api_model or not settings.api_key:
        raise SystemExit("DeepSeek API base, model, and RAG_BOOK_API_KEY are required for RAGAS.")

    service = RagService(settings)
    try:
        dataset = build_dataset(service, args.limit)
        if not len(dataset):
            raise SystemExit("No golden questions. Add one with rag-book golden-add first.")

        judge = ChatOpenAI(
            model=settings.api_model,
            api_key=settings.api_key,
            base_url=settings.api_base,
            temperature=0,
            max_retries=1,
            request_timeout=settings.request_timeout,
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, context_precision],
            llm=judge,
            raise_exceptions=True,
        )
        report = result.to_pandas().to_dict(orient="records")
        output = Path(args.output)
        if not output.is_absolute():
            output = PROJECT_DIR / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("Report: %s" % output)
    finally:
        service.close()


if __name__ == "__main__":
    main()
