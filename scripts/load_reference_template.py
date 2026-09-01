"""Load the reference handbook and register its golden evaluation questions."""

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rag_book_agent.config import load_settings  # noqa: E402
from rag_book_agent.service import RagService  # noqa: E402


def main() -> None:
    settings = load_settings(PROJECT_DIR)
    service = RagService(settings)
    handbook = PROJECT_DIR / "samples" / "reference_handbook.md"
    questions_path = PROJECT_DIR / "samples" / "reference_questions.jsonl"
    try:
        result = service.ingest(handbook)
        chunks = service.storage.list_chunks()
        rows = []
        for line in questions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        added = 0
        for row in rows:
            ids = [
                int(chunk["id"])
                for chunk in chunks
                if row["evidence"] and row["evidence"] in chunk["text"]
            ]
            service.storage.add_golden_question(row["question"], ids, row["reference_answer"])
            added += 1
        print(
            json.dumps({"ingest": result, "golden_questions": added}, ensure_ascii=False, indent=2)
        )
    finally:
        service.close()


if __name__ == "__main__":
    main()
