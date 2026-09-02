"""Rebuild the small, version-controlled knowledge base and its golden set."""

import json
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rag_book_agent.config import load_settings  # noqa: E402
from rag_book_agent.service import RagService  # noqa: E402


def clear_data(data_dir: Path) -> None:
    """Delete only the runtime data directory, never source samples or models."""
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    settings = load_settings(PROJECT_DIR)
    data_dir = Path(settings.data_dir)
    clear_data(data_dir)

    service = RagService(settings)
    try:
        imports = []
        paths = [PROJECT_DIR / "samples" / "reference_handbook.md", PROJECT_DIR / "samples" / "rag_introduction.md"]
        cmrc_doc = PROJECT_DIR / "samples" / "cmrc2018_eval" / "cmrc2018_contexts.md"
        if cmrc_doc.exists():
            paths.append(cmrc_doc)
        for path in paths:
            imports.append(service.ingest(path))

        question_paths = [PROJECT_DIR / "samples" / "reference_questions.jsonl"]
        cmrc_questions = PROJECT_DIR / "samples" / "cmrc2018_eval" / "questions.jsonl"
        if cmrc_questions.exists():
            question_paths.append(cmrc_questions)
        child_chunks = service.storage.list_chunks(active_only=True)
        added = 0
        for questions_path in question_paths:
            for line in questions_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                expected = [
                    int(chunk["id"])
                    for chunk in child_chunks
                    if row.get("evidence") and row["evidence"] in chunk["text"]
                ]
                service.storage.add_golden_question(row["question"], expected, row.get("reference_answer", ""))
                added += 1
        print(json.dumps({"imports": imports, "golden_questions": added, "stats": service.stats()}, ensure_ascii=False, indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
