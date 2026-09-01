"""Prepare a moderate CMRC2018 Chinese QA RAG evaluation subset."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "cmrc2018_train.json"
OUT_DOC = ROOT / "data" / "cmrc2018_eval" / "cmrc2018_contexts.md"
OUT_QA = ROOT / "data" / "cmrc2018_eval" / "questions.jsonl"
LIMIT = 300


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    contexts = []
    questions = []
    seen = set()
    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()
            if not context or context in seen:
                continue
            seen.add(context)
            contexts.append(context)
            for qa in paragraph["qas"]:
                answer = qa["answers"][0]["text"]
                questions.append({"question": qa["question"], "reference_answer": answer, "evidence": answer})
                if len(questions) >= LIMIT:
                    break
            if len(questions) >= LIMIT:
                break
        if len(questions) >= LIMIT:
            break
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.write_text("\n\n".join("## 文章 %d\n\n%s" % (i + 1, text) for i, text in enumerate(contexts)), encoding="utf-8")
    OUT_QA.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in questions) + "\n", encoding="utf-8")
    print(json.dumps({"contexts": len(contexts), "questions": len(questions), "document": str(OUT_DOC)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
