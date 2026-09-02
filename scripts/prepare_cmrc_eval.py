"""Prepare a moderate CMRC2018 Chinese QA RAG evaluation subset."""

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "samples" / "cmrc2018_train.json"
OUT_DOC = ROOT / "samples" / "cmrc2018_eval" / "cmrc2018_contexts.md"
OUT_QA = ROOT / "samples" / "cmrc2018_eval" / "questions.jsonl"
LIMIT = 300
SOURCE_URL = "https://raw.githubusercontent.com/ymcui/cmrc2018/master/data/cmrc2018_train.json"


def main() -> None:
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading CMRC2018 from the public project...")
        urllib.request.urlretrieve(SOURCE_URL, SOURCE)
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    contexts = []
    questions = []
    seen = set()
    articles = payload.get("data", []) if isinstance(payload, dict) else payload
    for article in articles:
        context = article.get("context_text", article.get("context", "")).strip()
        if not context or context in seen:
            continue
        seen.add(context)
        contexts.append(context)
        for qa in article.get("qas", []):
            answers = qa.get("answers", [])
            answer = answers[0].get("text", "") if answers and isinstance(answers[0], dict) else (answers[0] if answers else "")
            question = qa.get("query_text", qa.get("question", ""))
            if question and answer:
                questions.append({"question": question, "reference_answer": answer, "evidence": answer})
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
