from pathlib import Path

from rag_book_agent.config import Settings
from rag_book_agent.evaluation import Evaluator
from rag_book_agent.service import RagService


def make_service(tmp_path: Path) -> RagService:
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        chunk_size=300,
        chunk_overlap=30,
        sparse_top_k=10,
        dense_top_k=10,
        rerank_top_k=6,
        context_top_k=3,
    )
    return RagService(settings)


def test_ingest_search_answer_and_feedback(tmp_path):
    book = tmp_path / "book.md"
    book.write_text(
        "# 混合检索\n\n向量检索适合语义相似问题。BM25 适合关键词精确匹配。\n\n"
        "## 重排\n\n重排模型会重新判断问题和候选段落的相关性。",
        encoding="utf-8",
    )
    service = make_service(tmp_path)
    try:
        first = service.ingest(book)
        second = service.ingest(book)
        results = service.search("为什么需要重排模型")
        answer = service.ask("为什么需要重排模型")
        service.add_feedback("helpful")

        assert first["imported"] == 1
        assert first["chunks"] == 2
        assert second["skipped"] == 1
        assert results
        assert "重排" in results[0].chunk.text
        assert answer.sources
        assert "[S1]" in answer.text
        feedback_count = service.storage.connection.execute(
            "SELECT COUNT(*) FROM feedback"
        ).fetchone()[0]
        assert feedback_count == 1
    finally:
        service.close()


def test_replacing_changed_document_removes_old_chunks(tmp_path):
    book = tmp_path / "book.md"
    book.write_text("# 旧内容\n\n只存在于旧版本的词语。", encoding="utf-8")
    service = make_service(tmp_path)
    try:
        service.ingest(book)
        book.write_text("# 新内容\n\n这是替换后的新版本。", encoding="utf-8")
        service.ingest(book)

        stored_text = "\n".join(row["text"] for row in service.storage.list_chunks())
        assert "旧版本的词语" not in stored_text
        assert service.search("替换后的新版本")
        assert service.stats()["documents"] == 1
    finally:
        service.close()


def test_evaluation_reports_recall_and_mrr(tmp_path):
    book = tmp_path / "book.md"
    book.write_text("# 引用\n\n每个事实都应该附带可以验证的引用。", encoding="utf-8")
    service = make_service(tmp_path)
    try:
        service.ingest(book)
        chunk_id = service.storage.list_chunks()[0]["id"]
        service.storage.add_golden_question("事实为什么要有引用", [chunk_id])
        report = Evaluator(service.storage, service.retriever).run(top_k=5)

        assert report["question_count"] == 1
        assert report["recall_at_k"] == 1.0
        assert report["mrr_at_k"] == 1.0
    finally:
        service.close()


def test_search_can_be_limited_to_one_document(tmp_path):
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("# Alpha\n\nReranking sorts relevant evidence.", encoding="utf-8")
    second.write_text("# Beta\n\nReranking is mentioned in unrelated notes.", encoding="utf-8")
    service = make_service(tmp_path)
    try:
        service.ingest(first)
        service.ingest(second)
        first_document_id = service.storage.list_chunks()[0]["document_id"]
        results = service.search("What is reranking?", document_ids=[first_document_id])

        assert results
        assert all(result.chunk.document_id == first_document_id for result in results)
    finally:
        service.close()
