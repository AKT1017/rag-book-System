from rag_book_agent.chunking import ChapterChunker
from rag_book_agent.models import Page, ParsedDocument


def test_markdown_headings_are_preserved():
    document = ParsedDocument(
        path="book.md",
        title="book",
        file_type="markdown",
        content_hash="abc",
        pages=[Page(1, "# 第一章\n\n第一段内容。\n\n## 第二节\n\n第二段内容。")],
    )
    chunks = ChapterChunker(chunk_size=200, overlap=20).split(document)

    assert len(chunks) == 2
    assert chunks[0].heading == "第一章"
    assert chunks[1].heading == "第二节"
    assert chunks[0].page_start == 1


def test_long_text_is_split_without_empty_chunks():
    text = "这是一个很长的句子。" * 80
    document = ParsedDocument("book.txt", "book", "text", "abc", [Page(1, text)])
    chunks = ChapterChunker(chunk_size=220, overlap=30).split(document)

    assert len(chunks) > 2
    assert all(chunk.text for chunk in chunks)
    assert all(len(chunk.text) <= 220 for chunk in chunks)


def test_chinese_chapter_title_is_inferred_without_markdown():
    document = ParsedDocument(
        "book.pdf", "book", "pdf", "abc",
        [Page(1, "第一章 检索基础\n\n这里是正文内容，用于说明检索的基本概念。")],
    )
    chunks = ChapterChunker(chunk_size=220, overlap=20).split(document)
    assert chunks[0].heading == "第一章 检索基础"


def test_chunker_creates_parent_and_child_chunks():
    document = ParsedDocument("book.md", "book", "markdown", "abc", [Page(1, "# A\n\n" + "正文内容。" * 500)])
    chunks = ChapterChunker(chunk_size=220, overlap=20, parent_size=500).split(document)
    assert any(chunk.chunk_type == "parent" for chunk in chunks)
    assert any(chunk.chunk_type == "child" and chunk.parent_id is not None for chunk in chunks)
