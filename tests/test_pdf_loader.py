from pathlib import Path

from pypdf import PdfWriter

from rag_book_agent.ingest.loaders import DocumentLoader


def test_pdf_loader_keeps_page_numbers_and_falls_back(tmp_path: Path):
    path = tmp_path / "book.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    with path.open("wb") as handle:
        writer.write(handle)

    document = DocumentLoader().load(path)
    assert document.file_type == "pdf"
    assert len(document.pages) == 1
    assert document.pages[0].number == 1
