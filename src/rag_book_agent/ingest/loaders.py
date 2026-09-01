import hashlib
import re
from pathlib import Path
from typing import List

from pypdf import PdfReader

from rag_book_agent.models import Page, ParsedDocument


class UnsupportedDocumentError(ValueError):
    pass


class DocumentLoader:
    supported_extensions = {".pdf", ".md", ".markdown", ".txt"}

    def load(self, path: Path) -> ParsedDocument:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("Document does not exist: %s" % path)

        extension = path.suffix.lower()
        if extension not in self.supported_extensions:
            raise UnsupportedDocumentError("Unsupported document type: %s" % extension)

        content_hash = self._hash_file(path)
        if extension == ".pdf":
            pages = self._load_pdf(path)
            file_type = "pdf"
        else:
            pages = self._load_text(path)
            file_type = "markdown" if extension in {".md", ".markdown"} else "text"

        title = path.stem.replace("_", " ").strip()
        return ParsedDocument(
            path=str(path),
            title=title,
            file_type=file_type,
            content_hash=content_hash,
            pages=pages,
        )

    def find_documents(self, path: Path) -> List[Path]:
        path = path.expanduser().resolve()
        if path.is_file():
            return [path]
        if not path.is_dir():
            raise FileNotFoundError("Path does not exist: %s" % path)

        documents = []
        for item in path.rglob("*"):
            if item.is_file() and item.suffix.lower() in self.supported_extensions:
                documents.append(item)
        return sorted(documents)

    def _load_pdf(self, path: Path) -> List[Page]:
        # MarkItDown is the preferred document conversion layer when available.
        try:
            pages = self._load_pdf_markitdown(path)
            if any(page.text for page in pages):
                return pages
        except (ImportError, OSError, RuntimeError, ValueError):
            pass
        # PyMuPDF preserves blocks and reading order better for multi-column books.
        try:
            return self._load_pdf_pymupdf(path)
        except (ImportError, OSError, RuntimeError, ValueError):
            return self._load_pdf_pypdf(path)

    def _load_pdf_markitdown(self, path: Path) -> List[Page]:
        from markitdown import MarkItDown

        result = MarkItDown().convert(str(path))
        text = self._normalize(getattr(result, "text_content", "") or "")
        raw_pages = [part for part in text.split("\f") if part.strip()]
        if not raw_pages:
            raw_pages = [text]
        return [Page(number=index, text=part) for index, part in enumerate(raw_pages, start=1)]

    def _load_pdf_pymupdf(self, path: Path) -> List[Page]:
        import fitz

        document = fitz.open(str(path))
        pages = []
        try:
            for number, pdf_page in enumerate(document, start=1):
                blocks = pdf_page.get_text("blocks", sort=True)
                text_parts = []
                for block in blocks:
                    if len(block) >= 5 and block[4]:
                        text_parts.append(block[4])
                text = self._normalize("\n".join(text_parts))
                pages.append(Page(number=number, text=text))
        finally:
            document.close()
        return pages

    def _load_pdf_pypdf(self, path: Path) -> List[Page]:
        reader = PdfReader(str(path), strict=False)
        pages = []
        for number, pdf_page in enumerate(reader.pages, start=1):
            try:
                text = pdf_page.extract_text() or ""
            except (OSError, ValueError, RuntimeError):
                text = ""
            pages.append(Page(number=number, text=self._normalize(text)))
        return pages

    def _load_text(self, path: Path) -> List[Page]:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8-sig")
        return [Page(number=1, text=self._normalize(text))]

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Join words split at a line ending while keeping paragraph boundaries.
        text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()
