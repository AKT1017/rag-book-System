import re
from typing import List, Tuple

from rag_book_agent.models import Chunk, ParsedDocument


class ChapterChunker:
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

    def __init__(self, chunk_size: int = 1800, overlap: int = 220, parent_size: int = None):
        if chunk_size < 200:
            raise ValueError("chunk_size must be at least 200 characters")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be between 0 and chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.parent_size = parent_size

    def split(self, document: ParsedDocument) -> List[Chunk]:
        # Join pages before splitting so paragraphs, lists, and tables can cross page breaks.
        page_parts = []
        for page in document.pages:
            page_parts.append("<!-- PAGE_START: %d -->\n%s\n<!-- PAGE_END: %d -->" %
                              (page.number, page.text, page.number))
        full_text = "\n\n".join(page_parts)
        chunks = []
        position = 0
        current_heading = ""
        for heading, text in self._sections(full_text, current_heading):
            if heading:
                current_heading = heading
            parent_texts = ([text] if self.parent_size is None else
                            self._split_text(text, self.parent_size))
            for parent_text in parent_texts:
                if not parent_text.strip():
                    continue
                start_page, end_page = self._page_range(parent_text)
                clean_parent = self._clean_page_tags(parent_text).strip()
                parent_position = position if self.parent_size is not None else None
                if self.parent_size is not None:
                    chunks.append(Chunk(None, 0, clean_parent, current_heading,
                                        start_page, end_page, position, None, "parent"))
                    position += 1
                for piece in self._split_text(clean_parent, self.chunk_size):
                    if piece.strip():
                        chunks.append(Chunk(None, 0, piece.strip(), current_heading,
                                            start_page, end_page, position,
                                            parent_position, "child"))
                        position += 1
        return chunks

    @staticmethod
    def _page_range(text: str) -> Tuple[int, int]:
        pages = [int(value) for value in re.findall(r"<!-- PAGE_(?:START|END): (\d+) -->", text)]
        return (min(pages), max(pages)) if pages else (1, 1)

    @staticmethod
    def _clean_page_tags(text: str) -> str:
        return re.sub(r"<!-- PAGE_(?:START|END): \d+ -->\s*", "", text)

    def _sections(self, text: str, initial_heading: str) -> List[Tuple[str, str]]:
        sections = []
        heading = initial_heading
        buffer = []

        for line in text.splitlines():
            match = self.heading_pattern.match(line.strip())
            if not match and self._looks_like_heading(line, buffer):
                match = re.match(r"^(.+)$", line.strip())
            if match:
                if buffer:
                    sections.append((heading, "\n".join(buffer).strip()))
                    buffer = []
                heading = match.group(2).strip() if match.lastindex == 2 else match.group(1).strip()
            else:
                buffer.append(line)

        if buffer:
            sections.append((heading, "\n".join(buffer).strip()))
        if not sections and text.strip():
            sections.append((heading, text.strip()))
        if sections and not sections[0][0]:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines and len(lines[0]) <= 80 and len(lines) > 1:
                inferred = lines[0].strip("# ")
                body = sections[0][1]
                if body.startswith(lines[0]):
                    body = body[len(lines[0]) :].strip()
                sections[0] = (inferred, body)
        return sections

    @staticmethod
    def _looks_like_heading(line: str, buffer: List[str]) -> bool:
        value = line.strip()
        if not value or len(value) > 80 or len(buffer) > 2:
            return False
        if re.match(r"^(第[一二三四五六七八九十百0-9]+[章节部分]|[0-9]+(\.[0-9]+)*[、. ])", value):
            return True
        return value.endswith(("章", "节", "篇", "部分")) and len(value) < 30

    def _split_text(self, text: str, size: int = None) -> List[str]:
        size = size or self.chunk_size
        paragraphs = self._protected_blocks(text)
        pieces = []
        current = ""

        for paragraph in paragraphs:
            if len(paragraph) > size:
                if current:
                    pieces.append(current)
                    current = ""
                pieces.extend(self._split_long_paragraph(paragraph, size))
                continue

            candidate = paragraph if not current else current + "\n\n" + paragraph
            if len(candidate) <= size:
                current = candidate
            else:
                pieces.append(current)
                prefix = current[-min(self.overlap, size // 4) :] if self.overlap else ""
                current = (prefix + "\n\n" + paragraph).strip()

        if current:
            pieces.append(current)
        return pieces

    @staticmethod
    def _protected_blocks(text: str) -> List[str]:
        """Keep Markdown tables and fenced code blocks together as indivisible blocks."""
        lines = text.splitlines()
        blocks, current, in_fence = [], [], False

        def flush():
            if current:
                value = "\n".join(current).strip()
                if value:
                    blocks.append(value)
                current[:] = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                current.append(line)
                in_fence = not in_fence
                if not in_fence:
                    flush()
                continue
            is_table = stripped.startswith("|")
            if in_fence or is_table:
                current.append(line)
                continue
            if current:
                flush()
            if stripped:
                blocks.append(stripped)
        flush()
        return blocks

    def _split_long_paragraph(self, paragraph: str, target_size: int) -> List[str]:
        pieces = []
        start = 0
        while start < len(paragraph):
            end = min(start + target_size, len(paragraph))
            if end < len(paragraph):
                boundary = max(
                    paragraph.rfind("。", start, end),
                    paragraph.rfind(". ", start, end),
                    paragraph.rfind("；", start, end),
                    paragraph.rfind("; ", start, end),
                )
                if boundary > start + target_size // 2:
                    end = boundary + 1
            pieces.append(paragraph[start:end].strip())
            if end >= len(paragraph):
                break
            start = max(end - self.overlap, start + 1)
        return pieces
