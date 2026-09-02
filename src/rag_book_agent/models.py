from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Page:
    number: int
    text: str
    parser: str = "native"
    ocr_used: bool = False
    ocr_confidence: float = 0.0
    image_count: int = 0
    table_count: int = 0
    ocr_error: str = ""


@dataclass
class ParsedDocument:
    path: str
    title: str
    file_type: str
    content_hash: str
    pages: List[Page]


@dataclass
class Chunk:
    id: Optional[int]
    document_id: int
    text: str
    heading: str
    page_start: int
    page_end: int
    position: int
    parent_id: Optional[int] = None
    chunk_type: str = "child"


@dataclass
class SearchResult:
    chunk: Chunk
    document_title: str
    document_path: str
    sparse_score: float = 0.0
    dense_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0

    @property
    def citation(self) -> str:
        page = str(self.chunk.page_start)
        if self.chunk.page_end != self.chunk.page_start:
            page += "-" + str(self.chunk.page_end)
        return "%s, p.%s" % (self.document_title, page)


@dataclass
class Answer:
    text: str
    sources: List[SearchResult] = field(default_factory=list)
    mode: str = "extractive"


@dataclass
class EvaluationItem:
    question: str
    expected_chunk_ids: List[int]
    reference_answer: str = ""
