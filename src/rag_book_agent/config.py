import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Settings:
    data_dir: str = "data"
    database_name: str = "rag_book.db"
    chunk_size: int = 1800
    chunk_overlap: int = 220
    sparse_top_k: int = 30
    dense_top_k: int = 30
    rerank_top_k: int = 12
    context_top_k: int = 6
    rrf_k: int = 60
    chroma_dir: str = "data/chroma"
    chroma_collection: str = "rag_chunks"
    embedding_model: str = ""
    embedding_local_files_only: bool = False
    reranker_model: str = ""
    reranker_local_files_only: bool = False
    web_search_url: str = ""
    web_search_enabled: bool = False
    web_search_provider: str = "duckduckgo"
    deepseek_web_search: bool = True
    deepseek_web_max_tool_calls: int = 6
    deepseek_web_max_output_tokens: int = 1400
    web_search_candidate_multiplier: int = 3
    web_search_per_domain_limit: int = 2
    web_search_max_workers: int = 4
    web_search_discovery_timeout_seconds: int = 8
    web_search_page_timeout_seconds: int = 8
    web_search_max_page_chars: int = 8000
    web_search_user_agent: str = "Mozilla/5.0 (compatible; RagBookResearch/1.0)"
    api_base: str = ""
    api_model: str = ""
    api_key_env: str = "RAG_BOOK_API_KEY"
    request_timeout: int = 90
    pdf_backend: str = "pymupdf4llm"
    pdf_min_text_chars: int = 20
    pdf_ocr_enabled: bool = True
    pdf_ocr_engine: str = "rapidocr"
    pdf_render_dpi: int = 220
    pdf_keep_images: bool = False
    pdf_formula_enabled: bool = True
    pdf_table_enabled: bool = True
    parent_chunk_size: int = 3600
    child_chunk_size: int = 900
    agent_timeout_seconds: int = 45
    agent_max_react_steps: int = 3
    agent_local_confidence_threshold: float = 0.8
    agent_local_context_chars: int = 6000
    agent_web_context_chars: int = 4000
    agent_web_timeout_seconds: int = 30
    agent_planner_timeout_seconds: float = 1.5
    agent_fast_question_max_chars: int = 32
    agent_two_stage_full_sources: int = 3

    @property
    def database_path(self) -> Path:
        return Path(self.data_dir) / self.database_name

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_dir)

    def resolve_model_path(self, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            path = self.database_path.parent.parent / path
        return str(path)

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    @classmethod
    def load(cls, path: Path) -> "Settings":
        if not path.exists():
            settings = cls()
            settings.save(path)
            return settings

        with path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)

        allowed = set(cls.__dataclass_fields__.keys())
        clean_values = {key: value for key, value in values.items() if key in allowed}
        return cls(**clean_values)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2, ensure_ascii=False)
            handle.write("\n")


def load_settings(project_dir: Path) -> Settings:
    _load_env_file(project_dir / ".env")
    config_path = project_dir / "config.json"
    settings = Settings.load(config_path)
    data_dir = Path(settings.data_dir)
    if not data_dir.is_absolute():
        settings.data_dir = str(project_dir / data_dir)
    return settings


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines without replacing explicit environment values."""
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
