"""
Every tunable setting in one place, read from environment variables.

`.env` in the project folder is loaded first; a variable already set in the real
environment wins over `.env` (handy for one-off overrides, e.g. in Docker). Every
setting has a default equal to the value that used to be hardcoded, so a `.env`
holding only the API keys behaves exactly as before. See `.env.example` for the full
list with explanations.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).parent

load_dotenv(PROJECT_DIR / ".env")


def _raw(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None  # an empty `KEY=` line means "use the default"


def _str(name: str, default: str) -> str:
    return _raw(name) or default


def _int(name: str, default: int) -> int:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _float(name: str, default: float) -> float:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


def _path(name: str, default: str) -> Path:
    path = Path(_str(name, default))
    return path if path.is_absolute() else PROJECT_DIR / path


@dataclass(frozen=True)
class Settings:
    # --- Elasticsearch ---
    es_host: str
    es_index: str  # the alias every query goes through (see es_index.py / index_admin.py)

    # --- Models ---
    embed_model: str
    embed_dims: int  # must match embed_model's dense output; baked into the index mapping
    reranker_model: str
    llm_model: str
    llm_temperature: float
    model_device: str  # auto | cuda | cpu

    # --- Retrieval (per search_documents call) ---
    dense_k: int
    sparse_k: int
    fused_top_n: int
    final_top_n: int
    sparse_top_tokens: int
    rrf_k: int
    rerank_max_length: int

    # --- Ingestion: chunking + embedding (affects newly uploaded documents only) ---
    chunk_target_tokens: int
    chunk_max_tokens: int
    chunk_overlap_ratio: float
    embed_batch_size: int
    embed_max_length: int

    # --- App ---
    max_history_messages: int
    chat_db_path: Path
    trace_name: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            es_host=_str("ES_HOST", "http://localhost:9200"),
            es_index=_str("ES_INDEX", "document-chunks"),
            embed_model=_str("EMBED_MODEL", "BAAI/bge-m3"),
            embed_dims=_int("EMBED_DIMS", 1024),
            reranker_model=_str("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            llm_model=_str("LLM_MODEL", "gemini-2.5-flash"),
            llm_temperature=_float("LLM_TEMPERATURE", 0.2),
            model_device=_str("MODEL_DEVICE", "auto").lower(),
            dense_k=_int("RETRIEVAL_DENSE_K", 20),
            sparse_k=_int("RETRIEVAL_SPARSE_K", 20),
            fused_top_n=_int("RETRIEVAL_FUSED_TOP_N", 20),
            final_top_n=_int("RETRIEVAL_FINAL_TOP_N", 5),
            sparse_top_tokens=_int("RETRIEVAL_SPARSE_TOP_TOKENS", 32),
            rrf_k=_int("RETRIEVAL_RRF_K", 60),
            rerank_max_length=_int("RERANK_MAX_LENGTH", 1024),
            chunk_target_tokens=_int("CHUNK_TARGET_TOKENS", 450),
            chunk_max_tokens=_int("CHUNK_MAX_TOKENS", 600),
            chunk_overlap_ratio=_float("CHUNK_OVERLAP_RATIO", 0.15),
            embed_batch_size=_int("EMBED_BATCH_SIZE", 8),
            embed_max_length=_int("EMBED_MAX_LENGTH", 8192),
            max_history_messages=_int("MAX_HISTORY_MESSAGES", 6),
            chat_db_path=_path("CHAT_DB_PATH", "chat_history.db"),
            trace_name=_str("TRACE_NAME", "document-qa"),
        )

    def __post_init__(self):
        """Fail at startup with a clear message instead of deep inside a search."""
        errors = []
        positive = [
            "embed_dims", "dense_k", "sparse_k", "fused_top_n", "final_top_n", "sparse_top_tokens",
            "rrf_k", "rerank_max_length", "chunk_target_tokens", "chunk_max_tokens",
            "embed_batch_size", "embed_max_length",
        ]
        for name in positive:
            if getattr(self, name) <= 0:
                errors.append(f"{name.upper()} must be > 0 (got {getattr(self, name)})")
        if self.max_history_messages < 0:
            errors.append(f"MAX_HISTORY_MESSAGES must be >= 0 (got {self.max_history_messages})")
        if self.model_device not in {"auto", "cuda", "cpu"}:
            errors.append(f"MODEL_DEVICE must be auto, cuda or cpu (got {self.model_device!r})")
        if not 0 <= self.llm_temperature <= 2:
            errors.append(f"LLM_TEMPERATURE must be between 0 and 2 (got {self.llm_temperature})")
        if not 0 <= self.chunk_overlap_ratio < 1:
            errors.append(f"CHUNK_OVERLAP_RATIO must be in [0, 1) (got {self.chunk_overlap_ratio})")
        if self.chunk_target_tokens > self.chunk_max_tokens:
            errors.append("CHUNK_TARGET_TOKENS must be <= CHUNK_MAX_TOKENS")
        if self.final_top_n > self.fused_top_n:
            errors.append("RETRIEVAL_FINAL_TOP_N must be <= RETRIEVAL_FUSED_TOP_N (rerank picks from the fused list)")
        if errors:
            raise ValueError("Invalid settings in .env / environment:\n  - " + "\n  - ".join(errors))


settings = Settings.from_env()
