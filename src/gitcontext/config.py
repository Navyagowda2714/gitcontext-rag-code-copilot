"""Runtime configuration (env vars prefixed ``GITCONTEXT_``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["bm25", "vector", "hybrid", "hybrid_rerank"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GITCONTEXT_", env_file=".env", extra="ignore")

    index_dir: Path = Path(".gitcontext-index")
    repo_url: str | None = None  # e.g. https://github.com/acme/shopfront, used to build links
    mode: RetrievalMode = "hybrid_rerank"
    candidates: int = 30  # per-retriever candidate pool before fusion
    rrf_k: int = 60
    top_k: int = 6  # chunks handed to the generator
    max_chunks_per_doc: int = 2
    type_diversity_penalty: float = 0.03  # MMR-style penalty per same-type chunk in context
    min_evidence: float = 0.35  # IDF-weighted query coverage needed to attempt an answer
    max_commits: int = 500
    log_level: str = "INFO"


def get_settings() -> Settings:
    return Settings()
