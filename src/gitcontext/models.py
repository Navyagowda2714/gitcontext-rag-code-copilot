"""Core data model: documents from connectors, chunks for retrieval, answers with citations."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    CODE = "code"
    DOC = "doc"
    COMMIT = "commit"
    PR = "pr"
    DISCUSSION = "discussion"
    INCIDENT = "incident"


class Document(BaseModel):
    """A unit produced by an ingestion connector (a file, a commit, a PR, a postmortem...)."""

    doc_id: str
    source_type: SourceType
    title: str
    text: str
    path: str | None = None
    url: str | None = None
    author: str | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable span of a document plus the metadata needed to cite it."""

    chunk_id: str
    doc_id: str
    source_type: SourceType
    title: str
    text: str
    path: str | None = None
    symbol: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    author: str | None = None
    pr_number: int | None = None
    commit_sha: str | None = None
    updated_at: datetime | None = None
    url: str | None = None
    language: str | None = None

    @property
    def location(self) -> str:
        """Human-readable pointer: ``path:L10-L42``, ``PR #134``, ``commit abc1234``..."""
        if self.source_type == SourceType.COMMIT and self.commit_sha:
            return self.commit_sha[:7]
        if self.source_type == SourceType.PR and self.pr_number is not None:
            return f"PR #{self.pr_number}"
        if self.source_type == SourceType.DISCUSSION and self.pr_number is not None:
            return f"Discussion #{self.pr_number}"
        if self.path and self.start_line:
            return f"{self.path}:L{self.start_line}-L{self.end_line}"
        return self.path or self.doc_id

    @property
    def search_text(self) -> str:
        """Text used for indexing: a contextual header (title/path/symbol) plus the body."""
        header = " ".join(x for x in (self.title, self.path, self.symbol) if x)
        return f"{header}\n{self.text}"


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    scores: dict[str, float] = Field(default_factory=dict)


class Citation(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    source_type: SourceType
    title: str
    location: str
    url: str | None = None
    snippet: str


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    evidence_score: float = 0.0
    model: str = "unknown"
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    retrieved: list[ScoredChunk] = Field(default_factory=list)
