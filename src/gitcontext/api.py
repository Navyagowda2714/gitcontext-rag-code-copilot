"""FastAPI service: ``POST /ask``, ``GET /search``, ``POST /index``, ``GET /health``."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .config import RetrievalMode, Settings, get_settings
from .engine import GitContext
from .models import Citation

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    mode: RetrievalMode | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    refused: bool
    refusal_reason: str | None
    evidence_score: float
    model: str
    latency_ms: float


class SearchHit(BaseModel):
    chunk_id: str
    source_type: str
    title: str
    location: str
    url: str | None
    score: float


class IndexRequest(BaseModel):
    path: str
    fixtures_dir: str | None = None


def create_app(engine: GitContext | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="GitContext",
        version="0.1.0",
        description="Your codebase, answered: grounded Q&A over code, history, PRs, postmortems.",
    )
    state: dict[str, GitContext | None] = {"engine": engine}

    def get_engine() -> GitContext:
        if state["engine"] is None:
            try:
                state["engine"] = GitContext.load(settings=settings)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        eng = state["engine"]
        assert eng is not None
        return eng

    @app.get("/health")
    def health() -> dict[str, object]:
        eng = state["engine"]
        return {
            "status": "ok",
            "indexed_chunks": len(eng.chunks) if eng else None,
            "model": eng.llm.model if eng else None,
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        answer = get_engine().ask(req.question, mode=req.mode)
        logger.info(
            "ask refused=%s citations=%d latency_ms=%.0f",
            answer.refused,
            len(answer.citations),
            answer.latency_ms,
        )
        return AskResponse(
            **answer.model_dump(exclude={"retrieved", "input_tokens", "output_tokens"})
        )

    @app.get("/search", response_model=list[SearchHit])
    def search(
        q: str = Query(..., min_length=2),
        k: int = Query(8, ge=1, le=50),
        mode: RetrievalMode = "hybrid_rerank",
    ) -> list[SearchHit]:
        return [
            SearchHit(
                chunk_id=h.chunk.chunk_id,
                source_type=h.chunk.source_type.value,
                title=h.chunk.title,
                location=h.chunk.location,
                url=h.chunk.url,
                score=h.score,
            )
            for h in get_engine().search(q, k=k, mode=mode)
        ]

    @app.post("/index")
    def index(req: IndexRequest) -> dict[str, object]:
        path = Path(req.path)
        if not path.is_dir():
            raise HTTPException(status_code=400, detail=f"not a directory: {req.path}")
        eng = GitContext.build(
            path,
            fixtures_dir=Path(req.fixtures_dir) if req.fixtures_dir else None,
            settings=settings,
            llm=state["engine"].llm if state["engine"] else None,
        )
        state["engine"] = eng
        return {"indexed_chunks": len(eng.chunks)}

    return app
