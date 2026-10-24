"""High-level facade: index a repository, persist it, search it and answer questions."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .answer import AnswerGenerator
from .chunking import chunk_documents
from .config import RetrievalMode, Settings, get_settings
from .ingest import load_repository
from .llm import LLMClient, get_llm
from .models import Answer, Chunk, ScoredChunk
from .retrieval import HybridRetriever, diversify, make_embedder
from .retrieval.rerank import Reranker

logger = logging.getLogger(__name__)

CHUNKS_FILE = "chunks.jsonl"
META_FILE = "meta.json"


def save_index(chunks: list[Chunk], index_dir: Path, meta: dict[str, object]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(c.model_dump_json() + "\n")
    (index_dir / META_FILE).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def load_chunks(index_dir: Path) -> list[Chunk]:
    path = index_dir / CHUNKS_FILE
    if not path.exists():
        raise FileNotFoundError(f"no index at {index_dir}; run `ask-repo index <path>` first")
    with path.open(encoding="utf-8") as fh:
        return [Chunk.model_validate_json(line) for line in fh if line.strip()]


class GitContext:
    """Owns the retriever + generator. Construct with :meth:`build` or :meth:`load`."""

    def __init__(
        self,
        chunks: list[Chunk],
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        embedder: str = "tfidf",
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.chunks = chunks
        self.retriever = HybridRetriever(
            chunks,
            embedder=make_embedder(embedder),
            reranker=reranker,
            rrf_k=self.settings.rrf_k,
            candidates=self.settings.candidates,
        )
        self.llm = llm or get_llm()
        self.generator = AnswerGenerator(self.llm, min_evidence=self.settings.min_evidence)

    @classmethod
    def build(
        cls,
        repo_path: Path,
        fixtures_dir: Path | None = None,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        persist: bool = True,
    ) -> GitContext:
        settings = settings or get_settings()
        t0 = time.perf_counter()
        docs = load_repository(
            repo_path,
            fixtures_dir=fixtures_dir,
            repo_url=settings.repo_url,
            max_commits=settings.max_commits,
        )
        chunks = chunk_documents(docs)
        logger.info("indexed %d documents into %d chunks", len(docs), len(chunks))
        if persist:
            save_index(
                chunks,
                settings.index_dir,
                {
                    "repo_path": str(Path(repo_path).resolve()),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "documents": dict(Counter(d.source_type.value for d in docs)),
                    "chunks": dict(Counter(c.source_type.value for c in chunks)),
                    "build_seconds": round(time.perf_counter() - t0, 3),
                },
            )
        return cls(chunks, settings=settings, llm=llm)

    @classmethod
    def load(cls, settings: Settings | None = None, llm: LLMClient | None = None) -> GitContext:
        settings = settings or get_settings()
        return cls(load_chunks(settings.index_dir), settings=settings, llm=llm)

    def search(
        self, question: str, k: int = 10, mode: RetrievalMode | None = None
    ) -> list[ScoredChunk]:
        return self.retriever.search(question, k=k, mode=mode or self.settings.mode)

    def ask(self, question: str, mode: RetrievalMode | None = None) -> Answer:
        t0 = time.perf_counter()
        pool = self.search(question, k=self.settings.candidates, mode=mode)
        hits = diversify(
            pool,
            self.settings.top_k,
            self.settings.max_chunks_per_doc,
            self.settings.type_diversity_penalty,
        )
        evidence = self.retriever.evidence_score(question, hits)
        answer = self.generator.generate(question, hits, evidence)
        answer.latency_ms = (time.perf_counter() - t0) * 1000
        return answer
