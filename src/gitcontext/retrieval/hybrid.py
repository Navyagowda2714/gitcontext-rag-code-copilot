"""Hybrid retrieval: BM25 + dense vectors fused with Reciprocal Rank Fusion, then reranked."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..config import RetrievalMode
from ..models import Chunk, ScoredChunk
from .bm25 import BM25
from .embeddings import Embedder, TfidfEmbedder
from .rerank import HeuristicReranker, Reranker
from .text import query_terms, tokenize


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """RRF (Cormack et al., 2009): score(d) = sum over lists of 1 / (k + rank(d))."""
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] += 1.0 / (k + rank)
    return dict(fused)


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
        candidates: int = 30,
    ) -> None:
        if not chunks:
            raise ValueError("cannot build a retriever over zero chunks")
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        texts = [c.search_text for c in chunks]
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.embedder = embedder or TfidfEmbedder()
        self.embedder.fit(texts)
        self.matrix = self.embedder.embed(texts)
        self.reranker = reranker or HeuristicReranker(idf=self.bm25.idf_of)
        self.rrf_k = rrf_k
        self.candidates = candidates

    # -- first stage -------------------------------------------------------------------------
    def bm25_search(self, query: str, k: int) -> list[ScoredChunk]:
        hits = self.bm25.search(tokenize(query), k)
        return [ScoredChunk(chunk=self.chunks[i], score=s, scores={"bm25": s}) for i, s in hits]

    def vector_search(self, query: str, k: int) -> list[ScoredChunk]:
        q = self.embedder.embed([query])[0]
        sims = self.matrix @ q
        order = np.argsort(-sims, kind="stable")[:k]
        return [
            ScoredChunk(
                chunk=self.chunks[i], score=float(sims[i]), scores={"vector": float(sims[i])}
            )
            for i in order
            if sims[i] > 0
        ]

    def hybrid_search(self, query: str, k: int) -> list[ScoredChunk]:
        n = max(k, self.candidates)
        lexical = self.bm25_search(query, n)
        dense = self.vector_search(query, n)
        fused = reciprocal_rank_fusion(
            [[s.chunk.chunk_id for s in lexical], [s.chunk.chunk_id for s in dense]], k=self.rrf_k
        )
        detail: dict[str, dict[str, float]] = defaultdict(dict)
        for rank, s in enumerate(lexical, start=1):
            detail[s.chunk.chunk_id] |= {"bm25": round(s.score, 4), "bm25_rank": rank}
        for rank, s in enumerate(dense, start=1):
            detail[s.chunk.chunk_id] |= {"vector": round(s.score, 4), "vector_rank": rank}
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            ScoredChunk(chunk=self.by_id[cid], score=score, scores={**detail[cid], "rrf": score})
            for cid, score in ranked
        ]

    # -- full pipeline -----------------------------------------------------------------------
    def search(
        self, query: str, k: int = 10, mode: RetrievalMode = "hybrid_rerank"
    ) -> list[ScoredChunk]:
        if mode == "bm25":
            return self.bm25_search(query, k)
        if mode == "vector":
            return self.vector_search(query, k)
        if mode == "hybrid":
            return self.hybrid_search(query, k)
        pool = self.hybrid_search(query, self.candidates)
        return self.reranker.rerank(query, pool)[:k]

    def evidence_score(self, query: str, hits: list[ScoredChunk], top_n: int = 3) -> float:
        """IDF-weighted share of query terms covered by the top hits (0..1).

        Unseen terms get maximal IDF, so questions about things the index has never heard of
        ("kubernetes autoscaling") score low and trigger a refusal instead of a hallucination.
        """
        terms = query_terms(query)
        if not terms or not hits:
            return 0.0
        total = sum(self.bm25.idf_of(t) for t in terms)
        union: set[str] = set()
        best_single = 0.0
        for h in hits[:top_n]:
            toks = set(tokenize(h.chunk.search_text))
            union |= toks
            best_single = max(best_single, sum(self.bm25.idf_of(t) for t in terms if t in toks))
        union_cov = sum(self.bm25.idf_of(t) for t in terms if t in union) / total
        # Blend: one chunk should carry most of the question; the union may fill in the rest.
        return 0.5 * best_single / total + 0.5 * union_cov


def diversify(
    hits: list[ScoredChunk], k: int, max_per_doc: int = 2, type_penalty: float = 0.03
) -> list[ScoredChunk]:
    """Greedy, MMR-style selection of the generator's context window.

    Caps chunks per document and subtracts ``type_penalty`` for every already-selected chunk of
    the same source type, so near-tied candidates from *other* artefacts (the PR or postmortem
    behind a piece of code) win the last slots instead of a fifth slice of the same file.
    """
    per_doc: dict[str, int] = defaultdict(int)
    per_type: dict[str, int] = defaultdict(int)
    remaining = list(hits)
    out: list[ScoredChunk] = []
    while remaining and len(out) < k:
        eligible = [h for h in remaining if per_doc[h.chunk.doc_id] < max_per_doc]
        if not eligible:
            break
        best = max(eligible, key=lambda h: h.score - type_penalty * per_type[h.chunk.source_type])
        remaining.remove(best)
        per_doc[best.chunk.doc_id] += 1
        per_type[best.chunk.source_type] += 1
        out.append(best)
    return out
