from __future__ import annotations

import numpy as np
import pytest

from gitcontext.engine import GitContext
from gitcontext.models import Chunk, ScoredChunk, SourceType
from gitcontext.retrieval import BM25, HashingEmbedder, diversify, reciprocal_rank_fusion, tokenize


def test_tokenizer_splits_identifiers_and_drops_stopwords() -> None:
    toks = tokenize("How does refreshAccessToken call refresh_if_needed?")
    assert {"refresh", "access", "token", "refreshaccesstoken", "need"} <= set(toks)
    assert "how" not in toks and "does" not in toks


def test_bm25_prefers_matching_documents_and_maxes_unseen_idf() -> None:
    corpus = [tokenize(t) for t in ["token refresh lock", "invoice proration", "cache stampede"]]
    bm25 = BM25(corpus)
    hits = bm25.search(tokenize("refresh token"), k=3)
    assert hits[0][0] == 0 and len(hits) == 1
    assert bm25.idf_of("kubernetes") == bm25.max_idf > bm25.idf_of("token")


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60)
    assert max(fused, key=fused.get) == "b"
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    emb = HashingEmbedder(dim=256)
    a = emb.embed(["token refresh lock", "token refresh lock", "invoice"])
    assert np.allclose(a[0], a[1])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert float(a[0] @ a[2]) < float(a[0] @ a[1])


def _sc(cid: str, doc: str, stype: SourceType, score: float) -> ScoredChunk:
    chunk = Chunk(chunk_id=cid, doc_id=doc, source_type=stype, title=cid, text="x")
    return ScoredChunk(chunk=chunk, score=score)


def test_diversify_caps_per_document_and_prefers_new_source_types() -> None:
    hits = [
        _sc("c1", "code:a", SourceType.CODE, 1.00),
        _sc("c2", "code:a", SourceType.CODE, 0.99),
        _sc("c3", "code:a", SourceType.CODE, 0.98),
        _sc("c4", "code:b", SourceType.CODE, 0.97),
        _sc("p1", "pr:1", SourceType.PR, 0.95),
    ]
    picked = [h.chunk.chunk_id for h in diversify(hits, k=3, max_per_doc=2, type_penalty=0.03)]
    assert picked == ["c1", "c2", "p1"]


def test_hybrid_rerank_surfaces_session_code_for_refresh_question(engine: GitContext) -> None:
    hits = engine.search("How does token refresh work?", k=5)
    paths = [h.chunk.path for h in hits]
    assert "shopfront/auth/session.py" in paths[:3]
    assert all("rr_coverage" in h.scores for h in hits)


def test_explicit_pr_reference_is_honoured(engine: GitContext) -> None:
    hits = engine.search("What did #152 change?", k=3)
    assert hits[0].chunk.doc_id == "pr:152"


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid", "hybrid_rerank"])
def test_every_mode_returns_ranked_results(engine: GitContext, mode: str) -> None:
    hits = engine.search("webhook signature verification", k=5, mode=mode)  # type: ignore[arg-type]
    assert hits and hits[0].score >= hits[-1].score
    assert any(h.chunk.path == "shopfront/billing/webhooks.py" for h in hits)
