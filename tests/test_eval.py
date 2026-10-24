from __future__ import annotations

import pytest

from gitcontext.eval import load_questions, mrr, ndcg_at_k, recall_at_k
from gitcontext.eval.metrics import matches_gold
from gitcontext.models import Chunk, SourceType


def test_ranking_metrics_on_handcrafted_ranking() -> None:
    gold = ["a", "b"]
    labels = [None, "a", None, "b", None]
    assert recall_at_k(labels, gold, 2) == 0.5
    assert recall_at_k(labels, gold, 5) == 1.0
    assert mrr(labels) == 0.5
    ideal = 1 + 1 / 1.584962500721156
    assert ndcg_at_k(labels, gold, 5) == pytest.approx(
        (1 / 1.584962500721156 + 1 / 2.321928094887362) / ideal
    )


def test_commit_gold_matches_by_subject() -> None:
    c = Chunk(
        chunk_id="x",
        doc_id="commit:abc",
        source_type=SourceType.COMMIT,
        title="fix: Banker's rounding",
        text="",
    )
    assert matches_gold(c, "commit~banker's rounding")
    assert not matches_gold(c, "pr:1")


def test_question_set_is_well_formed() -> None:
    qs = load_questions()
    assert len([q for q in qs if q.gold]) >= 25
    assert len({q.id for q in qs}) == len(qs)
    assert any(not q.gold for q in qs)
