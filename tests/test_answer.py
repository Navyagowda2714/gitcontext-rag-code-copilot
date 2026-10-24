from __future__ import annotations

from gitcontext.answer import AnswerGenerator, renumber_citations
from gitcontext.engine import GitContext
from gitcontext.llm import ScriptedLLM
from gitcontext.models import Chunk, ScoredChunk, SourceType


def _hits() -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(
                chunk_id=f"c{i}",
                doc_id=f"doc{i}",
                source_type=SourceType.CODE if i == 1 else SourceType.PR,
                title=f"t{i}",
                text=f"text {i}",
                pr_number=None if i == 1 else 100 + i,
            ),
            score=1.0,
        )
        for i in (1, 2, 3)
    ]


def test_renumber_drops_invalid_and_orders_by_first_use() -> None:
    text, order = renumber_citations("B is true [3]. A [1][3]. Bogus [9].", n_sources=3)
    assert order == [3, 1]
    assert text == "B is true [1]. A [2][1]. Bogus ."


def test_refuses_below_evidence_threshold_without_calling_llm() -> None:
    llm = ScriptedLLM([])  # would raise if called
    ans = AnswerGenerator(llm, min_evidence=0.5).generate("q?", _hits(), evidence=0.2)
    assert ans.refused and "below threshold" in (ans.refusal_reason or "")
    assert llm.calls == []


def test_citations_map_back_to_the_cited_chunks() -> None:
    llm = ScriptedLLM(["The PR explains it [3]. The code does it [1]."])
    ans = AnswerGenerator(llm).generate("why?", _hits(), evidence=0.9)
    assert not ans.refused
    assert ans.answer == "The PR explains it [1]. The code does it [2]."
    assert [c.chunk_id for c in ans.citations] == ["c3", "c1"]
    assert ans.citations[0].location == "PR #103"
    prompt = llm.calls[0]["messages"][0].content
    assert '<source id="1" type="code"' in prompt and "<question>" in prompt


def test_model_refusal_and_uncited_answers_are_refused() -> None:
    gen = AnswerGenerator(ScriptedLLM(["INSUFFICIENT_EVIDENCE", "An answer with no citations."]))
    first = gen.generate("q?", _hits(), evidence=0.9)
    second = gen.generate("q?", _hits(), evidence=0.9)
    assert first.refused and "insufficient" in (first.refusal_reason or "")
    assert second.refused and "no valid citations" in (second.refusal_reason or "")


def test_end_to_end_answer_cites_code_pr_and_postmortem(engine: GitContext) -> None:
    ans = engine.ask("How does token refresh work?")
    assert not ans.refused
    docs = {c.doc_id for c in ans.citations}
    assert "code:shopfront/auth/session.py" in docs
    assert "pr:134" in docs
    assert "incident:INC-2026-014" in docs
    assert all(c.url for c in ans.citations)


def test_out_of_scope_question_is_refused(engine: GitContext) -> None:
    ans = engine.ask("How do we configure Kubernetes horizontal pod autoscaling?")
    assert ans.refused and ans.citations == []
