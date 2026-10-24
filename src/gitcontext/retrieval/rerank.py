"""Second-stage rerankers.

``HeuristicReranker`` (default) is a transparent cross-feature scorer that looks at the query and
the candidate *together* (unlike first-stage retrievers that score them independently):

* IDF-weighted query-term coverage of the chunk
* matches in the title / path / symbol name
* query bigrams that appear verbatim in the chunk
* the fused first-stage score
* a source-type prior inferred from the question's intent ("why did ... break" -> incidents/PRs)
* explicit references (``#134``, a file path, a symbol) in the question

``LLMReranker`` and ``CrossEncoderReranker`` implement the same protocol for higher quality when a
model is available.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from itertools import pairwise
from typing import Protocol

from ..llm import LLMClient, Message, parse_json_loose
from ..models import ScoredChunk, SourceType
from .text import query_terms, tokenize

IdfFn = Callable[[str], float]


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]: ...


INTENT_KEYWORDS: dict[SourceType, tuple[str, ...]] = {
    SourceType.INCIDENT: (
        "incident",
        "outage",
        "postmortem",
        "post-mortem",
        "root cause",
        "broke",
        "broken",
        "went wrong",
        "sev",
        "impact",
        "logged out",
        "page",
        "paged",
        "alert",
        "duplicate",
    ),
    SourceType.PR: (
        "pr ",
        "pull request",
        "review",
        "reviewer",
        "decided",
        "decision",
        "why did we",
        "why do we",
        "rationale",
        "discussed",
        "approved",
        "trade-off",
        "tradeoff",
    ),
    SourceType.COMMIT: (
        "commit",
        "who changed",
        "who wrote",
        "when was",
        "when did",
        "introduced",
        "history",
        "last change",
        "changed",
        "author",
        "added",
    ),
    SourceType.DISCUSSION: ("discussion", "proposal", "rfc", "should we", "consider", "debate"),
    SourceType.DOC: (
        "docs",
        "documentation",
        "readme",
        "runbook",
        "how do i",
        "how to",
        "setup",
        "set up",
        "configure",
        "deploy",
        "on-call",
        "steps",
    ),
    SourceType.CODE: (
        "function",
        "class",
        "method",
        "implement",
        "implementation",
        "code",
        "where is",
        "how does",
        "constant",
        "returns",
        "raise",
        "parameter",
        "signature",
    ),
}

_PR_REF = re.compile(r"(?:#|pr\s*#?|pull request\s*#?)(\d+)", re.IGNORECASE)
_PATHISH = re.compile(r"[\w./-]+\.(?:py|ts|tsx|js|go|md|rs|java)\b")


def detect_intents(query: str) -> set[SourceType]:
    q = f" {query.lower()} "
    return {stype for stype, kws in INTENT_KEYWORDS.items() if any(k in q for k in kws)}


class HeuristicReranker:
    name = "heuristic"

    def __init__(
        self,
        idf: IdfFn,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.idf = idf
        self.w = {
            "coverage": 0.35,
            "title": 0.15,
            "bigram": 0.10,
            "fused": 0.25,
            "intent": 0.12,
            "explicit": 0.30,
        } | (weights or {})

    def features(self, query: str, cand: ScoredChunk, max_fused: float) -> dict[str, float]:
        terms = query_terms(query)
        c = cand.chunk
        body_toks = tokenize(c.search_text)
        body = set(body_toks)
        head = set(tokenize(" ".join(x for x in (c.title, c.path or "", c.symbol or "") if x)))
        total_idf = sum(self.idf(t) for t in terms) or 1.0
        coverage = sum(self.idf(t) for t in terms if t in body) / total_idf
        title = sum(self.idf(t) for t in terms if t in head) / total_idf
        q_bigrams = set(pairwise(terms))
        c_bigrams = set(pairwise(body_toks))
        bigram = len(q_bigrams & c_bigrams) / len(q_bigrams) if q_bigrams else 0.0
        intents = detect_intents(query)
        intent = 1.0 if c.source_type in intents else 0.0
        explicit = 0.0
        for m in _PR_REF.finditer(query):
            if c.pr_number is not None and int(m.group(1)) == c.pr_number:
                explicit = 1.0
        for p in _PATHISH.findall(query):
            if c.path and c.path.endswith(p):
                explicit = 1.0
        if c.symbol and c.symbol not in {"<module>"} and c.symbol.split(".")[-1] in query:
            explicit = max(explicit, 0.7)
        return {
            "coverage": coverage,
            "title": title,
            "bigram": bigram,
            "fused": cand.score / max_fused if max_fused else 0.0,
            "intent": intent,
            "explicit": explicit,
        }

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        max_fused = max(c.score for c in candidates)
        out: list[ScoredChunk] = []
        for cand in candidates:
            feats = self.features(query, cand, max_fused)
            score = sum(self.w[k] * v for k, v in feats.items())
            out.append(
                ScoredChunk(
                    chunk=cand.chunk,
                    score=score,
                    scores={**cand.scores, **{f"rr_{k}": round(v, 4) for k, v in feats.items()}},
                )
            )
        return sorted(out, key=lambda s: s.score, reverse=True)


class LLMReranker:
    """Pointwise relevance grading by an LLM (0-3), combined with the first-stage score."""

    name = "llm"

    def __init__(self, llm: LLMClient, max_candidates: int = 15) -> None:
        self.llm = llm
        self.max_candidates = max_candidates

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        head, tail = candidates[: self.max_candidates], candidates[self.max_candidates :]
        listing = "\n\n".join(
            f"[{i}] ({c.chunk.source_type.value}) {c.chunk.title}\n{c.chunk.text[:600]}"
            for i, c in enumerate(head)
        )
        schema = {
            "type": "object",
            "properties": {"scores": {"type": "array", "items": {"type": "integer"}}},
            "required": ["scores"],
        }
        resp = self.llm.complete(
            [
                Message(
                    role="user",
                    content=(
                        f"Question: {query}\n\nRate how useful each passage is for answering the "
                        f"question, 0 (irrelevant) to 3 (directly answers). Return one score per "
                        f"passage in order.\n\n{listing}"
                    ),
                )
            ],
            system="You are a precise relevance judge for a code-search engine.",
            json_schema=schema,
        )
        try:
            grades = [float(x) for x in parse_json_loose(resp.text)["scores"]]
        except (ValueError, KeyError, TypeError):
            return candidates
        max_fused = max((c.score for c in head), default=1.0) or 1.0
        rescored = [
            ScoredChunk(
                chunk=c.chunk,
                score=(grades[i] / 3 if i < len(grades) else 0) + 0.2 * c.score / max_fused,
                scores={**c.scores, "llm_grade": grades[i] if i < len(grades) else 0},
            )
            for i, c in enumerate(head)
        ]
        return sorted(rescored, key=lambda s: s.score, reverse=True) + tail


class CrossEncoderReranker:
    """sentence-transformers cross-encoder (``pip install gitcontext[st]``)."""

    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError("pip install gitcontext[st] for CrossEncoderReranker") from exc
        self._model = CrossEncoder(model_name)

    def rerank(  # pragma: no cover - needs model download
        self, query: str, candidates: list[ScoredChunk]
    ) -> list[ScoredChunk]:
        scores = self._model.predict([(query, c.chunk.search_text[:2000]) for c in candidates])
        out = [
            ScoredChunk(chunk=c.chunk, score=float(s), scores={**c.scores, "ce": float(s)})
            for c, s in zip(candidates, scores, strict=True)
        ]
        return sorted(out, key=lambda s: s.score, reverse=True)
