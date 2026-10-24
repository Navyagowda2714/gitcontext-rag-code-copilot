"""Grounded answer generation with inline numbered citations and refusal."""

from __future__ import annotations

import re
import time

from ..llm import LLMClient, Message
from ..llm.offline import REFUSAL_TOKEN
from ..models import Answer, Citation, ScoredChunk
from .prompts import SYSTEM_PROMPT, build_user_prompt

_CITE = re.compile(r"\[(\d+)\]")
REFUSAL_MESSAGE = (
    "I couldn't find enough evidence in the indexed code, history, PRs, docs or postmortems "
    "to answer that confidently."
)


def _snippet(text: str, limit: int = 220) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def renumber_citations(text: str, n_sources: int) -> tuple[str, list[int]]:
    """Drop out-of-range citations and renumber the rest 1..m in order of first appearance.

    Returns the rewritten text and the original (1-based) source indices, in new order.
    """
    order: list[int] = []
    for m in _CITE.finditer(text):
        idx = int(m.group(1))
        if 1 <= idx <= n_sources and idx not in order:
            order.append(idx)
    mapping = {old: new for new, old in enumerate(order, start=1)}

    def sub(m: re.Match[str]) -> str:
        idx = int(m.group(1))
        return f"[{mapping[idx]}]" if idx in mapping else ""

    return _CITE.sub(sub, text).strip(), order


class AnswerGenerator:
    def __init__(self, llm: LLMClient, min_evidence: float = 0.35) -> None:
        self.llm = llm
        self.min_evidence = min_evidence

    def refuse(
        self, question: str, reason: str, evidence: float, hits: list[ScoredChunk]
    ) -> Answer:
        return Answer(
            question=question,
            answer=REFUSAL_MESSAGE,
            refused=True,
            refusal_reason=reason,
            evidence_score=evidence,
            model=self.llm.model,
            retrieved=hits,
        )

    def generate(self, question: str, hits: list[ScoredChunk], evidence: float) -> Answer:
        if not hits:
            return self.refuse(question, "no sources retrieved", evidence, hits)
        if evidence < self.min_evidence:
            return self.refuse(
                question,
                f"evidence score {evidence:.2f} below threshold {self.min_evidence:.2f}",
                evidence,
                hits,
            )
        start = time.perf_counter()
        resp = self.llm.complete(
            [Message(role="user", content=build_user_prompt(question, hits))],
            system=SYSTEM_PROMPT,
            temperature=0.0,
        )
        latency = (time.perf_counter() - start) * 1000
        if REFUSAL_TOKEN in resp.text:
            ans = self.refuse(question, "model judged the sources insufficient", evidence, hits)
            ans.input_tokens, ans.output_tokens = resp.input_tokens, resp.output_tokens
            return ans
        text, order = renumber_citations(resp.text, len(hits))
        if not order:
            ans = self.refuse(question, "answer contained no valid citations", evidence, hits)
            ans.input_tokens, ans.output_tokens = resp.input_tokens, resp.output_tokens
            return ans
        citations = []
        for new_idx, old_idx in enumerate(order, start=1):
            c = hits[old_idx - 1].chunk
            citations.append(
                Citation(
                    index=new_idx,
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    source_type=c.source_type,
                    title=c.title,
                    location=c.location,
                    url=c.url,
                    snippet=_snippet(c.text),
                )
            )
        return Answer(
            question=question,
            answer=text,
            citations=citations,
            evidence_score=evidence,
            model=resp.model,
            latency_ms=latency,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            retrieved=hits,
        )
