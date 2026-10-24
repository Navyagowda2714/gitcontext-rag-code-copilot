"""Plain-text rendering of answers and search results (CLI, logs, examples)."""

from __future__ import annotations

from ..models import Answer, ScoredChunk


def render_answer(answer: Answer, show_links: bool = True) -> str:
    lines = [answer.answer, ""]
    if answer.refused:
        lines.append(f"(refused: {answer.refusal_reason}; evidence={answer.evidence_score:.2f})")
        return "\n".join(lines)
    lines.append("Sources:")
    for c in answer.citations:
        lines.append(f"  [{c.index}] {c.source_type.value:<10} {c.location:<40} {c.title}")
        if show_links and c.url:
            lines.append(f"      {c.url}")
    lines.append("")
    lines.append(
        f"model={answer.model} evidence={answer.evidence_score:.2f} "
        f"latency={answer.latency_ms:.0f}ms tokens_in={answer.input_tokens} "
        f"tokens_out={answer.output_tokens}"
    )
    return "\n".join(lines)


def render_hits(hits: list[ScoredChunk]) -> str:
    rows = []
    for rank, h in enumerate(hits, start=1):
        c = h.chunk
        rows.append(
            f"{rank:>2}. {h.score:7.4f}  {c.source_type.value:<10} {c.location:<40} {c.title[:60]}"
        )
    return "\n".join(rows)
