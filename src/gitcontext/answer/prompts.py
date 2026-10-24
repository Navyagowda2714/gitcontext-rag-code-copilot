"""Prompt templates for grounded answering. The source block format is also parsed by the
offline provider, so keep the two in sync."""

from __future__ import annotations

from ..models import ScoredChunk

SYSTEM_PROMPT = """You are GitContext, a senior engineer answering questions about a codebase.
Answer ONLY from the numbered sources provided (code, commits, pull requests, discussions,
docs, incident postmortems). Rules:
- Cite every factual sentence with the source number(s) in square brackets, e.g. [1] or [2][3].
- Prefer connecting evidence across source types: what the code does, why (PR/discussion), and
  what went wrong before (postmortem).
- Mention file paths and symbols in backticks.
- Never use outside knowledge about this codebase. If the sources do not contain enough evidence
  to answer, reply with exactly: INSUFFICIENT_EVIDENCE
- Be concise: at most 6 sentences."""

MAX_SOURCE_CHARS = 1800


def _attr(value: str) -> str:
    return value.replace('"', "'").replace("\n", " ")


def format_sources(hits: list[ScoredChunk]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        code_symbol = c.source_type.value == "code" and c.symbol and c.symbol != "<module>"
        title = c.symbol if code_symbol else c.title
        body = c.text if len(c.text) <= MAX_SOURCE_CHARS else c.text[:MAX_SOURCE_CHARS] + "\n..."
        meta = []
        if c.author:
            meta.append(f"author: {c.author}")
        if c.updated_at:
            meta.append(f"date: {c.updated_at.date().isoformat()}")
        if meta and c.source_type.value in {"commit", "pr", "incident", "discussion"}:
            body = f"({', '.join(meta)})\n{body}"
        blocks.append(
            f'<source id="{i}" type="{c.source_type.value}" title="{_attr(title)}"'
            f' location="{_attr(c.location)}">\n{body}\n</source>'
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, hits: list[ScoredChunk]) -> str:
    return f"<sources>\n{format_sources(hits)}\n</sources>\n\n<question>\n{question}\n</question>"
