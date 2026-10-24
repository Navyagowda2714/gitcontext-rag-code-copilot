"""Heading-aware markdown chunking with breadcrumb titles (``Auth > Token refresh``)."""

from __future__ import annotations

import re

from ..models import Chunk, Document

_HEADING = re.compile(r"^(#{1,4})\s+(.*?)\s*#*\s*$")
MIN_SECTION_CHARS = 160
MAX_SECTION_CHARS = 1800


def _sections(text: str) -> list[tuple[list[str], int, int, str]]:
    """Return (breadcrumb, start_line, end_line, body) per heading section, fence-aware."""
    lines = text.splitlines()
    stack: list[tuple[int, str]] = []
    sections: list[tuple[list[str], int, int, str]] = []
    cur_start, cur_lines = 1, []
    in_fence = False

    def flush(end_line: int) -> None:
        body = "\n".join(cur_lines).strip()
        if body:
            sections.append(([t for _, t in stack], cur_start, end_line, body))

    for i, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            flush(i - 1)
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
            cur_start, cur_lines = i, [line]
        else:
            cur_lines.append(line)
    flush(len(lines))
    return sections


def chunk_markdown(doc: Document, title_prefix: str | None = None) -> list[Chunk]:
    merged: list[tuple[list[str], int, int, str]] = []
    for crumb, start, end, body in _sections(doc.text):
        if merged and len(merged[-1][3]) < MIN_SECTION_CHARS:
            p_crumb, p_start, _, p_body = merged.pop()
            merged.append((p_crumb, p_start, end, p_body + "\n\n" + body))
        else:
            merged.append((crumb, start, end, body))

    chunks: list[Chunk] = []
    for crumb, start, end, body in merged:
        for part_idx, part in enumerate(_split_paragraphs(body)):
            breadcrumb = " > ".join(crumb) if crumb else doc.title
            title = f"{title_prefix} > {breadcrumb}" if title_prefix else breadcrumb
            suffix = f":{part_idx}" if part_idx else ""
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#L{start}{suffix}",
                    doc_id=doc.doc_id,
                    source_type=doc.source_type,
                    title=title,
                    text=part,
                    path=doc.path,
                    symbol=crumb[-1] if crumb else None,
                    start_line=start,
                    end_line=end,
                    author=doc.author,
                    updated_at=doc.updated_at,
                    url=f"{doc.url}#L{start}" if doc.url else None,
                    language="markdown",
                    pr_number=(doc.metadata.get("related_prs") or [None])[0],
                )
            )
    return chunks


def _split_paragraphs(body: str) -> list[str]:
    if len(body) <= MAX_SECTION_CHARS:
        return [body]
    parts: list[str] = []
    buf = ""
    for para in body.split("\n\n"):
        if buf and len(buf) + len(para) > MAX_SECTION_CHARS:
            parts.append(buf.strip())
            buf = ""
        buf += para + "\n\n"
    if buf.strip():
        parts.append(buf.strip())
    return parts
