"""Structure-aware code chunking.

Python files are split along the AST: one chunk per top-level function, one per class (or one per
method for large classes, each carrying the class signature as context), plus a "module" chunk with
the docstring, imports and constants. Other languages use a declaration-boundary regex with a
fixed-window fallback.
"""

from __future__ import annotations

import ast
import re

from ..models import Chunk, Document

MAX_CHUNK_LINES = 80
MAX_CLASS_LINES = 40  # larger classes are split into one chunk per method
WINDOW_OVERLAP = 10


def _make_chunk(
    doc: Document,
    lines: list[str],
    start: int,
    end: int,
    symbol: str | None,
    prefix: str = "",
) -> Chunk:
    body = "\n".join(lines[start - 1 : end])
    text = f"{prefix}\n{body}" if prefix else body
    return Chunk(
        chunk_id=f"{doc.doc_id}#{symbol or 'L' + str(start)}",
        doc_id=doc.doc_id,
        source_type=doc.source_type,
        title=symbol or doc.title,
        text=text,
        path=doc.path,
        symbol=symbol,
        start_line=start,
        end_line=end,
        author=doc.author,
        commit_sha=doc.metadata.get("last_commit"),
        updated_at=doc.updated_at,
        url=f"{doc.url}#L{start}-L{end}" if doc.url else None,
        language=doc.metadata.get("language"),
    )


def _node_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([d.lineno for d in decorators] + [node.lineno])  # type: ignore[attr-defined]


def chunk_python(
    doc: Document, max_lines: int = MAX_CHUNK_LINES, max_class_lines: int = MAX_CLASS_LINES
) -> list[Chunk]:
    try:
        tree = ast.parse(doc.text)
    except SyntaxError:
        return chunk_generic(doc, max_lines)
    lines = doc.text.splitlines()
    chunks: list[Chunk] = []
    module_lines: list[int] = []
    defs = (ast.FunctionDef, ast.AsyncFunctionDef)

    for node in tree.body:
        start, end = _node_start(node), node.end_lineno or node.lineno
        if isinstance(node, defs):
            chunks.extend(_split_long(doc, lines, start, end, node.name, max_lines))
        elif isinstance(node, ast.ClassDef):
            if end - start + 1 <= max_class_lines:
                chunks.append(_make_chunk(doc, lines, start, end, node.name))
                continue
            methods = [n for n in node.body if isinstance(n, defs)]
            header_end = (_node_start(methods[0]) - 1) if methods else end
            chunks.append(_make_chunk(doc, lines, start, header_end, node.name))
            signature = lines[node.lineno - 1].strip()
            for m in methods:
                m_start, m_end = _node_start(m), m.end_lineno or m.lineno
                chunks.extend(
                    _split_long(
                        doc, lines, m_start, m_end, f"{node.name}.{m.name}", max_lines, signature
                    )
                )
        else:
            module_lines.extend(range(start, end + 1))

    if module_lines:
        text = "\n".join(lines[i - 1] for i in module_lines)
        if text.strip():
            chunk = _make_chunk(doc, lines, module_lines[0], module_lines[-1], None)
            chunk.text = text
            chunk.chunk_id = f"{doc.doc_id}#<module>"
            chunk.symbol = "<module>"
            chunk.title = f"{doc.path} (module)"
            chunks.insert(0, chunk)
    return chunks


def _split_long(
    doc: Document,
    lines: list[str],
    start: int,
    end: int,
    symbol: str,
    max_lines: int,
    prefix: str = "",
) -> list[Chunk]:
    if end - start + 1 <= max_lines:
        return [_make_chunk(doc, lines, start, end, symbol, prefix)]
    out: list[Chunk] = []
    step = max_lines - WINDOW_OVERLAP
    for i, s in enumerate(range(start, end + 1, step)):
        e = min(s + max_lines - 1, end)
        c = _make_chunk(doc, lines, s, e, symbol, prefix)
        c.chunk_id = f"{doc.doc_id}#{symbol}:{i}"
        out.append(c)
        if e == end:
            break
    return out


_DECL = re.compile(
    r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:"
    r"function\s*\*?\s*(?P<fn>\w+)"
    r"|(?:abstract\s+)?class\s+(?P<cls>\w+)"
    r"|(?:const|let)\s+(?P<arrow>\w+)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*(?::[^=]+)?=>"
    r"|interface\s+(?P<iface>\w+)"
    r"|func\s+(?:\([^)]*\)\s*)?(?P<go>\w+)"
    r"|(?:pub(?:\([^)]*\))?\s+)?fn\s+(?P<rs>\w+)"
    r"|def\s+(?P<rb>\w+)"
    r")"
)


def chunk_generic(doc: Document, max_lines: int = MAX_CHUNK_LINES) -> list[Chunk]:
    """Split on top-level declarations (JS/TS/Go/Rust/Ruby...); fall back to line windows."""
    lines = doc.text.splitlines()
    if not lines:
        return []
    boundaries: list[tuple[int, str | None]] = []
    for i, line in enumerate(lines, start=1):
        m = _DECL.match(line)
        if m:
            name = next((v for v in m.groupdict().values() if v), None)
            boundaries.append((i, name))
    chunks: list[Chunk] = []
    if not boundaries:
        return _split_long(doc, lines, 1, len(lines), doc.path or doc.doc_id, max_lines)
    if boundaries[0][0] > 1:
        header = _make_chunk(doc, lines, 1, boundaries[0][0] - 1, None)
        if header.text.strip():
            header.chunk_id, header.symbol = f"{doc.doc_id}#<module>", "<module>"
            header.title = f"{doc.path} (module)"
            chunks.append(header)
    for idx, (start, name) in enumerate(boundaries):
        end = boundaries[idx + 1][0] - 1 if idx + 1 < len(boundaries) else len(lines)
        while end > start and not lines[end - 1].strip():
            end -= 1
        chunks.extend(_split_long(doc, lines, start, end, name or f"L{start}", max_lines))
    return chunks
