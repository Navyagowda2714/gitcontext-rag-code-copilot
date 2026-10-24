"""Source-aware chunking: code by symbol, docs by heading, commits whole, threads by comment."""

from __future__ import annotations

from ..models import Chunk, Document, SourceType
from .code import chunk_generic, chunk_python
from .markdown import chunk_markdown

THREAD_CHUNK_CHARS = 1200

__all__ = ["chunk_document", "chunk_documents", "chunk_generic", "chunk_markdown", "chunk_python"]


def _record_chunk(doc: Document, idx: int, title: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}#{idx}",
        doc_id=doc.doc_id,
        source_type=doc.source_type,
        title=title,
        text=text,
        path=doc.path,
        author=doc.author,
        pr_number=doc.metadata.get("pr_number") or doc.metadata.get("number"),
        commit_sha=doc.metadata.get("sha"),
        updated_at=doc.updated_at,
        url=doc.url,
    )


def _chunk_thread(doc: Document) -> list[Chunk]:
    """PRs and discussions: description first, then review comments grouped into chunks."""
    head = doc.text
    files = doc.metadata.get("files") or []
    if files:
        head += "\n\nFiles: " + ", ".join(str(f) for f in files)
    chunks = [_record_chunk(doc, 0, doc.title, f"{doc.title}\n\n{head}".strip())]
    buf: list[str] = []
    size = 0
    for c in doc.metadata.get("comments", []):
        where = (
            f" on {c['path']}" + (f":{c['line']}" if c.get("line") else "") if c.get("path") else ""
        )
        line = f"@{c['author']}{where}: {c['body'].strip()}"
        if buf and size + len(line) > THREAD_CHUNK_CHARS:
            chunks.append(
                _record_chunk(doc, len(chunks), f"{doc.title} (discussion)", "\n\n".join(buf))
            )
            buf, size = [], 0
        buf.append(line)
        size += len(line)
    if buf:
        chunks.append(
            _record_chunk(doc, len(chunks), f"{doc.title} (discussion)", "\n\n".join(buf))
        )
    return chunks


def chunk_document(doc: Document) -> list[Chunk]:
    if doc.source_type == SourceType.CODE:
        if doc.metadata.get("language") == "python":
            return chunk_python(doc)
        return chunk_generic(doc)
    if doc.source_type in (SourceType.DOC, SourceType.INCIDENT):
        return chunk_markdown(
            doc, title_prefix=doc.title if doc.source_type == SourceType.INCIDENT else None
        )
    if doc.source_type == SourceType.COMMIT:
        sha = doc.metadata.get("sha", "")
        chunk = _record_chunk(doc, 0, doc.title, doc.text)
        chunk.chunk_id = f"{doc.doc_id}#0"
        chunk.commit_sha = sha
        return [chunk]
    return _chunk_thread(doc)


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d)]
