from __future__ import annotations

from gitcontext.chunking import chunk_document, chunk_generic, chunk_markdown, chunk_python
from gitcontext.models import Document, SourceType

PY = (
    '''"""Module docstring."""
import os

LIMIT = 10


def helper(x: int) -> int:
    """Double it."""
    return x * 2


class Small:
    def a(self):
        return 1


class Big:
    """A large class."""

    attr = 1

    def first(self):
'''
    + "        x = 1\n" * 30
    + """        return x

    @staticmethod
    def second():
"""
    + "        y = 2\n" * 30
    + """        return y
"""
)


def _doc(text: str, path: str = "pkg/mod.py", language: str = "python") -> Document:
    return Document(
        doc_id=f"code:{path}",
        source_type=SourceType.CODE,
        title=path,
        text=text,
        path=path,
        metadata={"language": language},
    )


def test_python_chunks_follow_ast_symbols_and_lines() -> None:
    chunks = {c.symbol: c for c in chunk_python(_doc(PY))}
    assert {"<module>", "helper", "Small", "Big", "Big.first", "Big.second"} <= set(chunks)
    helper = chunks["helper"]
    lines = PY.splitlines()
    assert lines[helper.start_line - 1].startswith("def helper")
    assert lines[helper.end_line - 1].strip() == "return x * 2"
    # decorators belong to the method chunk
    assert lines[chunks["Big.second"].start_line - 1].strip() == "@staticmethod"


def test_module_chunk_keeps_constants_and_methods_keep_class_context() -> None:
    chunks = {c.symbol: c for c in chunk_python(_doc(PY))}
    assert "LIMIT = 10" in chunks["<module>"].text
    assert "def helper" not in chunks["<module>"].text
    assert chunks["Big.first"].text.startswith("class Big:")
    assert "Small.a" not in chunks  # small classes stay whole


def test_syntax_error_falls_back_to_generic_chunker() -> None:
    chunks = chunk_python(_doc("def broken(:\n    pass\n" * 3))
    assert chunks and all(c.text for c in chunks)


def test_regex_chunker_handles_typescript_declarations() -> None:
    ts = (
        "// header comment\nimport x from 'y';\n\n"
        "export interface Store {\n  get(): string;\n}\n\n"
        "export async function refreshToken(s: Store): Promise<string> {\n  return s.get();\n}\n\n"
        "export const apiFetch = async (url: string) => {\n  return fetch(url);\n};\n"
    )
    chunks = chunk_generic(_doc(ts, "web/client.ts", "typescript"))
    symbols = [c.symbol for c in chunks]
    assert symbols == ["<module>", "Store", "refreshToken", "apiFetch"]
    assert chunks[2].text.startswith("export async function refreshToken")


def test_markdown_chunks_by_heading_with_breadcrumbs_and_fences() -> None:
    md = (
        "# Auth\n\nIntro paragraph that is long enough to stand on its own as a section of the "
        "document, describing the authentication subsystem in a little detail, including "
        "sign-in flows and how sessions are stored.\n\n"
        "## Token refresh\n\nRefresh happens before expiry. It is long enough to be its own "
        "section because it contains several sentences of real explanation text here.\n\n"
        "```bash\n# not a heading\necho hi\n```\n"
    )
    doc = Document(
        doc_id="doc:auth.md", source_type=SourceType.DOC, title="auth.md", text=md, path="auth.md"
    )
    chunks = chunk_markdown(doc)
    assert [c.title for c in chunks] == ["Auth", "Auth > Token refresh"]
    assert "# not a heading" in chunks[1].text


def test_pr_threads_split_description_and_review_comments() -> None:
    doc = Document(
        doc_id="pr:7",
        source_type=SourceType.PR,
        title="PR #7: Add cache",
        text="Adds a cache.",
        metadata={
            "pr_number": 7,
            "files": ["a.py"],
            "comments": [
                {"author": "bob", "body": "Why TTL 60?", "path": "a.py", "line": 3},
                {"author": "amy", "body": "Matches upstream TTL."},
            ],
        },
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 2
    assert all(c.pr_number == 7 for c in chunks)
    assert "Files: a.py" in chunks[0].text
    assert "@bob on a.py:3: Why TTL 60?" in chunks[1].text
