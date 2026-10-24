"""Ingestion connectors. Each yields :class:`~gitcontext.models.Document` objects."""

from __future__ import annotations

from pathlib import Path

from ..models import Document
from .fixtures import iter_fixture_documents
from .git_repo import is_git_repo, iter_commits, iter_files, remote_url

__all__ = ["iter_commits", "iter_files", "iter_fixture_documents", "load_repository"]


def load_repository(
    root: Path,
    fixtures_dir: Path | None = None,
    repo_url: str | None = None,
    max_commits: int = 500,
) -> list[Document]:
    """Collect every source for a repository: files, commits and collaboration fixtures."""
    root = Path(root).resolve()
    repo_url = repo_url or (remote_url(root) if is_git_repo(root) else None)
    docs = list(iter_files(root, repo_url))
    docs.extend(iter_commits(root, max_commits=max_commits, repo_url=repo_url))
    fixtures_dir = fixtures_dir or root / ".gitcontext"
    docs.extend(iter_fixture_documents(fixtures_dir, repo_url))
    return docs
