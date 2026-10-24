"""File-based connector for PRs, discussions and incident reports.

The GitHub API is optional, so collaboration artefacts can also be exported to a directory::

    .gitcontext/
      prs/134.json            # or .yaml
      discussions/27.yaml
      incidents/2026-03-14-token-refresh-storm.md   # markdown + YAML front matter

See ``examples/sample_repo/.gitcontext`` for the exact schema.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ..models import Document, SourceType

logger = logging.getLogger(__name__)


class Comment(BaseModel):
    author: str
    body: str
    created_at: datetime | None = None
    path: str | None = None
    line: int | None = None


class PullRequest(BaseModel):
    number: int
    title: str
    author: str
    body: str = ""
    state: str = "merged"
    url: str | None = None
    created_at: datetime | None = None
    merged_at: datetime | None = None
    labels: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    comments: list[Comment] = Field(default_factory=list)


class Discussion(BaseModel):
    number: int
    title: str
    author: str
    body: str = ""
    category: str | None = None
    url: str | None = None
    created_at: datetime | None = None
    comments: list[Comment] = Field(default_factory=list)


class Incident(BaseModel):
    id: str
    title: str
    severity: str | None = None
    date: datetime | None = None
    authors: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    related_prs: list[int] = Field(default_factory=list)
    url: str | None = None
    body: str


def _load_structured(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(raw)
    return yaml.safe_load(raw)


def _split_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    if raw.startswith("---"):
        _, fm, body = raw.split("---", 2)
        return yaml.safe_load(fm) or {}, body.strip()
    return {}, raw


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return None


def pr_to_document(pr: PullRequest, repo_url: str | None = None) -> Document:
    url = pr.url or (f"{repo_url}/pull/{pr.number}" if repo_url else None)
    return Document(
        doc_id=f"pr:{pr.number}",
        source_type=SourceType.PR,
        title=f"PR #{pr.number}: {pr.title}",
        text=pr.body,
        url=url,
        author=pr.author,
        updated_at=pr.merged_at or pr.created_at,
        metadata={
            "pr_number": pr.number,
            "state": pr.state,
            "labels": pr.labels,
            "files": pr.files,
            "comments": [c.model_dump(mode="json") for c in pr.comments],
        },
    )


def iter_fixture_documents(fixtures_dir: Path, repo_url: str | None = None) -> Iterator[Document]:
    if not fixtures_dir.is_dir():
        return
    for path in sorted((fixtures_dir / "prs").glob("*")):
        if path.suffix in {".json", ".yaml", ".yml"}:
            yield pr_to_document(PullRequest.model_validate(_load_structured(path)), repo_url)
    for path in sorted((fixtures_dir / "discussions").glob("*")):
        if path.suffix not in {".json", ".yaml", ".yml"}:
            continue
        d = Discussion.model_validate(_load_structured(path))
        yield Document(
            doc_id=f"discussion:{d.number}",
            source_type=SourceType.DISCUSSION,
            title=f"Discussion #{d.number}: {d.title}",
            text=d.body,
            url=d.url or (f"{repo_url}/discussions/{d.number}" if repo_url else None),
            author=d.author,
            updated_at=d.created_at,
            metadata={
                "number": d.number,
                "category": d.category,
                "comments": [c.model_dump(mode="json") for c in d.comments],
            },
        )
    for path in sorted((fixtures_dir / "incidents").glob("*")):
        if path.suffix == ".md":
            front, body = _split_front_matter(path.read_text(encoding="utf-8"))
            front["body"] = body
            front["date"] = _as_datetime(front.get("date"))
            inc = Incident.model_validate(front)
        elif path.suffix in {".json", ".yaml", ".yml"}:
            inc = Incident.model_validate(_load_structured(path))
        else:
            continue
        rel = path.relative_to(fixtures_dir.parent).as_posix()
        yield Document(
            doc_id=f"incident:{inc.id}",
            source_type=SourceType.INCIDENT,
            title=f"{inc.id}: {inc.title}",
            text=inc.body,
            path=rel,
            url=inc.url or (f"{repo_url}/blob/HEAD/{rel}" if repo_url else None),
            author=", ".join(inc.authors) or None,
            updated_at=inc.date,
            metadata={
                "incident_id": inc.id,
                "severity": inc.severity,
                "services": inc.services,
                "related_prs": inc.related_prs,
                "language": "markdown",
            },
        )
