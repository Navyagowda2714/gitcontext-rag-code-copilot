from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from gitcontext.ingest import iter_commits, iter_files, iter_fixture_documents
from gitcontext.ingest.github_api import GitHubConnector
from gitcontext.models import SourceType
from gitcontext.samples import plan_history


def test_commits_are_read_from_git_log_with_metadata(sample_repo: Path) -> None:
    commits = list(iter_commits(sample_repo, repo_url="https://example.com/r"))
    assert len(commits) == 15
    fix = next(c for c in commits if "proactively" in c.title)
    assert fix.author == "Priya Raman"
    assert fix.metadata["pr_number"] == 134
    assert fix.updated_at is not None and fix.updated_at.date().isoformat() == "2026-03-20"
    changed = {f["path"] for f in fix.metadata["files"]}
    assert {"shopfront/auth/session.py", "docs/auth.md"} <= changed
    assert fix.url == f"https://example.com/r/commit/{fix.metadata['sha']}"


def test_working_tree_files_skip_fixtures_and_carry_last_author(sample_repo: Path) -> None:
    docs = {d.path: d for d in iter_files(sample_repo)}
    assert "shopfront/auth/session.py" in docs
    assert not any(p.startswith(".gitcontext") for p in docs)
    assert "shopfront/api/__init__.py" not in docs  # docstring-only file is noise
    session = docs["shopfront/auth/session.py"]
    assert session.source_type == SourceType.CODE
    assert session.author == "Priya Raman"
    assert docs["docs/auth.md"].source_type == SourceType.DOC


def test_fixture_connector_parses_prs_discussions_and_postmortems(sample_repo: Path) -> None:
    docs = {d.doc_id: d for d in iter_fixture_documents(sample_repo / ".gitcontext")}
    assert {"pr:134", "pr:141", "pr:152", "pr:118", "discussion:27", "discussion:31"} <= set(docs)
    inc = docs["incident:INC-2026-014"]
    assert inc.metadata["severity"] == "SEV-2"
    assert inc.metadata["related_prs"] == [134]
    assert inc.text.startswith("# Postmortem")
    assert len(docs["pr:134"].metadata["comments"]) == 5


def test_github_connector_maps_api_payloads() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/pulls"):
            return httpx.Response(
                200,
                json=[
                    {
                        "number": 9,
                        "title": "Fix bug",
                        "user": {"login": "amy"},
                        "body": "Body",
                        "state": "closed",
                        "html_url": "https://github.com/o/r/pull/9",
                        "created_at": "2026-01-01T00:00:00Z",
                        "merged_at": "2026-01-02T00:00:00Z",
                        "labels": [{"name": "bug"}],
                    }
                ],
            )
        if path.endswith("/issues/9/comments"):
            return httpx.Response(
                200,
                json=[
                    {"user": {"login": "bob"}, "body": "LGTM", "created_at": "2026-01-01T01:00:00Z"}
                ],
            )
        if path.endswith("/pulls/9/comments"):
            return httpx.Response(200, json=[])
        if path.endswith("/pulls/9/files"):
            return httpx.Response(200, json=[{"filename": "a.py"}])
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.github.com")
    docs = GitHubConnector("o", "r", client=client).fetch_pull_requests(limit=5)
    assert len(docs) == 1
    pr = docs[0]
    assert pr.doc_id == "pr:9" and pr.metadata["state"] == "merged"
    assert pr.metadata["files"] == ["a.py"]
    assert pr.metadata["comments"][0]["author"] == "bob"


def test_history_planner_rejects_edits_that_do_not_match(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 2\n")
    history = [
        {"message": "init", "author": "A <a@x>", "date": "2026-01-01T00:00:00+00:00"},
        {
            "message": "bump",
            "author": "A <a@x>",
            "date": "2026-01-02T00:00:00+00:00",
            "change": [{"path": "a.py", "old": "x = 1", "new": "x = 2"}],
        },
    ]
    snaps = plan_history(tmp_path, history)
    assert snaps[0]["a.py"] == "x = 1\n" and snaps[1]["a.py"] == "x = 2\n"
    history[1]["change"][0]["new"] = "x = 3"
    with pytest.raises(ValueError):
        plan_history(tmp_path, history)
