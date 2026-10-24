"""Optional GitHub REST connector for pull requests and their review discussion.

Requires a token in ``GITHUB_TOKEN`` for private repos / higher rate limits. Not used by the offline
demo; the fixture connector covers the same schema without network access.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from ..models import Document
from .fixtures import Comment, PullRequest, pr_to_document

API = "https://api.github.com"


class GitHubConnector:
    def __init__(
        self,
        owner: str,
        repo: str,
        token: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.owner, self.repo = owner, repo
        token = token or os.getenv("GITHUB_TOKEN")
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = client or httpx.Client(base_url=API, headers=headers, timeout=30)

    def _get(self, path: str, **params: Any) -> Any:
        resp = self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def fetch_pull_requests(self, state: str = "closed", limit: int = 50) -> list[Document]:
        base = f"/repos/{self.owner}/{self.repo}"
        pulls = self._get(f"{base}/pulls", state=state, per_page=min(limit, 100))[:limit]
        docs: list[Document] = []
        for p in pulls:
            number = p["number"]
            comments = [
                Comment(author=c["user"]["login"], body=c["body"], created_at=c["created_at"])
                for c in self._get(f"{base}/issues/{number}/comments")
            ]
            comments += [
                Comment(
                    author=c["user"]["login"],
                    body=c["body"],
                    created_at=c["created_at"],
                    path=c.get("path"),
                    line=c.get("line"),
                )
                for c in self._get(f"{base}/pulls/{number}/comments")
            ]
            files = [f["filename"] for f in self._get(f"{base}/pulls/{number}/files")]
            pr = PullRequest(
                number=number,
                title=p["title"],
                author=p["user"]["login"],
                body=p.get("body") or "",
                state="merged" if p.get("merged_at") else p["state"],
                url=p["html_url"],
                created_at=p["created_at"],
                merged_at=p.get("merged_at"),
                labels=[lbl["name"] for lbl in p.get("labels", [])],
                files=files,
                comments=comments,
            )
            docs.append(pr_to_document(pr))
        return docs
