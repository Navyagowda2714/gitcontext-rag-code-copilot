"""Call the FastAPI app in-process (no server needed) and print the JSON response.

Against a running server (``make serve``) the equivalent is::

    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question": "How does token refresh work?"}' | jq
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from gitcontext import GitContext
from gitcontext.api import create_app
from gitcontext.config import Settings
from gitcontext.samples import build_sample_repo


def main() -> None:
    repo = build_sample_repo(Path(".cache/shopfront-api"))
    engine = GitContext.build(repo, settings=Settings(index_dir=Path(".cache/index")))
    client = TestClient(create_app(engine=engine))
    resp = client.post(
        "/ask", json={"question": "Why didn't we use a distributed Redis lock for token refresh?"}
    )
    body = resp.json()
    print(json.dumps({k: body[k] for k in ("answer", "refused", "evidence_score")}, indent=2))
    for c in body["citations"]:
        print(f"[{c['index']}] {c['source_type']:<8} {c['location']}  {c['url']}")


if __name__ == "__main__":
    main()
