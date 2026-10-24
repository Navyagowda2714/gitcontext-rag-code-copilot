"""End-to-end offline demo.

1. Replays the bundled sample history into a real git repo (.cache/shopfront-api)
2. Indexes code, commits, PRs, discussions, docs and postmortems
3. Asks a few questions and prints grounded answers with citations

Run with ``make demo`` (offline) or set ``LLM_PROVIDER=anthropic`` + ``ANTHROPIC_API_KEY`` to use
Claude as the generator.
"""

from __future__ import annotations

from pathlib import Path

from gitcontext import GitContext
from gitcontext.config import Settings
from gitcontext.interfaces.render import render_answer
from gitcontext.samples import build_sample_repo

QUESTIONS = [
    "How does token refresh work?",
    "Why did customers receive duplicate invoices?",
    "Who changed the session manager to refresh tokens proactively, and when?",
    "How do we configure Kubernetes horizontal pod autoscaling?",
]


def main() -> None:
    cache = Path(".cache")
    repo = build_sample_repo(cache / "shopfront-api")
    settings = Settings(index_dir=cache / "index")
    engine = GitContext.build(repo, settings=settings)
    print(f"Indexed {len(engine.chunks)} chunks from {repo} (generator: {engine.llm.model})\n")
    for q in QUESTIONS:
        print("=" * 100)
        print(f"Q: {q}\n")
        print(render_answer(engine.ask(q)))
        print()


if __name__ == "__main__":
    main()
