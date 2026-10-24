"""Materialise the bundled sample repository as a real git repo with a scripted history.

Used by the demo, the evaluation and the tests, so every run indexes genuine ``git log`` output
(authors, dates, diffs) rather than mocked commits.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml


def _examples_dir() -> Path:
    """Locate ``examples/``: env override, the source checkout, then the working directory."""
    candidates = [Path(p) for p in (os.getenv("GITCONTEXT_EXAMPLES_DIR"),) if p] + [
        Path(__file__).resolve().parents[2] / "examples",
        Path.cwd() / "examples",
    ]
    return next((c for c in candidates if (c / "sample_repo").is_dir()), candidates[-1])


EXAMPLES_DIR = _examples_dir()
SAMPLE_REPO = EXAMPLES_DIR / "sample_repo"
SAMPLE_HISTORY = EXAMPLES_DIR / "sample_history.yaml"


def _all_files(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def _git(cwd: Path, *args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, env=env)


def plan_history(source: Path, history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return the full file snapshot after each commit by reverse-applying later edits."""
    final = _all_files(source)
    introduced: dict[str, int] = {}
    for i, commit in enumerate(history):
        for path in commit.get("add", []):
            introduced[path] = i
        for pattern in commit.get("add_glob", []):
            for p in source.glob(pattern):
                if p.is_file():
                    introduced[p.relative_to(source).as_posix()] = i
    for path in final:
        introduced.setdefault(path, 0)

    snapshots: list[dict[str, str]] = [{} for _ in history]
    current = dict(final)
    for i in range(len(history) - 1, -1, -1):
        snapshots[i] = {p: c for p, c in current.items() if introduced[p] <= i}
        for change in reversed(history[i].get("change", [])):
            path, old, new = change["path"], change.get("old", ""), change["new"]
            if new not in current[path]:
                raise ValueError(f"commit {i}: text to revert not found in {path}: {new!r}")
            current[path] = current[path].replace(new, old, 1)
    return snapshots


def build_sample_repo(
    dest: Path,
    source: Path = SAMPLE_REPO,
    history_file: Path = SAMPLE_HISTORY,
    remote: str | None = "https://github.com/shopfront/shopfront-api",
) -> Path:
    """Create ``dest`` as a git repository whose history follows ``history_file``."""
    history: list[dict[str, Any]] = yaml.safe_load(history_file.read_text(encoding="utf-8"))
    snapshots = plan_history(source, history)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    _git(dest, "init", "-q", "-b", "main")
    _git(dest, "config", "user.name", "Shopfront Bot")
    _git(dest, "config", "user.email", "bot@shopfront.dev")
    _git(dest, "config", "commit.gpgsign", "false")
    if remote:
        _git(dest, "remote", "add", "origin", remote)
    previous: dict[str, str] = {}
    for commit, snapshot in zip(history, snapshots, strict=True):
        for path in set(previous) - set(snapshot):
            (dest / path).unlink()
        for path, content in snapshot.items():
            if previous.get(path) != content:
                target = dest / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
        previous = snapshot
        name, _, email = commit["author"].partition(" <")
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": name,
            "GIT_AUTHOR_EMAIL": email.rstrip(">"),
            "GIT_AUTHOR_DATE": commit["date"],
            "GIT_COMMITTER_NAME": name,
            "GIT_COMMITTER_EMAIL": email.rstrip(">"),
            "GIT_COMMITTER_DATE": commit["date"],
        }
        message = commit["message"] + (
            "\n\n" + commit["body"].strip() if commit.get("body") else ""
        )
        _git(dest, "add", "-A")
        _git(dest, "commit", "-q", "--allow-empty", "-m", message, env=env)
    return dest
