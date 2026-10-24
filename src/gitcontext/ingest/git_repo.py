"""Local git repository connector: working-tree files plus ``git log`` history (via subprocess)."""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from ..models import Document, SourceType

logger = logging.getLogger(__name__)

CODE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".rb": "ruby",
    ".kt": "kotlin",
    ".cs": "csharp",
    ".sql": "sql",
    ".sh": "shell",
}
DOC_EXTENSIONS = {".md", ".rst", ".txt"}
EXCLUDED_DIRS = {
    ".git",
    ".gitcontext",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
MAX_FILE_BYTES = 256_000
MIN_CODE_CHARS = 120


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def is_git_repo(root: Path) -> bool:
    try:
        return _git(root, "rev-parse", "--is-inside-work-tree").strip() == "true"
    except (RuntimeError, FileNotFoundError):
        return False


def remote_url(root: Path) -> str | None:
    """Return the ``origin`` remote as an https URL usable for links, if any."""
    try:
        url = _git(root, "config", "--get", "remote.origin.url").strip()
    except RuntimeError:
        return None
    if not url:
        return None
    ssh = re.match(r"git@([^:]+):(.+?)(?:\.git)?$", url)
    if ssh:
        return f"https://{ssh.group(1)}/{ssh.group(2)}"
    return url.removesuffix(".git")


def head_sha(root: Path) -> str | None:
    try:
        return _git(root, "rev-parse", "HEAD").strip()
    except RuntimeError:
        return None


def _last_touch_map(root: Path) -> dict[str, tuple[str, str, datetime]]:
    """Map path -> (sha, author, date) of the most recent commit touching it, in one git call."""
    out: dict[str, tuple[str, str, datetime]] = {}
    try:
        log = _git(root, "log", "--format=\x1e%H\x1f%an\x1f%aI", "--name-only")
    except RuntimeError:
        return out
    for record in log.split("\x1e"):
        if not record.strip():
            continue
        header, _, files = record.partition("\n")
        sha, author, date = header.split("\x1f")
        for f in files.split("\n"):
            f = f.strip()
            if f and f not in out:
                out[f] = (sha, author, datetime.fromisoformat(date))
    return out


def iter_files(root: Path, repo_url: str | None = None) -> Iterator[Document]:
    """Yield code and documentation files from the working tree."""
    root = root.resolve()
    touch = _last_touch_map(root) if is_git_repo(root) else {}
    sha = head_sha(root) if touch else None
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel.parts) or not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in CODE_EXTENSIONS:
            stype, language = SourceType.CODE, CODE_EXTENSIONS[ext]
        elif ext in DOC_EXTENSIONS:
            stype, language = SourceType.DOC, "markdown" if ext == ".md" else "text"
        else:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            logger.info("skipping large file %s", rel)
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if stype == SourceType.CODE and len(text.strip()) < MIN_CODE_CHARS:
            continue  # e.g. docstring-only __init__.py files add noise, not knowledge
        rel_str = rel.as_posix()
        last = touch.get(rel_str)
        url = f"{repo_url}/blob/{sha or 'HEAD'}/{rel_str}" if repo_url else None
        yield Document(
            doc_id=f"{stype.value}:{rel_str}",
            source_type=stype,
            title=rel_str,
            text=text,
            path=rel_str,
            url=url,
            author=last[1] if last else None,
            updated_at=last[2] if last else None,
            metadata={"language": language, "last_commit": last[0] if last else None},
        )


_PR_REF = re.compile(r"\(#(\d+)\)|(?:PR|pull request)\s*#(\d+)", re.IGNORECASE)


def iter_commits(
    root: Path, max_commits: int = 500, repo_url: str | None = None
) -> Iterator[Document]:
    """Yield one document per non-merge commit: message, author, date and changed files."""
    if not is_git_repo(root):
        return
    fmt = "\x1e%H\x1f%an\x1f%ae\x1f%aI\x1f%s\x1f%b\x1f"
    log = _git(root, "log", "--no-merges", f"-n{max_commits}", f"--format={fmt}", "--numstat")
    for record in log.split("\x1e"):
        if not record.strip():
            continue
        sha, author, email, date, subject, body, numstat = record.split("\x1f", 6)
        files: list[dict[str, str | int]] = []
        for line in numstat.strip().splitlines():
            parts = line.split("\t")
            if len(parts) == 3:
                added, deleted, fpath = parts
                files.append(
                    {
                        "path": fpath,
                        "added": int(added) if added.isdigit() else 0,
                        "deleted": int(deleted) if deleted.isdigit() else 0,
                    }
                )
        pr_match = _PR_REF.search(subject + "\n" + body)
        pr_number = int(pr_match.group(1) or pr_match.group(2)) if pr_match else None
        changed = "\n".join(f"  {f['path']} (+{f['added']} -{f['deleted']})" for f in files)
        text = f"{subject}\n\n{body.strip()}\n\nFiles changed:\n{changed}".strip()
        yield Document(
            doc_id=f"commit:{sha[:12]}",
            source_type=SourceType.COMMIT,
            title=subject,
            text=text,
            url=f"{repo_url}/commit/{sha}" if repo_url else None,
            author=author,
            updated_at=datetime.fromisoformat(date),
            metadata={"sha": sha, "email": email, "files": files, "pr_number": pr_number},
        )
