"""``ask-repo`` command-line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from .config import RetrievalMode, get_settings
from .engine import GitContext
from .interfaces.render import render_answer, render_hits
from .logging_utils import configure_logging

app = typer.Typer(
    add_completion=False,
    help="GitContext: your codebase, answered. Index a repo, then ask it questions.",
    no_args_is_help=True,
)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging.")) -> None:
    configure_logging(logging.DEBUG if verbose else get_settings().log_level)


def _settings(index_dir: Path | None, repo_url: str | None = None):
    settings = get_settings()
    if index_dir is not None:
        settings.index_dir = index_dir
    if repo_url is not None:
        settings.repo_url = repo_url
    return settings


@app.command()
def index(
    path: Path = typer.Argument(..., exists=True, file_okay=False, help="Repository to index."),
    fixtures: Path | None = typer.Option(
        None, help="Directory with prs/, discussions/, incidents/ (default: <path>/.gitcontext)."
    ),
    index_dir: Path | None = typer.Option(None, help="Where to store the index."),
    repo_url: str | None = typer.Option(None, help="Base URL for links."),
) -> None:
    """Ingest code, git history, docs, PRs, discussions and postmortems."""
    settings = _settings(index_dir, repo_url)
    engine = GitContext.build(path, fixtures_dir=fixtures, settings=settings)
    meta = json.loads((settings.index_dir / "meta.json").read_text())
    typer.echo(f"Indexed {len(engine.chunks)} chunks into {settings.index_dir}")
    typer.echo(
        "  documents: " + ", ".join(f"{k}={v}" for k, v in sorted(meta["documents"].items()))
    )
    typer.echo("  chunks:    " + ", ".join(f"{k}={v}" for k, v in sorted(meta["chunks"].items())))


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question about the codebase."),
    mode: str | None = typer.Option(None, help="bm25 | vector | hybrid | hybrid_rerank"),
    index_dir: Path | None = typer.Option(None),
    as_json: bool = typer.Option(False, "--json", help="Emit the full Answer object as JSON."),
) -> None:
    """Answer a question with inline citations (or refuse when evidence is insufficient)."""
    engine = GitContext.load(settings=_settings(index_dir))
    answer = engine.ask(question, mode=mode)  # type: ignore[arg-type]
    if as_json:
        typer.echo(answer.model_dump_json(indent=2, exclude={"retrieved"}))
    else:
        typer.echo(render_answer(answer))


@app.command()
def search(
    query: str = typer.Argument(...),
    k: int = typer.Option(8, "--k", "-k", help="Number of results."),
    mode: str = typer.Option("hybrid_rerank", help="bm25 | vector | hybrid | hybrid_rerank"),
    index_dir: Path | None = typer.Option(None),
) -> None:
    """Show ranked retrieval results without generating an answer."""
    engine = GitContext.load(settings=_settings(index_dir))
    mode_t: RetrievalMode = mode  # type: ignore[assignment]
    typer.echo(render_hits(engine.search(query, k=k, mode=mode_t)))


@app.command("sample-repo")
def sample_repo(dest: Path = typer.Argument(Path(".cache/shopfront-api"))) -> None:
    """Materialise the bundled sample repository (with real git history) at DEST."""
    from .samples import build_sample_repo

    path = build_sample_repo(dest)
    typer.echo(f"Sample repository created at {path}")


@app.command("eval")
def run_evaluation(
    out: Path | None = typer.Option(None, help="Write the full JSON report here."),
) -> None:
    """Run the retrieval + answer evaluation on the bundled sample repo."""
    from .eval.runner import format_report, run_eval

    report = run_eval()
    typer.echo(format_report(report))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        typer.echo(f"\nFull report written to {out}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    index_dir: Path | None = typer.Option(None),
) -> None:
    """Serve the FastAPI app (POST /ask, GET /search, POST /index)."""
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(settings=_settings(index_dir)), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    app()
