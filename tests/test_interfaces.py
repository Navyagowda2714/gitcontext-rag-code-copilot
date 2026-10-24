from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from gitcontext.api import create_app
from gitcontext.cli import app as cli_app
from gitcontext.engine import GitContext
from gitcontext.interfaces.slack_bot import format_slack_message
from gitcontext.llm import OfflineProvider, get_llm, parse_json_loose


def test_api_ask_search_and_health(engine: GitContext) -> None:
    client = TestClient(create_app(engine=engine))
    assert client.get("/health").json()["indexed_chunks"] == len(engine.chunks)
    body = client.post(
        "/ask", json={"question": "Why did customers receive duplicate invoices?"}
    ).json()
    assert not body["refused"]
    assert any(c["doc_id"] == "incident:INC-2026-022" for c in body["citations"])
    hits = client.get("/search", params={"q": "token bucket redis", "k": 3}).json()
    assert len(hits) == 3 and hits[0]["source_type"] in {"code", "pr", "commit", "doc"}
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_api_without_index_returns_503(tmp_path: Path) -> None:
    from gitcontext.config import Settings

    client = TestClient(create_app(settings=Settings(index_dir=tmp_path / "missing")))
    assert client.post("/ask", json={"question": "anything here?"}).status_code == 503


def test_cli_index_then_ask(
    sample_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    runner = CliRunner()
    idx = tmp_path / "idx"
    res = runner.invoke(cli_app, ["index", str(sample_repo), "--index-dir", str(idx)])
    assert res.exit_code == 0, res.output
    assert "incident=3" in res.output
    res = runner.invoke(
        cli_app, ["ask", "How are prorated amounts rounded?", "--index-dir", str(idx)]
    )
    assert res.exit_code == 0, res.output
    assert "Sources:" in res.output and "billing" in res.output


def test_slack_message_links_citations(engine: GitContext) -> None:
    msg = format_slack_message(engine.ask("How does token refresh work?"))
    assert "<https://github.com/shopfront/shopfront-api" in msg["blocks"][0]["text"]["text"]
    assert msg["blocks"][1]["type"] == "context"


def test_llm_factory_falls_back_to_offline(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm = get_llm()
    assert isinstance(llm, OfflineProvider)
    assert "falling back to offline" in caplog.text


def test_parse_json_loose_handles_fences_and_prose() -> None:
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_loose('Sure! Here it is: {"scores": [3, 1]} hope that helps') == {
        "scores": [3, 1]
    }
