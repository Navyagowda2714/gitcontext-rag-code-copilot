"""Minimal Slack adapter (``pip install gitcontext[slack]``).

Responds to ``@gitcontext <question>`` mentions and the ``/ask-repo`` slash command in Socket Mode.
Environment: ``SLACK_BOT_TOKEN`` (xoxb-...), ``SLACK_APP_TOKEN`` (xapp-..., Socket Mode).
"""

from __future__ import annotations

import os
import re
from typing import Any

from ..engine import GitContext
from ..models import Answer

_CITE = re.compile(r"\[(\d+)\]")


def format_slack_message(answer: Answer) -> dict[str, Any]:
    """Render an answer as Slack Block Kit: citations become links, sources listed in context."""
    urls = {c.index: c.url for c in answer.citations}

    def link(m: re.Match[str]) -> str:
        idx = int(m.group(1))
        return f"<{urls[idx]}|[{idx}]>" if urls.get(idx) else f"[{idx}]"

    body = _CITE.sub(link, answer.answer).replace("**", "*")
    blocks: list[dict[str, Any]] = [{"type": "section", "text": {"type": "mrkdwn", "text": body}}]
    if answer.citations:
        sources = "\n".join(
            f"[{c.index}] {c.source_type.value} · "
            + (f"<{c.url}|{c.location}>" if c.url else c.location)
            for c in answer.citations
        )
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": sources}]})
    return {"text": answer.answer, "blocks": blocks}


def create_slack_app(engine: GitContext, bot_token: str | None = None) -> Any:
    try:
        from slack_bolt import App
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError("pip install gitcontext[slack] to use the Slack bot") from exc

    app = App(token=bot_token or os.environ["SLACK_BOT_TOKEN"])

    @app.event("app_mention")
    def on_mention(event: dict[str, Any], say: Any) -> None:  # pragma: no cover - needs Slack
        question = re.sub(r"<@[^>]+>", "", event.get("text", "")).strip()
        if question:
            say(**format_slack_message(engine.ask(question)), thread_ts=event.get("ts"))

    @app.command("/ask-repo")
    def on_command(ack: Any, command: dict[str, Any], respond: Any) -> None:  # pragma: no cover
        ack()
        respond(**format_slack_message(engine.ask(command.get("text", ""))))

    return app


def run_socket_mode() -> None:  # pragma: no cover - needs Slack credentials
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    app = create_slack_app(GitContext.load())
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":  # pragma: no cover
    run_socket_mode()
