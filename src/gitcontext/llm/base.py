"""Provider-agnostic LLM interface.

Every provider implements :class:`LLMClient`. Callers only depend on this module, so swapping
Anthropic for OpenAI, Ollama or the deterministic offline stand-in is a configuration change.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = "unknown"
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    def json(self) -> Any:  # type: ignore[override]
        """Parse the response text as JSON, tolerating markdown code fences."""
        return parse_json_loose(self.text)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat-completion contract shared by all providers."""

    model: str

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_loose(text: str) -> Any:
    """Parse JSON from model output that may be wrapped in fences or surrounded by prose."""
    match = _FENCE.search(text)
    candidate = match.group(1) if match else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = min((i for i in (candidate.find("{"), candidate.find("[")) if i >= 0), default=-1)
        if start < 0:
            raise
        end = max(candidate.rfind("}"), candidate.rfind("]"))
        return json.loads(candidate[start : end + 1])


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) used when a provider does not report usage."""
    return max(1, len(text) // 4)
