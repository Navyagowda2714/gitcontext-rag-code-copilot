"""Hosted / local LLM providers. SDK imports are lazy so the package works without extras."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from .base import LLMResponse, Message, estimate_tokens

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
SMALL_ANTHROPIC_MODEL = "claude-haiku-5-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "llama3.1"


def _schema_instruction(json_schema: dict[str, Any] | None) -> str:
    if not json_schema:
        return ""
    return (
        "\n\nRespond with a single JSON object only (no prose, no code fences) that validates "
        f"against this JSON schema:\n{json.dumps(json_schema)}"
    )


class AnthropicProvider:
    """Claude via the official ``anthropic`` SDK (``pip install gitcontext[anthropic]``)."""

    def __init__(
        self, model: str | None = None, api_key: str | None = None, max_tokens: int = 1024
    ):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "Install the 'anthropic' extra: pip install gitcontext[anthropic]"
            ) from exc
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        start = time.perf_counter()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=(system or "") + _schema_instruction(json_schema),
            messages=[m.model_dump() for m in messages],
            temperature=temperature,
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        return LLMResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            model=self.model,
        )


class OpenAIProvider:
    """OpenAI chat completions (``pip install gitcontext[openai]``)."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError("Install the 'openai' extra: pip install gitcontext[openai]") from exc
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_OPENAI_MODEL)
        self._client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        start = time.perf_counter()
        payload: list[dict[str, str]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend(m.model_dump() for m in messages)
        kwargs: dict[str, Any] = {}
        if json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }
        resp = self._client.chat.completions.create(
            model=self.model, messages=payload, temperature=temperature, **kwargs
        )
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=(time.perf_counter() - start) * 1000,
            model=self.model,
        )


class OllamaProvider:
    """Local models through the Ollama HTTP API (no SDK needed)."""

    def __init__(self, model: str | None = None, base_url: str | None = None, timeout: float = 120):
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_OLLAMA_MODEL)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip(
            "/"
        )
        self.timeout = timeout

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        start = time.perf_counter()
        payload_msgs: list[dict[str, str]] = []
        if system:
            payload_msgs.append({"role": "system", "content": system})
        payload_msgs.extend(m.model_dump() for m in messages)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": payload_msgs,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_schema:
            body["format"] = json_schema
        resp = httpx.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("message", {}).get("content", "")
        return LLMResponse(
            text=text,
            input_tokens=data.get("prompt_eval_count") or estimate_tokens(str(payload_msgs)),
            output_tokens=data.get("eval_count") or estimate_tokens(text),
            latency_ms=(time.perf_counter() - start) * 1000,
            model=self.model,
        )
