"""LLM abstraction layer: one protocol, several providers, deterministic offline fallback."""

from __future__ import annotations

import logging
import os

from .base import LLMClient, LLMResponse, Message, parse_json_loose
from .offline import OfflineProvider, ScriptedLLM
from .providers import AnthropicProvider, OllamaProvider, OpenAIProvider

logger = logging.getLogger(__name__)

__all__ = [
    "AnthropicProvider",
    "LLMClient",
    "LLMResponse",
    "Message",
    "OfflineProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "ScriptedLLM",
    "get_llm",
    "parse_json_loose",
]


def get_llm(provider: str | None = None, model: str | None = None) -> LLMClient:
    """Build the configured provider (env ``LLM_PROVIDER``), falling back to offline when no key.

    Resolution order when ``LLM_PROVIDER`` is unset: Anthropic if ``ANTHROPIC_API_KEY`` is set,
    then OpenAI if ``OPENAI_API_KEY`` is set, otherwise the deterministic offline provider.
    """
    name = (provider or os.getenv("LLM_PROVIDER") or "").strip().lower()
    if not name:
        if os.getenv("ANTHROPIC_API_KEY"):
            name = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            name = "openai"
        else:
            name = "offline"
    try:
        if name == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            return AnthropicProvider(model=model)
        if name == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            return OpenAIProvider(model=model)
        if name == "ollama":
            return OllamaProvider(model=model)
        if name != "offline":
            raise RuntimeError(f"unknown LLM_PROVIDER '{name}'")
    except (ImportError, RuntimeError) as exc:
        logger.warning(
            "LLM provider '%s' unavailable (%s); falling back to offline mode", name, exc
        )
    return OfflineProvider()
