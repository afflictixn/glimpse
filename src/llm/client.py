"""LLMClient protocol and factory."""
from __future__ import annotations

from typing import Protocol

from src.llm.types import Message, ToolSpec


class LLMClient(Protocol):
    """Provider-agnostic interface for LLM completions with optional tool use."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> Message: ...


def create_llm_client(provider: str, model: str, **kwargs: object) -> LLMClient:
    """Instantiate an LLMClient for the given provider and model.

    kwargs are forwarded to the provider constructor (e.g. api_key, temperature).
    """
    if provider == "openai":
        from src.llm.providers.openai import OpenAIClient
        return OpenAIClient(model=model, **kwargs)
    if provider == "gemini":
        from src.llm.providers.gemini import GeminiClient
        return GeminiClient(model=model, **kwargs)
    if provider == "ollama":
        from src.llm.providers.ollama import OllamaClient
        return OllamaClient(model=model, **kwargs)
    raise ValueError(f"Unknown LLM provider: {provider!r}")
