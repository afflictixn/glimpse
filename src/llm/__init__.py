"""Shared LLM abstraction layer — provider-agnostic types, protocol, and factory."""

from src.llm.client import LLMClient, create_llm_client
from src.llm.types import Message, ToolCall, ToolSpec

__all__ = [
    "LLMClient",
    "Message",
    "ToolCall",
    "ToolSpec",
    "create_llm_client",
]
