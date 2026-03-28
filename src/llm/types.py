"""Model-agnostic types for the LLM abstraction layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A tool invocation requested by the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: str | None = None


@dataclass
class ToolSpec:
    """Portable tool definition passed to any LLM provider."""
    name: str
    description: str
    parameters: dict[str, str]


@dataclass
class Message:
    """Provider-agnostic conversation message.

    role is one of: "system", "user", "assistant", "tool".
    tool_calls is populated when the assistant requests tool invocations.
    tool_call_id is set when role="tool" to tie a result back to a request.
    raw_provider_content may hold the original provider-specific object so it
    can be replayed verbatim (e.g. Gemini thought_signature preservation).
    """
    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""
    tool_name: str = ""
    raw_provider_content: Any = None
