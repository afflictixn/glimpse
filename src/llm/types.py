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
class ContentPart:
    """A single part of a multimodal message (text or image)."""
    type: str  # "text" or "image"
    text: str = ""
    image_data: bytes = b""
    mime_type: str = "image/jpeg"


@dataclass
class Message:
    """Provider-agnostic conversation message.

    role is one of: "system", "user", "assistant", "tool".
    content can be a plain string or a list of ContentPart for multimodal messages.
    tool_calls is populated when the assistant requests tool invocations.
    tool_call_id is set when role="tool" to tie a result back to a request.
    raw_provider_content may hold the original provider-specific object so it
    can be replayed verbatim (e.g. Gemini thought_signature preservation).
    """
    role: str
    content: str | list[ContentPart] = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""
    tool_name: str = ""
    raw_provider_content: Any = None


def text_content(msg: Message) -> str:
    """Extract text from content regardless of whether it's str or list[ContentPart]."""
    if isinstance(msg.content, str):
        return msg.content
    return "\n".join(p.text for p in msg.content if p.type == "text")
