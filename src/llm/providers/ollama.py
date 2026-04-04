"""Ollama provider — translates between internal types and Ollama's /api/chat."""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
import uuid
from typing import Any

from src.general_agent.ollama_client import parse_tool_calls, build_tools_prompt, tool_spec_to_ollama
from src.llm.types import Message, ToolCall, ToolSpec

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30


class OllamaClient:
    """LLMClient implementation backed by a local Ollama server."""

    def __init__(
        self,
        model: str = "gemma3:4b",
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout: float = kwargs.pop("timeout", _DEFAULT_TIMEOUT)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> Message:
        ollama_messages = self._to_ollama_messages(messages, tools)
        raw = await asyncio.to_thread(self._call_api, ollama_messages)
        content = raw.get("message", {}).get("content", "")

        # Parse any tool calls from the response text
        if tools:
            parsed = parse_tool_calls(content)
            if parsed.has_tool_calls:
                return Message(
                    role="assistant",
                    content=parsed.text,
                    tool_calls=[
                        ToolCall(
                            id=uuid.uuid4().hex[:8],
                            name=tc.name,
                            arguments=tc.arguments,
                        )
                        for tc in parsed.tool_calls
                    ],
                )

        return Message(role="assistant", content=content)

    def _to_ollama_messages(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> list[dict]:
        """Convert internal messages to Ollama format, injecting tool prompt if needed."""
        result: list[dict] = []

        for msg in messages:
            if msg.role == "system":
                content = msg.content
                # Inject tool descriptions into system prompt
                if tools and result == []:  # first system message
                    ollama_tools = [
                        tool_spec_to_ollama(t.name, t.description, t.parameters)
                        for t in tools
                    ]
                    content += "\n\n" + build_tools_prompt(ollama_tools)
                result.append({"role": "system", "content": content})
            elif msg.role == "tool":
                result.append({
                    "role": "user",
                    "content": f"Tool result from {msg.tool_name}:\n{msg.content}",
                })
            elif msg.role == "assistant":
                result.append({"role": "assistant", "content": msg.content})
            else:
                result.append({"role": "user", "content": msg.content})

        # If tools but no system message was present, prepend one
        if tools and not any(m.get("role") == "system" for m in result):
            ollama_tools = [
                tool_spec_to_ollama(t.name, t.description, t.parameters)
                for t in tools
            ]
            result.insert(0, {
                "role": "system",
                "content": build_tools_prompt(ollama_tools),
            })

        return result

    def _call_api(self, messages: list[dict]) -> dict:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512},
        }

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())
