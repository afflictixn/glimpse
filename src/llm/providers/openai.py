"""OpenAI provider — translates between internal types and the OpenAI Responses API."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from openai import AsyncOpenAI

from src.llm.types import Message, ToolCall, ToolSpec

logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT = 30


class OpenAIClient:
    """LLMClient implementation backed by the OpenAI Responses API."""

    def __init__(self, model: str = "gpt-5.4-mini", **kwargs: Any) -> None:
        self._model = model
        self._api_key: str | None = kwargs.pop("api_key", None)
        self._timeout: float = kwargs.pop("timeout", _DEFAULT_TIMEOUT)
        self._reasoning_effort: str | None = kwargs.pop("reasoning_effort", None)
        self._client = AsyncOpenAI(api_key=self._api_key)
        self._extra_kwargs = kwargs

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> Message:
        instructions, input_list = self._build_input(messages)

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "input": input_list,
            **self._extra_kwargs,
        }

        if instructions:
            call_kwargs["instructions"] = instructions

        if tools:
            call_kwargs["tools"] = [self._to_responses_tool(t) for t in tools]

        if self._reasoning_effort:
            call_kwargs["reasoning"] = {"effort": self._reasoning_effort}

        response = await asyncio.wait_for(
            self._client.responses.create(**call_kwargs),
            timeout=self._timeout,
        )
        return self._from_response(response)

    # ── Translators ───────────────────────────────────────────

    @staticmethod
    def _build_input(
        messages: list[Message],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Split system messages into instructions, convert the rest to Responses API input."""
        system_parts: list[str] = []
        input_list: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
                continue

            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    input_list.append({
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    })
                if msg.content:
                    input_list.append({"role": "assistant", "content": msg.content})
                continue

            if msg.role == "tool":
                input_list.append({
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id,
                    "output": msg.content,
                })
                continue

            input_list.append({"role": msg.role, "content": msg.content})

        instructions = "\n\n".join(system_parts) if system_parts else None
        return instructions, input_list

    @staticmethod
    def _to_responses_tool(spec: ToolSpec) -> dict[str, Any]:
        properties = {
            name: {"type": "string", "description": desc}
            for name, desc in spec.parameters.items()
        }
        return {
            "type": "function",
            "name": spec.name,
            "description": spec.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(spec.parameters.keys()),
            },
        }

    @staticmethod
    def _from_response(response: Any) -> Message:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if hasattr(content, "text"):
                        text_parts.append(content.text)
            elif item.type == "function_call":
                try:
                    args = json.loads(item.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {"_raw": item.arguments}
                tool_calls.append(ToolCall(
                    id=item.call_id,
                    name=item.name,
                    arguments=args,
                ))

        return Message(
            role="assistant",
            content="\n".join(text_parts),
            tool_calls=tool_calls,
        )
