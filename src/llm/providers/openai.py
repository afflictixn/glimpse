"""OpenAI provider — translates between internal types and the OpenAI chat completions API."""
from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from src.llm.types import Message, ToolCall, ToolSpec

logger = logging.getLogger(__name__)


class OpenAIClient:
    """LLMClient implementation backed by the OpenAI SDK."""

    def __init__(self, model: str = "gpt-4o-mini", **kwargs: Any) -> None:
        self._model = model
        self._api_key: str | None = kwargs.pop("api_key", None)
        self._client = AsyncOpenAI(api_key=self._api_key)
        self._extra_kwargs = kwargs

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> Message:
        oai_messages = [self._to_oai_message(m) for m in messages]
        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": oai_messages,
            **self._extra_kwargs,
        }

        if tools:
            call_kwargs["tools"] = [self._to_oai_tool(t) for t in tools]

        response = await self._client.chat.completions.create(**call_kwargs)
        choice = response.choices[0]
        return self._from_oai_choice(choice)

    # ── Translators ───────────────────────────────────────────

    @staticmethod
    def _to_oai_message(msg: Message) -> dict[str, Any]:
        out: dict[str, Any] = {"role": msg.role}

        if msg.content:
            out["content"] = msg.content

        if msg.role == "assistant" and msg.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
            out.setdefault("content", None)

        if msg.role == "tool":
            out["tool_call_id"] = msg.tool_call_id
            out.setdefault("content", "")

        return out

    @staticmethod
    def _to_oai_tool(spec: ToolSpec) -> dict[str, Any]:
        properties = {
            name: {"type": "string", "description": desc}
            for name, desc in spec.parameters.items()
        }
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(spec.parameters.keys()),
                },
            },
        }

    @staticmethod
    def _from_oai_choice(choice: Any) -> Message:
        msg = choice.message
        content = msg.content or ""
        tool_calls: list[ToolCall] = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {"_raw": tc.function.arguments}

                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        return Message(role="assistant", content=content, tool_calls=tool_calls)
