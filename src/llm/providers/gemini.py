"""Gemini provider — translates between internal types and the google-genai SDK."""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from google import genai
from google.genai import types as gtypes

from src.llm.types import Message, ToolCall, ToolSpec

logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT = 30


class GeminiClient:
    """LLMClient implementation backed by the Google GenAI SDK."""

    def __init__(self, model: str = "gemini-2.0-flash", **kwargs: Any) -> None:
        self._model = model
        api_key: str | None = kwargs.pop("api_key", None)
        self._timeout: float = kwargs.pop("timeout", _DEFAULT_TIMEOUT)
        self._client = genai.Client(api_key=api_key)
        self._extra_kwargs = kwargs

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> Message:
        system_instruction, contents = self._split_system(messages)

        config_kwargs: dict[str, Any] = {**self._extra_kwargs}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        tools_list: list[gtypes.Tool] = [
            gtypes.Tool(google_search=gtypes.GoogleSearch()),
        ]
        if tools:
            tools_list.append(gtypes.Tool(function_declarations=[
                self._to_gemini_func(t) for t in tools
            ]))
        config_kwargs["tools"] = tools_list
        config_kwargs["tool_config"] = gtypes.ToolConfig(
            include_server_side_tool_invocations=True,
        )
        config = gtypes.GenerateContentConfig(**config_kwargs)

        response = await asyncio.wait_for(
            self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            ),
            timeout=self._timeout,
        )
        return self._from_gemini_response(response)

    # ── Translators ───────────────────────────────────────────

    @staticmethod
    def _split_system(
        messages: list[Message],
    ) -> tuple[str | None, list[gtypes.Content]]:
        """Separate system messages (-> system_instruction) from conversation turns."""
        system_parts: list[str] = []
        contents: list[gtypes.Content] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
                continue

            # For assistant messages with tool calls, replay the raw Gemini
            # Content object to preserve thought_signature on functionCall parts.
            # This is required by Gemini 3 models.
            if msg.role == "assistant" and msg.raw_provider_content is not None:
                contents.append(msg.raw_provider_content)
                continue

            role = "model" if msg.role == "assistant" else "user"

            parts: list[gtypes.Part] = []

            if msg.content:
                parts.append(gtypes.Part(text=msg.content))

            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append(gtypes.Part(
                        function_call=gtypes.FunctionCall(
                            name=tc.name,
                            args=tc.arguments,
                            id=tc.id,
                        )
                    ))

            if msg.role == "tool":
                parts = [gtypes.Part(
                    function_response=gtypes.FunctionResponse(
                        name=msg.tool_name or msg.tool_call_id,
                        id=msg.tool_call_id,
                        response={"result": msg.content},
                    )
                )]
                role = "user"

            if parts:
                contents.append(gtypes.Content(role=role, parts=parts))

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return system_instruction, contents

    @staticmethod
    def _to_gemini_func(spec: ToolSpec) -> gtypes.FunctionDeclaration:
        properties: dict[str, Any] = {
            name: gtypes.Schema(type="STRING", description=desc)
            for name, desc in spec.parameters.items()
        }
        return gtypes.FunctionDeclaration(
            name=spec.name,
            description=spec.description,
            parameters=gtypes.Schema(
                type="OBJECT",
                properties=properties,
                required=list(spec.parameters.keys()),
            ),
        )

    @staticmethod
    def _from_gemini_response(response: Any) -> Message:
        candidate = response.candidates[0]
        content_parts = candidate.content.parts

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for part in content_parts:
            if part.text:
                text_parts.append(part.text)
            if part.function_call:
                fc = part.function_call
                call_id = getattr(fc, "id", None) or uuid.uuid4().hex[:8]
                tool_calls.append(ToolCall(
                    id=call_id,
                    name=fc.name,
                    arguments=dict(fc.args) if fc.args else {},
                ))

        citations = _extract_citations(candidate)
        if citations:
            text_parts.append(citations)

        return Message(
            role="assistant",
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            raw_provider_content=candidate.content,
        )


def _extract_citations(candidate: Any) -> str:
    """Build a sources footer from grounding metadata, if present."""
    metadata = getattr(candidate, "grounding_metadata", None)
    if not metadata:
        return ""
    chunks = getattr(metadata, "grounding_chunks", None)
    if not chunks:
        return ""

    seen: set[str] = set()
    lines: list[str] = []
    for chunk in chunks:
        web = getattr(chunk, "web", None)
        if not web:
            continue
        uri = getattr(web, "uri", "")
        title = getattr(web, "title", "")
        if uri and uri not in seen:
            seen.add(uri)
            label = title or uri
            lines.append(f"- [{label}]({uri})")

    if not lines:
        return ""
    return "\n\nSources:\n" + "\n".join(lines)
