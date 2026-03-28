"""Thin async wrapper around Ollama /api/chat with prompt-based tool-calling.

Gemma 3 doesn't support Ollama's native tool-calling API, so we embed tool
descriptions in the system prompt and parse structured JSON tool calls from
the model's text output.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Max tool-call rounds before we force a text reply
MAX_TOOL_ROUNDS = 5

# Regex to find tool call blocks in model output
# Match <tool_call>{...}</tool_call> — also handles markdown-fenced variants
# like ```tool_call>{...}</tool_call>``` that Gemma sometimes produces
_TOOL_CALL_RE = re.compile(
    r'`{0,3}\s*<?tool_call>?\s*(\{.*?\})\s*</?\s*tool_call\s*>?\s*`{0,3}',
    re.DOTALL,
)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ParsedResponse:
    """Parsed model response — may contain tool calls, text, or both."""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def build_tools_prompt(tools: list[dict]) -> str:
    """Build a system prompt section describing available tools."""
    lines = [
        "You have access to the following tools. To call a tool, output a "
        "<tool_call> block with a JSON object containing \"name\" and \"arguments\".\n",
        "Example:",
        '<tool_call>{"name": "web_search", "arguments": {"query": "example"}}</tool_call>\n',
        "You may call multiple tools. After you receive tool results, use them to "
        "form your final answer. When you're done, just respond with plain text "
        "(no <tool_call> tags).\n",
        "Available tools:",
    ]
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        param_strs = []
        for pname, pinfo in params.items():
            param_strs.append(f"  - {pname}: {pinfo.get('description', '')}")
        lines.append(f"\n**{name}**: {desc}")
        if param_strs:
            lines.extend(param_strs)

    return "\n".join(lines)


def tool_spec_to_ollama(name: str, description: str, parameters: dict[str, str]) -> dict:
    """Convert our ToolSpec parameters dict to a tool definition dict."""
    properties = {}
    for param_name, param_desc in parameters.items():
        properties[param_name] = {"type": "string", "description": param_desc}

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(parameters.keys()),
            },
        },
    }


# Gemma model artifacts that should never appear in user-facing output
_ARTIFACT_RE = re.compile(
    r'<start_of_image>|<end_of_turn>|<start_of_turn>|<bos>|<eos>|<pad>',
)


def sanitize_response(text: str) -> str:
    """Strip model artifacts and clean up response text."""
    text = _ARTIFACT_RE.sub("", text)
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def parse_tool_calls(text: str) -> ParsedResponse:
    """Parse tool calls from model output text."""
    tool_calls: list[ToolCall] = []

    for match in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(match.group(1))
            tool_calls.append(ToolCall(
                name=obj.get("name", ""),
                arguments=obj.get("arguments", {}),
            ))
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Failed to parse tool call: %s", match.group(0)[:200])

    # Strip tool_call blocks from text to get the plain-text portion
    clean_text = _TOOL_CALL_RE.sub("", text).strip()
    clean_text = sanitize_response(clean_text)

    return ParsedResponse(text=clean_text, tool_calls=tool_calls)


class OllamaChat:
    """Async Ollama chat client with prompt-based tool-calling loop."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma3:12b",
        timeout_s: int = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s

    def _call_api(self, messages: list[dict]) -> dict:
        """Blocking call to Ollama /api/chat. Run via asyncio.to_thread."""
        payload: dict[str, Any] = {
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
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            return json.loads(resp.read())

    async def chat(self, messages: list[dict]) -> str:
        """Single chat round — send messages, get text response."""
        raw = await asyncio.to_thread(self._call_api, messages)
        return raw.get("message", {}).get("content", "")

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_executor,
    ) -> str:
        """Full prompt-based tool-calling loop.

        1. Inject tool descriptions into system prompt
        2. Send to LLM
        3. Parse <tool_call> blocks from response
        4. Execute tools, append results
        5. Repeat until no more tool calls or MAX_TOOL_ROUNDS

        Returns the final text response.
        tool_executor is an async callable: (tool_name, **kwargs) -> str
        """
        # Build augmented messages with tool descriptions in system prompt
        tools_prompt = build_tools_prompt(tools)
        current_messages = list(messages)

        # Find and augment the system message, or prepend one
        has_system = any(m.get("role") == "system" for m in current_messages)
        if has_system:
            for i, m in enumerate(current_messages):
                if m.get("role") == "system":
                    current_messages[i] = {
                        "role": "system",
                        "content": m["content"] + "\n\n" + tools_prompt,
                    }
                    break
        else:
            current_messages.insert(0, {"role": "system", "content": tools_prompt})

        for round_num in range(MAX_TOOL_ROUNDS):
            raw_response = await self.chat(current_messages)
            parsed = parse_tool_calls(raw_response)

            if not parsed.has_tool_calls:
                # No tool calls — return the text as final answer
                return parsed.text or raw_response

            # Append the assistant's full response (including tool_call tags)
            current_messages.append({"role": "assistant", "content": raw_response})

            # Execute each tool call and feed results back
            tool_results: list[str] = []
            for tc in parsed.tool_calls:
                logger.info("Tool call [round %d]: %s(%s)", round_num + 1, tc.name, tc.arguments)
                try:
                    result = await tool_executor(tc.name, **tc.arguments)
                except Exception as e:
                    result = json.dumps({"error": str(e)})
                tool_results.append(f"Result from {tc.name}:\n{result}")

            # Feed tool results back as a user message
            current_messages.append({
                "role": "user",
                "content": "Tool results:\n\n" + "\n\n".join(tool_results)
                    + "\n\nNow use these results to answer the original question. "
                    "Do not call any more tools unless absolutely necessary.",
            })

        # Exhausted rounds — one final call to get text
        raw_response = await self.chat(current_messages)
        parsed = parse_tool_calls(raw_response)
        return parsed.text or raw_response
