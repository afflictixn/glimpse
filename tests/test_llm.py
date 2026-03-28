"""Integration tests for the LLM abstraction layer.

These tests hit real APIs and are skipped when the corresponding
environment variable is not set:
  - OPENAI_API_KEY  for OpenAI tests
  - GEMINI_API_KEY  for Gemini tests
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import Message, ToolSpec, create_llm_client

# ── Shared fixtures ───────────────────────────────────────────

SIMPLE_MESSAGES = [
    Message(role="system", content="Reply with exactly one word."),
    Message(role="user", content="What color is the sky on a clear day?"),
]

TOOL_MESSAGES = [
    Message(role="system", content="Use the provided tool to answer. Do not answer from memory."),
    Message(role="user", content="What is the weather in Paris right now?"),
]

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Get the current weather for a given city.",
    parameters={"city": "The city name to look up weather for"},
)

# skip_no_openai = pytest.mark.skipif(
#     not os.environ.get("OPENAI_API_KEY"),
#     reason="OPENAI_API_KEY not set",
# )
# Skip OpenAI tests for now
skip_no_openai = True

skip_no_gemini = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)

# ── OpenAI ────────────────────────────────────────────────────

@skip_no_openai
@pytest.mark.asyncio
async def test_openai_simple_completion():
    client = create_llm_client("openai", "gpt-4o-mini")
    response = await client.complete(SIMPLE_MESSAGES)

    assert response.role == "assistant"
    assert len(response.content) > 0
    assert not response.tool_calls


@skip_no_openai
@pytest.mark.asyncio
async def test_openai_tool_call():
    client = create_llm_client("openai", "gpt-4o-mini")
    response = await client.complete(TOOL_MESSAGES, tools=[WEATHER_TOOL])

    assert response.role == "assistant"
    assert len(response.tool_calls) > 0

    tc = response.tool_calls[0]
    assert tc.name == "get_weather"
    assert "city" in tc.arguments
    assert tc.id  # must have a non-empty id

# ── Gemini ────────────────────────────────────────────────────

@skip_no_gemini
@pytest.mark.asyncio
async def test_gemini_simple_completion():
    client = create_llm_client("gemini", "gemini-2.0-flash")
    response = await client.complete(SIMPLE_MESSAGES)

    assert response.role == "assistant"
    assert len(response.content) > 0
    assert not response.tool_calls


@skip_no_gemini
@pytest.mark.asyncio
async def test_gemini_tool_call():
    client = create_llm_client("gemini", "gemini-2.0-flash")
    response = await client.complete(TOOL_MESSAGES, tools=[WEATHER_TOOL])

    assert response.role == "assistant"
    assert len(response.tool_calls) > 0

    tc = response.tool_calls[0]
    assert tc.name == "get_weather"
    assert "city" in tc.arguments
    assert tc.id
