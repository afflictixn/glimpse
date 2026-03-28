"""Tests for the SocialContextAgent — messaging app detection, name extraction,
and context gathering pipeline."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.process.social_context_agent import SocialContextAgent
from src.storage.models import AppType


def _make_image():
    return Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))


def _make_mock_tools():
    """Create a mock ToolRegistry."""
    tools = AsyncMock()
    tools.call = AsyncMock(return_value=json.dumps([]))
    return tools


class TestMessagingAppDetection:
    def test_detects_messages_app(self):
        agent = SocialContextAgent(tools=_make_mock_tools())
        assert agent._is_messaging_app("Messages", None)
        assert agent._is_messaging_app("messages", None)

    def test_detects_whatsapp(self):
        agent = SocialContextAgent(tools=_make_mock_tools())
        assert agent._is_messaging_app("WhatsApp", None)

    def test_detects_telegram(self):
        agent = SocialContextAgent(tools=_make_mock_tools())
        assert agent._is_messaging_app("Telegram", None)

    def test_detects_slack(self):
        agent = SocialContextAgent(tools=_make_mock_tools())
        assert agent._is_messaging_app("Slack", None)

    def test_ignores_non_messaging(self):
        agent = SocialContextAgent(tools=_make_mock_tools())
        assert not agent._is_messaging_app("Safari", None)
        assert not agent._is_messaging_app("Code", None)
        assert not agent._is_messaging_app(None, None)

    def test_detects_web_messenger_in_window(self):
        agent = SocialContextAgent(tools=_make_mock_tools())
        assert agent._is_messaging_app("Chrome", "WhatsApp Web")
        assert agent._is_messaging_app("Safari", "Telegram Web")


@pytest.mark.asyncio
class TestNameExtraction:
    async def test_extracts_names(self):
        agent = SocialContextAgent(
            tools=_make_mock_tools(),
            ollama_base_url="http://localhost:11434",
        )
        # Mock the Ollama call
        agent._call_ollama = MagicMock(return_value='["John Smith", "Alice"]')

        names = await agent._extract_names("John Smith: Hey how are you?\nAlice: Good thanks!")
        assert names == ["John Smith", "Alice"]

    async def test_handles_empty_response(self):
        agent = SocialContextAgent(tools=_make_mock_tools())
        agent._call_ollama = MagicMock(return_value="[]")

        names = await agent._extract_names("some text")
        assert names == []

    async def test_handles_malformed_response(self):
        agent = SocialContextAgent(tools=_make_mock_tools())
        agent._call_ollama = MagicMock(return_value="not json at all")

        names = await agent._extract_names("some text")
        assert names == []

    async def test_handles_ollama_failure(self):
        agent = SocialContextAgent(tools=_make_mock_tools())
        agent._call_ollama = MagicMock(side_effect=Exception("connection refused"))

        names = await agent._extract_names("some text")
        assert names == []

    async def test_strips_markdown_fences(self):
        agent = SocialContextAgent(tools=_make_mock_tools())
        agent._call_ollama = MagicMock(return_value='```json\n["Bob"]\n```')

        names = await agent._extract_names("Bob: hello")
        assert names == ["Bob"]

    async def test_limits_to_5_names(self):
        agent = SocialContextAgent(tools=_make_mock_tools())
        agent._call_ollama = MagicMock(
            return_value='["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]'
        )

        names = await agent._extract_names("text")
        assert len(names) == 5


@pytest.mark.asyncio
class TestContextGathering:
    async def test_gathers_imessage_and_contacts(self):
        tools = _make_mock_tools()

        async def mock_call(tool_name, **kwargs):
            if tool_name == "imessage_conversation":
                return json.dumps([
                    {"from": "John", "text": "See you tomorrow!", "timestamp": "2026-03-28 10:00"},
                ])
            if tool_name == "contacts_search":
                return json.dumps([{
                    "given_name": "John",
                    "family_name": "Smith",
                    "birthday": "3-15",
                    "organization": "Acme Corp",
                }])
            if tool_name == "contact_lookup":
                return json.dumps({"ocr_mentions": [], "event_mentions": []})
            if tool_name == "memory_query":
                return json.dumps([{"fact": "Met at conference last year"}])
            return json.dumps([])

        tools.call = mock_call
        agent = SocialContextAgent(tools=tools)

        context = await agent._gather_context(["John Smith"])
        assert "John Smith" in context
        ctx = context["John Smith"]
        assert "recent_messages" in ctx
        assert "contact_info" in ctx
        assert ctx["contact_info"]["birthday"] == "3-15"
        assert "memories" in ctx

    async def test_handles_tool_failures_gracefully(self):
        tools = _make_mock_tools()
        tools.call = AsyncMock(side_effect=Exception("tool broken"))

        agent = SocialContextAgent(tools=tools)
        context = await agent._gather_context(["Someone"])
        # Should not crash, just return empty context
        assert context == {} or "Someone" not in context or context["Someone"] == {}


@pytest.mark.asyncio
class TestBuildSummary:
    async def test_builds_summary_with_birthday(self):
        agent = SocialContextAgent(tools=_make_mock_tools())
        context = {
            "Alice": {
                "recent_messages": [{"text": "My birthday is April 1st!"}],
                "memories": [{"fact": "Likes hiking"}],
            }
        }
        analysis = {
            "insights": ["Alice's birthday is April 1st"],
            "birthday": "April 1st",
            "action_items": [],
        }
        summary = agent._build_summary(["Alice"], context, "Messages", analysis)
        assert "Alice" in summary
        assert "April 1st" in summary
        assert "hiking" in summary

    async def test_empty_summary_when_no_insights(self):
        agent = SocialContextAgent(tools=_make_mock_tools())
        context = {"Bob": {"recent_messages": [{"text": "hey"}]}}
        analysis = {"insights": [], "birthday": None, "action_items": []}
        summary = agent._build_summary(["Bob"], context, "Messages", analysis)
        assert summary == ""


@pytest.mark.asyncio
class TestFullPipeline:
    async def test_skips_non_messaging_app(self):
        agent = SocialContextAgent(tools=_make_mock_tools())
        result = await agent.process(_make_image(), "some text", "Safari", "Google")
        assert result is None

    async def test_skips_empty_ocr(self):
        agent = SocialContextAgent(tools=_make_mock_tools())
        result = await agent.process(_make_image(), "", "Messages", None)
        assert result is None

    async def test_skips_when_no_names_found(self):
        agent = SocialContextAgent(tools=_make_mock_tools())
        agent._call_ollama = MagicMock(return_value="[]")
        result = await agent.process(
            _make_image(),
            "Random text with no clear contact names visible on screen",
            "Messages",
            None,
        )
        assert result is None

    async def test_returns_event_when_context_found(self):
        tools = _make_mock_tools()

        async def mock_call(tool_name, **kwargs):
            if tool_name == "imessage_conversation":
                return json.dumps([{"from": "Bob", "text": "My birthday is March 30th! Let's celebrate"}])
            if tool_name == "contacts_search":
                return json.dumps([{"given_name": "Bob"}])
            return json.dumps([])

        tools.call = mock_call
        agent = SocialContextAgent(tools=tools)
        agent._call_ollama = MagicMock(return_value='["Bob"]')

        # Mock the analysis to return real extracted facts
        async def mock_analyze(names, context):
            return {
                "insights": ["Bob's birthday is March 30th", "Bob wants to celebrate"],
                "birthday": "March 30th",
                "action_items": ["Plan celebration with Bob"],
            }
        agent._analyze_messages = mock_analyze

        result = await agent.process(
            _make_image(),
            "Bob: Hey! My birthday is March 30th! Let's celebrate over coffee",
            "Messages",
            "Bob - iMessage",
        )

        assert result is not None
        assert result.agent_name == "social_context"
        assert "Bob" in result.summary
        assert "March 30th" in result.summary
        assert "Bob" in result.metadata["contacts"]
