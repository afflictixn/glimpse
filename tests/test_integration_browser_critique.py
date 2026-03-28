"""Integration test: BrowserContentAgent → IntelligenceLayer → CritiqueReasoningAgent → GeneralAgent.

Tests the full pipeline with mocked osascript and LLM to verify wiring end-to-end.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from src.intelligence.critique_agent import CritiqueReasoningAgent
from src.intelligence.intelligence_layer import IntelligenceLayer
from src.llm.types import Message
from src.process.browser_content_agent import BrowserContentAgent
from src.storage.models import Action, AppType, Event


def _make_allowlist(tmp_path: Path) -> Path:
    entries = [
        {"pattern": "amazon.com/", "selector": "#dp-container", "label": "Amazon product"},
    ]
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps(entries))
    return path


def _mock_osascript_sequence(outputs: list[tuple[str, int]]):
    """Mock osascript with a sequence of (stdout, returncode) pairs."""
    idx = 0

    async def _create(*args, **kwargs):
        nonlocal idx
        stdout, rc = outputs[idx % len(outputs)]
        idx += 1
        mock = AsyncMock()
        mock.communicate = AsyncMock(return_value=(stdout.encode(), b""))
        mock.returncode = rc
        mock.kill = MagicMock()
        return mock

    return _create


class TestFullPipeline:
    """Simulate the full flow: capture → BrowserContentAgent → IntelligenceLayer → CritiqueAgent."""

    @pytest.mark.asyncio
    async def test_allowlisted_page_triggers_critique(self, tmp_path):
        """An Amazon product page should:
        1. BrowserContentAgent extracts content
        2. Event flows to IntelligenceLayer
        3. CritiqueReasoningAgent analyzes and produces an Action
        4. Action is pushed to GeneralAgent
        """
        # -- Setup BrowserContentAgent --
        browser_agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))

        meta_js_result = json.dumps({
            "url": "https://www.amazon.com/dp/B09V3KXJPB",
            "title": "Wireless Earbuds",
            "meta": {"og:title": "Wireless Earbuds - Amazing Sound"},
        })
        full_js_result = json.dumps({
            "url": "https://www.amazon.com/dp/B09V3KXJPB",
            "title": "Wireless Earbuds",
            "meta": {"og:title": "Wireless Earbuds - Amazing Sound"},
            "text": (
                "Wireless Earbuds - Amazing Sound\n"
                "Price: $29.99\n"
                "Rating: 3.2 out of 5 stars (47 reviews)\n\n"
                "Top reviews:\n"
                "1 star - Battery died after 2 weeks\n"
                "1 star - Left earbud stopped working\n"
                "5 stars - Great for the price!\n"
                "2 stars - Uncomfortable after 30 minutes\n"
                "1 star - Arrived broken, no refund offered"
            ),
        })

        osascript_mock = _mock_osascript_sequence([
            (meta_js_result, 0),
            (full_js_result, 0),
        ])

        # -- Run BrowserContentAgent --
        img = Image.new("RGB", (100, 100))
        with patch(
            "src.process.browser_content_agent.asyncio.create_subprocess_exec",
            side_effect=osascript_mock,
        ):
            event = await browser_agent.process(img, "", "Google Chrome", "Wireless Earbuds")

        assert event is not None
        assert event.agent_name == "browser_content"
        assert event.app_type == AppType.BROWSER
        assert "text" in event.metadata
        assert "Battery died" in event.metadata["text"]
        assert event.metadata["allowlist_label"] == "Amazon product"

        # -- Setup CritiqueReasoningAgent with mock LLM --
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=Message(
            role="assistant",
            content=json.dumps({
                "findings": (
                    "Low rating (3.2/5) with only 47 reviews. "
                    "Multiple 1-star reviews report battery failure and broken units. "
                    "One reviewer reports no refund was offered."
                ),
                "severity": "warning",
            }),
        ))

        critique_agent = CritiqueReasoningAgent(llm=mock_llm)

        # -- Setup mock DB --
        mock_db = AsyncMock()
        mock_db.get_events_for_frame = AsyncMock(return_value=[
            {"agent_name": "gemma", "summary": "User browsing Amazon product page for wireless earbuds"},
        ])
        mock_db.insert_action = AsyncMock()

        # -- Setup mock GeneralAgent --
        mock_general_agent = AsyncMock()
        mock_general_agent.push = AsyncMock()

        # -- Wire IntelligenceLayer --
        intelligence = IntelligenceLayer(
            agents=[critique_agent],
            db=mock_db,
            general_agent=mock_general_agent,
        )

        # Simulate: give the event an id (as if DB assigned it)
        event.id = 42
        event.frame_id = 1

        # -- Submit event and process --
        await intelligence.submit(event)

        # Run the intelligence layer briefly to process the event
        run_task = asyncio.create_task(intelligence.run())
        await asyncio.sleep(0.3)
        await intelligence.stop()
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass

        # -- Verify CritiqueAgent was called --
        mock_llm.complete.assert_awaited_once()
        call_args = mock_llm.complete.call_args
        messages = call_args[0][0]
        assert any("consumer protection" in m.content.lower() for m in messages if m.role == "system")
        assert any("Battery died" in m.content for m in messages if m.role == "user")

        # -- Verify Action was stored in DB --
        mock_db.insert_action.assert_awaited_once()
        action = mock_db.insert_action.call_args[0][0]
        assert isinstance(action, Action)
        assert action.action_type == "critique"
        assert "3.2" in action.action_description
        assert action.metadata["severity"] == "warning"
        assert action.agent_name == "critique"

        # -- Verify Action was pushed to GeneralAgent --
        mock_general_agent.push.assert_awaited_once()
        push_call = mock_general_agent.push.call_args
        assert push_call[0][0] == "action"
        push_data = push_call[0][1]
        assert push_data["action_type"] == "critique"
        assert push_data["agent_name"] == "critique"

    @pytest.mark.asyncio
    async def test_non_allowlisted_page_skips_critique(self, tmp_path):
        """A non-allowlisted page should produce an Event but NOT trigger the CritiqueAgent."""
        browser_agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))

        js_result = json.dumps({
            "url": "https://www.example.com/random-page",
            "title": "Random Page",
            "meta": {},
        })
        osascript_mock = _mock_osascript_sequence([(js_result, 0)])

        img = Image.new("RGB", (100, 100))
        with patch(
            "src.process.browser_content_agent.asyncio.create_subprocess_exec",
            side_effect=osascript_mock,
        ):
            event = await browser_agent.process(img, "", "Google Chrome", "Random Page")

        assert event is not None
        assert "text" not in event.metadata

        mock_llm = AsyncMock()
        critique_agent = CritiqueReasoningAgent(llm=mock_llm)
        mock_db = AsyncMock()

        event.id = 1
        result = await critique_agent.reason(event, mock_db)

        assert result is None
        mock_llm.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_browser_app_produces_nothing(self, tmp_path):
        """A non-browser app should produce no Event at all."""
        browser_agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))

        img = Image.new("RGB", (100, 100))
        result = await browser_agent.process(img, "", "Xcode", "main.swift")
        assert result is None

    @pytest.mark.asyncio
    async def test_clean_page_produces_no_action(self, tmp_path):
        """A page the LLM deems clean should produce an Event but no Action."""
        browser_agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))

        meta_out = json.dumps({
            "url": "https://www.amazon.com/dp/B000001",
            "title": "Great Product",
            "meta": {},
        })
        full_out = json.dumps({
            "url": "https://www.amazon.com/dp/B000001",
            "title": "Great Product",
            "meta": {},
            "text": "Great Product\nPrice: $19.99\nRating: 4.8 out of 5 (2,340 reviews)\nTop reviews: all positive",
        })

        osascript_mock = _mock_osascript_sequence([
            (meta_out, 0),
            (full_out, 0),
        ])

        img = Image.new("RGB", (100, 100))
        with patch(
            "src.process.browser_content_agent.asyncio.create_subprocess_exec",
            side_effect=osascript_mock,
        ):
            event = await browser_agent.process(img, "", "Google Chrome", "Great Product")

        assert event is not None
        assert "text" in event.metadata

        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=Message(
            role="assistant",
            content=json.dumps({"findings": None}),
        ))

        critique_agent = CritiqueReasoningAgent(llm=mock_llm)
        mock_db = AsyncMock()
        mock_db.get_events_for_frame = AsyncMock(return_value=[])

        event.id = 1
        event.frame_id = 1
        result = await critique_agent.reason(event, mock_db)
        assert result is None
