"""Tests for CritiqueReasoningAgent — LLM and DB are mocked."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.intelligence.critique_agent import CritiqueReasoningAgent
from src.llm.types import Message
from src.storage.models import Action, AppType, Event


def _browser_event(
    text: str | None = "Some page content",
    url: str = "https://amazon.com/product/123",
    label: str = "Amazon product",
    frame_id: int | None = 1,
    event_id: int | None = 10,
) -> Event:
    metadata = {
        "url": url,
        "title": "Test Product",
        "meta": {},
        "browser": "Google Chrome",
    }
    if text is not None:
        metadata["text"] = text
        metadata["allowlist_label"] = label
    return Event(
        frame_id=frame_id,
        agent_name="browser_content",
        app_type=AppType.BROWSER,
        summary=f"Viewing [{label}]: Test Product (amazon.com)",
        metadata=metadata,
        id=event_id,
    )


def _gemma_event(frame_id: int = 1) -> Event:
    return Event(
        frame_id=frame_id,
        agent_name="gemma",
        app_type=AppType.BROWSER,
        summary="User is browsing an Amazon product page for headphones",
    )


def _mock_llm(response_json: dict) -> AsyncMock:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=Message(
        role="assistant",
        content=json.dumps(response_json),
    ))
    return llm


def _mock_db(gemma_events: list[dict] | None = None) -> AsyncMock:
    db = AsyncMock()
    if gemma_events is None:
        gemma_events = [
            {"agent_name": "gemma", "summary": "User browsing product page"},
        ]
    db.get_events_for_frame = AsyncMock(return_value=gemma_events)
    return db


# ── Gating ──

class TestGating:
    @pytest.mark.asyncio
    async def test_non_browser_content_event_returns_none(self):
        llm = _mock_llm({"findings": "something"})
        agent = CritiqueReasoningAgent(llm=llm)

        event = Event(agent_name="gemma", app_type=AppType.BROWSER, summary="test")
        result = await agent.reason(event, _mock_db())

        assert result is None
        llm.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_browser_content_without_text_returns_none(self):
        llm = _mock_llm({"findings": "something"})
        agent = CritiqueReasoningAgent(llm=llm)

        event = _browser_event(text=None)
        result = await agent.reason(event, _mock_db())

        assert result is None
        llm.complete.assert_not_awaited()


# ── Successful critique ──

class TestCritique:
    @pytest.mark.asyncio
    async def test_findings_produce_action(self):
        findings = "Only 12 reviews, 3 mention battery dying after 1 month"
        llm = _mock_llm({"findings": findings, "severity": "warning"})
        agent = CritiqueReasoningAgent(llm=llm)
        db = _mock_db()

        event = _browser_event()
        result = await agent.reason(event, db)

        assert isinstance(result, Action)
        assert result.action_type == "critique"
        assert result.action_description == findings
        assert result.agent_name == "critique"
        assert result.metadata["severity"] == "warning"
        assert result.metadata["url"] == "https://amazon.com/product/123"
        assert result.event_id == 10
        assert result.frame_id == 1

    @pytest.mark.asyncio
    async def test_no_findings_returns_none(self):
        llm = _mock_llm({"findings": None})
        agent = CritiqueReasoningAgent(llm=llm)

        result = await agent.reason(_browser_event(), _mock_db())
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_findings_returns_none(self):
        llm = _mock_llm({"findings": ""})
        agent = CritiqueReasoningAgent(llm=llm)

        result = await agent.reason(_browser_event(), _mock_db())
        assert result is None


# ── DB context lookup ──

class TestDbContext:
    @pytest.mark.asyncio
    async def test_queries_db_for_gemma_event(self):
        llm = _mock_llm({"findings": "Issue found", "severity": "info"})
        agent = CritiqueReasoningAgent(llm=llm)
        db = _mock_db()

        event = _browser_event(frame_id=42)
        await agent.reason(event, db)

        db.get_events_for_frame.assert_awaited_once_with(42)

    @pytest.mark.asyncio
    async def test_no_frame_id_skips_db_query(self):
        llm = _mock_llm({"findings": "Issue found", "severity": "info"})
        agent = CritiqueReasoningAgent(llm=llm)
        db = _mock_db()

        event = _browser_event(frame_id=None)
        await agent.reason(event, db)

        db.get_events_for_frame.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_db_failure_still_produces_result(self):
        llm = _mock_llm({"findings": "Bad reviews", "severity": "warning"})
        agent = CritiqueReasoningAgent(llm=llm)
        db = AsyncMock()
        db.get_events_for_frame = AsyncMock(side_effect=Exception("DB error"))

        event = _browser_event()
        result = await agent.reason(event, db)

        assert isinstance(result, Action)
        assert result.action_description == "Bad reviews"


# ── LLM failure ──

class TestLlmFailure:
    @pytest.mark.asyncio
    async def test_llm_error_returns_none(self):
        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=Exception("API error"))
        agent = CritiqueReasoningAgent(llm=llm)

        result = await agent.reason(_browser_event(), _mock_db())
        assert result is None

    @pytest.mark.asyncio
    async def test_malformed_json_returns_none(self):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=Message(
            role="assistant",
            content="This is not JSON at all",
        ))
        agent = CritiqueReasoningAgent(llm=llm)

        result = await agent.reason(_browser_event(), _mock_db())
        assert result is None

    @pytest.mark.asyncio
    async def test_markdown_fenced_json_is_parsed(self):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=Message(
            role="assistant",
            content='```json\n{"findings": "Found issue", "severity": "info"}\n```',
        ))
        agent = CritiqueReasoningAgent(llm=llm)

        result = await agent.reason(_browser_event(), _mock_db())
        assert isinstance(result, Action)
        assert result.action_description == "Found issue"


# ── Agent properties ──

class TestAgentProperties:
    def test_name(self):
        llm = AsyncMock()
        agent = CritiqueReasoningAgent(llm=llm)
        assert agent.name == "critique"
