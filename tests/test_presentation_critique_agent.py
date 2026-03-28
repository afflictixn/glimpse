"""Tests for PresentationCritiqueAgent — Ollama and DB are mocked."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.intelligence.presentation_critique_agent import PresentationCritiqueAgent
from src.storage.models import Action, AppType, Event


def _presentation_event(
    frame_id: int | None = 1,
    event_id: int | None = 10,
    summary: str = "User viewing a presentation slide",
) -> Event:
    return Event(
        frame_id=frame_id,
        agent_name="gemma",
        app_type=AppType.PRESENTATION,
        summary=summary,
        metadata={"app_name": "Keynote"},
        id=event_id,
    )


def _mock_db(snapshot_path: str | None = "/tmp/test_snapshot.jpg") -> AsyncMock:
    db = AsyncMock()
    if snapshot_path:
        db.get_frame = AsyncMock(return_value={"snapshot_path": snapshot_path})
    else:
        db.get_frame = AsyncMock(return_value=None)
    return db


def _dummy_image():
    from PIL import Image
    return Image.new("RGB", (200, 150), color=(255, 255, 255))


# ── Gating ──

class TestGating:
    @pytest.mark.asyncio
    async def test_non_presentation_event_returns_none(self):
        agent = PresentationCritiqueAgent()
        event = Event(agent_name="gemma", app_type=AppType.BROWSER, summary="browsing")
        result = await agent.reason(event, _mock_db())
        assert result is None

    @pytest.mark.asyncio
    async def test_other_app_type_returns_none(self):
        agent = PresentationCritiqueAgent()
        event = Event(agent_name="gemma", app_type=AppType.IDE, summary="coding")
        result = await agent.reason(event, _mock_db())
        assert result is None

    @pytest.mark.asyncio
    async def test_no_frame_id_returns_none(self):
        agent = PresentationCritiqueAgent()
        event = _presentation_event(frame_id=None)
        result = await agent.reason(event, _mock_db())
        assert result is None

    @pytest.mark.asyncio
    async def test_no_snapshot_returns_none(self):
        agent = PresentationCritiqueAgent()
        db = _mock_db(snapshot_path=None)
        result = await agent.reason(_presentation_event(), db)
        assert result is None

    @pytest.mark.asyncio
    async def test_nonexistent_snapshot_file_returns_none(self):
        agent = PresentationCritiqueAgent()
        db = _mock_db(snapshot_path="/nonexistent/path/snap.jpg")
        result = await agent.reason(_presentation_event(), db)
        assert result is None


# ── Successful critique ──

class TestCritique:
    @pytest.mark.asyncio
    async def test_findings_produce_action(self):
        agent = PresentationCritiqueAgent()
        critique_points = [
            "Title text is too small for readability at distance.",
            "Low contrast between grey text and white background.",
        ]
        ollama_response = json.dumps({
            "critique": critique_points,
            "verdict": "needs_work",
        })

        with (
            patch.object(agent, "_get_snapshot_path", return_value="/tmp/snap.jpg"),
            patch("src.intelligence.presentation_critique_agent.Image") as mock_pil,
            patch.object(agent, "_encode_image", return_value="base64data"),
            patch.object(agent, "_call_ollama", return_value=ollama_response),
        ):
            mock_pil.open.return_value = _dummy_image()
            result = await agent.reason(_presentation_event(), _mock_db())

        assert isinstance(result, Action)
        assert result.action_type == "presentation_critique"
        assert result.agent_name == "presentation_critique"
        assert result.metadata["verdict"] == "needs_work"
        assert len(result.metadata["critique"]) == 2
        assert result.event_id == 10
        assert result.frame_id == 1

    @pytest.mark.asyncio
    async def test_clean_verdict_returns_none(self):
        agent = PresentationCritiqueAgent()
        ollama_response = json.dumps({"critique": [], "verdict": "clean"})

        with (
            patch.object(agent, "_get_snapshot_path", return_value="/tmp/snap.jpg"),
            patch("src.intelligence.presentation_critique_agent.Image") as mock_pil,
            patch.object(agent, "_encode_image", return_value="base64data"),
            patch.object(agent, "_call_ollama", return_value=ollama_response),
        ):
            mock_pil.open.return_value = _dummy_image()
            result = await agent.reason(_presentation_event(), _mock_db())

        assert result is None

    @pytest.mark.asyncio
    async def test_critique_truncated_to_three_points(self):
        agent = PresentationCritiqueAgent()
        ollama_response = json.dumps({
            "critique": ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"],
            "verdict": "needs_work",
        })

        with (
            patch.object(agent, "_get_snapshot_path", return_value="/tmp/snap.jpg"),
            patch("src.intelligence.presentation_critique_agent.Image") as mock_pil,
            patch.object(agent, "_encode_image", return_value="base64data"),
            patch.object(agent, "_call_ollama", return_value=ollama_response),
        ):
            mock_pil.open.return_value = _dummy_image()
            result = await agent.reason(_presentation_event(), _mock_db())

        assert isinstance(result, Action)
        assert len(result.metadata["critique"]) == 3

    @pytest.mark.asyncio
    async def test_action_description_joins_points(self):
        agent = PresentationCritiqueAgent()
        ollama_response = json.dumps({
            "critique": ["Bad font", "Low contrast"],
            "verdict": "needs_work",
        })

        with (
            patch.object(agent, "_get_snapshot_path", return_value="/tmp/snap.jpg"),
            patch("src.intelligence.presentation_critique_agent.Image") as mock_pil,
            patch.object(agent, "_encode_image", return_value="base64data"),
            patch.object(agent, "_call_ollama", return_value=ollama_response),
        ):
            mock_pil.open.return_value = _dummy_image()
            result = await agent.reason(_presentation_event(), _mock_db())

        assert result.action_description == "Bad font | Low contrast"


# ── Ollama failure ──

class TestOllamaFailure:
    @pytest.mark.asyncio
    async def test_ollama_error_returns_none(self):
        agent = PresentationCritiqueAgent()

        with (
            patch.object(agent, "_get_snapshot_path", return_value="/tmp/snap.jpg"),
            patch("src.intelligence.presentation_critique_agent.Image") as mock_pil,
            patch.object(agent, "_encode_image", return_value="base64data"),
            patch.object(agent, "_call_ollama", side_effect=Exception("Connection refused")),
        ):
            mock_pil.open.return_value = _dummy_image()
            result = await agent.reason(_presentation_event(), _mock_db())

        assert result is None

    @pytest.mark.asyncio
    async def test_malformed_json_returns_none(self):
        agent = PresentationCritiqueAgent()

        with (
            patch.object(agent, "_get_snapshot_path", return_value="/tmp/snap.jpg"),
            patch("src.intelligence.presentation_critique_agent.Image") as mock_pil,
            patch.object(agent, "_encode_image", return_value="base64data"),
            patch.object(agent, "_call_ollama", return_value="not valid json"),
        ):
            mock_pil.open.return_value = _dummy_image()
            result = await agent.reason(_presentation_event(), _mock_db())

        assert result is None

    @pytest.mark.asyncio
    async def test_markdown_fenced_json_is_parsed(self):
        agent = PresentationCritiqueAgent()
        inner = json.dumps({"critique": ["Inconsistent fonts"], "verdict": "needs_work"})
        fenced = f"```json\n{inner}\n```"

        with (
            patch.object(agent, "_get_snapshot_path", return_value="/tmp/snap.jpg"),
            patch("src.intelligence.presentation_critique_agent.Image") as mock_pil,
            patch.object(agent, "_encode_image", return_value="base64data"),
            patch.object(agent, "_call_ollama", return_value=fenced),
        ):
            mock_pil.open.return_value = _dummy_image()
            result = await agent.reason(_presentation_event(), _mock_db())

        assert isinstance(result, Action)
        assert result.action_description == "Inconsistent fonts"


# ── DB failure ──

class TestDbFailure:
    @pytest.mark.asyncio
    async def test_db_error_returns_none(self):
        agent = PresentationCritiqueAgent()
        db = AsyncMock()
        db.get_frame = AsyncMock(side_effect=Exception("DB error"))

        result = await agent.reason(_presentation_event(), db)
        assert result is None


# ── Agent properties ──

class TestAgentProperties:
    def test_name(self):
        agent = PresentationCritiqueAgent()
        assert agent.name == "presentation_critique"

    def test_custom_model(self):
        agent = PresentationCritiqueAgent(model="llava:7b")
        assert agent.name == "presentation_critique"


# ── parse_response edge cases ──

class TestParseResponse:
    def test_empty_critique_with_needs_work_returns_none(self):
        agent = PresentationCritiqueAgent()
        event = _presentation_event()
        result = agent._parse_response(
            json.dumps({"critique": [], "verdict": "needs_work"}),
            event,
        )
        assert result is None

    def test_missing_verdict_defaults_to_clean(self):
        agent = PresentationCritiqueAgent()
        event = _presentation_event()
        result = agent._parse_response(
            json.dumps({"critique": []}),
            event,
        )
        assert result is None
