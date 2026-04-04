"""Lightweight system health tests covering models, DB, frame comparison,
activity feed, intelligence layer, and API endpoints end-to-end."""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image

from src.capture.activity_feed import ActivityFeed
from src.capture.event_tap import ActivityKind, CaptureTrigger
from src.capture.frame_compare import FrameComparer
from src.intelligence.intelligence_layer import IntelligenceLayer
from src.intelligence.reasoning_agent import ReasoningAgent
from src.storage.database import DatabaseManager
from src.storage.models import Action, AdditionalContext, AppType, Event, Frame, OCRResult


# ── Models ──


class TestModels:
    def test_frame_defaults(self):
        f = Frame()
        assert f.id is None
        assert f.focused is True
        assert f.timestamp == ""

    def test_event_metadata_isolation(self):
        e1 = Event(metadata={"a": 1})
        e2 = Event()
        assert e1.metadata == {"a": 1}
        assert e2.metadata == {}
        e2.metadata["b"] = 2
        assert "b" not in e1.metadata

    def test_action_fields(self):
        a = Action(event_id=1, frame_id=2, agent_name="x", action_type="flag",
                   action_description="test")
        assert a.event_id == 1
        assert a.action_type == "flag"


# ── Database ──


@pytest.mark.asyncio
class TestDatabase:
    async def test_insert_and_retrieve_frame(self, db: DatabaseManager):
        fid = await db.insert_frame(Frame(
            timestamp="2025-01-01T00:00:00Z",
            app_name="Safari",
            window_name="Google",
            capture_trigger="click",
        ))
        assert fid >= 1
        frame = await db.get_frame(fid)
        assert frame is not None
        assert frame["app_name"] == "Safari"

    async def test_ocr_insert_and_fts(self, db: DatabaseManager):
        fid = await db.insert_frame(Frame(
            timestamp="2025-01-01T00:00:00Z", app_name="Code",
            capture_trigger="manual",
        ))
        await db.insert_ocr(fid, OCRResult(
            text="async def hello_world():", text_json="[]", confidence=0.99,
        ))
        results = await db.search("hello_world")
        assert len(results) == 1
        assert results[0]["frame_id"] == fid

    async def test_event_roundtrip(self, db: DatabaseManager):
        fid = await db.insert_frame(Frame(
            timestamp="2025-01-01T00:00:00Z", capture_trigger="manual",
        ))
        evt = Event(agent_name="classifier", app_type=AppType.BROWSER,
                    summary="User opened docs", metadata={"url": "https://docs.python.org"})
        eid = await db.insert_event(fid, evt)
        assert eid >= 1

        rows = await db.get_events_for_frame(fid)
        assert len(rows) == 1
        assert rows[0]["agent_name"] == "classifier"

        search = await db.search_events(query="docs")
        assert len(search) == 1

    async def test_context_roundtrip(self, db: DatabaseManager):
        fid = await db.insert_frame(Frame(
            timestamp="2025-01-01T00:00:00Z", capture_trigger="manual",
        ))
        ctx = AdditionalContext(
            source="clipboard", content_type="text",
            content="some copied text", metadata={},
        )
        cid = await db.insert_context(fid, ctx)
        assert cid >= 1

        rows = await db.get_context_for_frame(fid)
        assert len(rows) == 1

        search = await db.search_context(query="copied")
        assert len(search) == 1

    async def test_action_roundtrip(self, db: DatabaseManager):
        fid = await db.insert_frame(Frame(
            timestamp="2025-01-01T00:00:00Z", capture_trigger="manual",
        ))
        eid = await db.insert_event(fid, Event(
            agent_name="a", app_type=AppType.IDE, summary="coding session",
        ))
        act = Action(
            event_id=eid, frame_id=fid, agent_name="reasoner",
            action_type="summarize", action_description="Long coding session detected",
        )
        aid = await db.insert_action(act)
        assert aid >= 1

        rows = await db.get_actions_for_event(eid)
        assert len(rows) == 1

        search = await db.search_actions(query="coding session")
        assert len(search) == 1

    async def test_counts(self, db: DatabaseManager):
        counts = await db.get_counts()
        assert counts["frames"] == 0
        await db.insert_frame(Frame(
            timestamp="2025-01-01T00:00:00Z", capture_trigger="manual",
        ))
        counts = await db.get_counts()
        assert counts["frames"] == 1

    async def test_raw_sql_select_only(self, db: DatabaseManager):
        with pytest.raises(ValueError, match="SELECT"):
            await db.execute_raw_sql("DROP TABLE frames")

    async def test_get_latest_frame_id_empty(self, db: DatabaseManager):
        latest = await db.get_latest_frame_id()
        assert latest is None

    async def test_get_latest_frame_id(self, db: DatabaseManager):
        f1 = await db.insert_frame(Frame(timestamp="2025-01-01T00:00:00Z", capture_trigger="manual"))
        f2 = await db.insert_frame(Frame(timestamp="2025-01-01T00:01:00Z", capture_trigger="manual"))
        latest = await db.get_latest_frame_id()
        assert latest == f2


# ── Frame Comparer ──


class TestFrameComparer:
    def test_first_frame_returns_max(self):
        cmp = FrameComparer()
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        assert cmp.compare(img) == 1.0

    def test_identical_frames_return_zero(self):
        cmp = FrameComparer()
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        cmp.compare(img)
        assert cmp.compare(img) == 0.0

    def test_different_frames_return_nonzero(self):
        cmp = FrameComparer()
        black = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        white = Image.fromarray(np.full((100, 100, 3), 255, dtype=np.uint8))
        cmp.compare(black)
        dist = cmp.compare(white)
        assert dist > 0.5

    def test_reset(self):
        cmp = FrameComparer()
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        cmp.compare(img)
        cmp.reset()
        assert cmp.compare(img) == 1.0


# ── Activity Feed ──


class TestActivityFeed:
    def test_poll_returns_none_initially(self):
        feed = ActivityFeed()
        assert feed.poll() is None

    def test_typing_pause_detection(self):
        feed = ActivityFeed(typing_pause_delay_ms=50, idle_capture_interval_ms=999_999)
        feed.record(ActivityKind.KEYBOARD)
        time.sleep(0.01)
        feed.record(ActivityKind.KEYBOARD)
        assert feed.poll() is None  # still typing

        time.sleep(0.35)  # exceed 300ms is_typing threshold
        result = feed.poll()
        assert result == CaptureTrigger.TYPING_PAUSE

    def test_idle_trigger(self):
        feed = ActivityFeed(typing_pause_delay_ms=500, idle_capture_interval_ms=50)
        time.sleep(0.1)
        result = feed.poll()
        assert result == CaptureTrigger.IDLE

    def test_mark_captured_resets_idle(self):
        feed = ActivityFeed(typing_pause_delay_ms=500, idle_capture_interval_ms=100)
        feed.mark_captured()
        assert feed.poll() is None


# ── Intelligence Layer ──


class StubReasoningAgent(ReasoningAgent):
    def __init__(self, action: Action | None = None):
        self._action = action

    @property
    def name(self) -> str:
        return "stub"

    async def reason(self, event: Event, db: DatabaseManager) -> Action | None:
        return self._action


class FailingReasoningAgent(ReasoningAgent):
    @property
    def name(self) -> str:
        return "failing"

    async def reason(self, event: Event, db: DatabaseManager) -> Action | None:
        raise RuntimeError("intentional failure")


@pytest.mark.asyncio
class TestIntelligenceLayer:
    async def test_submit_and_process(self, db: DatabaseManager):
        fid = await db.insert_frame(Frame(
            timestamp="2025-01-01T00:00:00Z", capture_trigger="manual",
        ))
        eid = await db.insert_event(fid, Event(
            agent_name="a", app_type=AppType.IDE, summary="test",
        ))
        action = Action(
            event_id=eid, frame_id=fid, agent_name="stub",
            action_type="log", action_description="logged something",
        )
        layer = IntelligenceLayer(agents=[StubReasoningAgent(action)], db=db)
        task = asyncio.create_task(layer.run())

        evt = Event(id=eid, frame_id=fid, agent_name="a",
                    app_type=AppType.IDE, summary="test")
        await layer.submit(evt)
        await asyncio.sleep(0.3)
        await layer.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        actions = await db.get_actions_for_event(eid)
        assert len(actions) == 1
        assert actions[0]["action_type"] == "log"

    async def test_failing_agent_does_not_crash(self, db: DatabaseManager):
        fid = await db.insert_frame(Frame(
            timestamp="2025-01-01T00:00:00Z", capture_trigger="manual",
        ))
        eid = await db.insert_event(fid, Event(
            agent_name="a", app_type=AppType.IDE, summary="test",
        ))
        layer = IntelligenceLayer(agents=[FailingReasoningAgent()], db=db)
        task = asyncio.create_task(layer.run())

        await layer.submit(Event(id=eid, frame_id=fid, agent_name="a",
                                 app_type=AppType.IDE, summary="test"))
        await asyncio.sleep(0.3)
        await layer.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # no crash = success


# ── API Endpoints ──


@pytest.mark.asyncio
class TestAPI:
    async def _seed(self, db: DatabaseManager) -> tuple[int, int, int]:
        fid = await db.insert_frame(Frame(
            timestamp="2025-06-15T12:00:00Z", app_name="Chrome",
            window_name="GitHub", capture_trigger="click",
        ))
        await db.insert_ocr(fid, OCRResult(
            text="pull request review comments",
            
            text_json="[]", confidence=0.95,
        ))
        eid = await db.insert_event(fid, Event(
            agent_name="classifier", app_type=AppType.BROWSER,
            summary="Code review session", metadata={"repo": "zexp"},
        ))
        await db.insert_context(fid, AdditionalContext(
            source="chrome_devtools", content_type="dom",
            content="<html>...</html>", metadata={"url": "https://github.com"},
        ))
        aid = await db.insert_action(Action(
            event_id=eid, frame_id=fid, agent_name="notifier",
            action_type="flag", action_description="Important PR detected",
        ))
        return fid, eid, aid

    async def test_health(self, app, db):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/health")
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "ok"
            assert "counts" in body

    async def test_search(self, app, db):
        await self._seed(db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/search", params={"q": "pull request"})
            assert r.status_code == 200
            results = r.json()
            assert len(results) >= 1
            assert "pull request" in results[0]["text"]

    async def test_frame_metadata(self, app, db):
        fid, _, _ = await self._seed(db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get(f"/frames/{fid}/metadata")
            assert r.status_code == 200
            body = r.json()
            assert body["frame_id"] == fid
            assert body["app_name"] == "Chrome"
            assert body["ocr"] is not None
            assert len(body["events"]) == 1
            assert len(body["context"]) == 1
            assert len(body["actions"]) == 1

    async def test_frame_not_found(self, app, db):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/frames/99999/metadata")
            assert r.status_code == 404

    async def test_events_search(self, app, db):
        await self._seed(db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/events", params={"q": "review"})
            assert r.status_code == 200
            assert len(r.json()) >= 1

    async def test_events_by_frame(self, app, db):
        fid, _, _ = await self._seed(db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get(f"/events/{fid}")
            assert r.status_code == 200
            assert len(r.json()) == 1

    async def test_context_search(self, app, db):
        await self._seed(db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/context", params={"source": "chrome_devtools"})
            assert r.status_code == 200
            assert len(r.json()) >= 1

    async def test_context_push(self, app, db):
        fid, _, _ = await self._seed(db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/context", json={
                "source": "git",
                "content_type": "diff",
                "content": "--- a/file.py\n+++ b/file.py",
                "frame_id": fid,
            })
            assert r.status_code == 200
            body = r.json()
            assert body["frame_id"] == fid
            assert body["context_id"] >= 1

    async def test_actions_search(self, app, db):
        await self._seed(db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/actions", params={"action_type": "flag"})
            assert r.status_code == 200
            assert len(r.json()) >= 1

    async def test_actions_by_event(self, app, db):
        _, eid, _ = await self._seed(db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get(f"/actions/{eid}")
            assert r.status_code == 200
            assert len(r.json()) == 1

    async def test_raw_sql(self, app, db):
        await self._seed(db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/raw_sql", json={
                "sql": "SELECT COUNT(*) as n FROM frames",
            })
            assert r.status_code == 200
            assert r.json()[0]["n"] >= 1

    async def test_raw_sql_rejects_writes(self, app, db):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/raw_sql", json={"sql": "DELETE FROM frames"})
            assert r.status_code == 400
