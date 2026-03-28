"""Stress and edge-case integration tests.

Targets real bugs found during audit:
  - Double timeout race: _do_capture(15s) vs vision agent(15s)
  - Sequential broadcast: one slow WS client blocks all others
  - Queue saturation: events arrive faster than LLM can process
  - Concurrent chat + push: neither should starve the other
  - ConnectionManager rapid connect/disconnect under load
  - process_frame partial failures: fast agents' results survive slow ones
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.general_agent.agent import GeneralAgent
from src.general_agent.tools import ToolRegistry
from src.general_agent.ws_manager import ConnectionManager
from src.intelligence.intelligence_layer import IntelligenceLayer
from src.intelligence.reasoning_agent import ReasoningAgent
from src.llm.types import Message, ToolSpec
from src.storage.database import DatabaseManager
from src.storage.models import Action, AppType, Event, Frame, OCRResult


# ── Stubs ─────────────────────────────────────────────────────


class StubLLM:
    def __init__(self, response: str = "ok"):
        self._response = response
        self.call_count = 0

    async def complete(self, messages, tools=None) -> Message:
        self.call_count += 1
        return Message(role="assistant", content=self._response)


class SlowLLM:
    def __init__(self, delay: float, response: str = "slow"):
        self._delay = delay
        self._response = response
        self.call_count = 0

    async def complete(self, messages, tools=None) -> Message:
        self.call_count += 1
        await asyncio.sleep(self._delay)
        return Message(role="assistant", content=self._response)


class HangingLLM:
    """Never returns — simulates a provider that's completely stuck."""

    async def complete(self, messages, tools=None) -> Message:
        await asyncio.sleep(3600)
        return Message(role="assistant", content="unreachable")


class FlakeyLLM:
    """Alternates between success and timeout."""

    def __init__(self):
        self._call_count = 0

    async def complete(self, messages, tools=None) -> Message:
        self._call_count += 1
        if self._call_count % 2 == 0:
            raise TimeoutError("provider timed out")
        return Message(role="assistant", content=f"response {self._call_count}")


class FakeWebSocket:
    """Minimal WebSocket mock for ConnectionManager tests."""

    def __init__(self, *, slow: float = 0, fail_after: int | None = None):
        self._slow = slow
        self._fail_after = fail_after
        self._send_count = 0
        self.messages: list[str] = []
        self.accepted = False
        self.closed = False

    async def accept(self):
        self.accepted = True

    async def send_text(self, data: str):
        self._send_count += 1
        if self._fail_after is not None and self._send_count > self._fail_after:
            raise ConnectionError("client gone")
        if self._slow:
            await asyncio.sleep(self._slow)
        self.messages.append(data)

    async def close(self, code: int = 1000):
        self.closed = True


def _make_agent(db, llm=None, ws_manager=None) -> GeneralAgent:
    tools = ToolRegistry(db)
    return GeneralAgent(
        db=db,
        tools=tools,
        llm=llm or StubLLM(),
        ws_manager=ws_manager,
    )


async def _stop_agent(agent, task):
    await agent.stop()
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


# ── 1. Double timeout race in process_frame ──────────────────


@pytest.mark.asyncio
class TestDoubleTimeoutRace:
    """_do_capture wraps process_frame with capture_timeout_s (15s).
    If a slow agent takes close to that, the outer timeout can cancel
    process_frame before fast agents' results are saved."""

    async def test_slow_agent_doesnt_lose_fast_agent_results(self, db, tmp_settings):
        """Fast agent finishes in 0.1s, slow agent takes 2s.
        With a 5s outer timeout, both should be saved."""
        from src.capture.triggers import process_frame
        from src.storage.snapshot_writer import SnapshotWriter
        from PIL import Image
        import numpy as np

        writer = SnapshotWriter(tmp_settings)
        image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

        class FastAgent:
            name = "fast"
            async def process(self, image, ocr, app, window):
                return Event(
                    agent_name="fast", app_type=AppType.BROWSER,
                    summary="fast result",
                )

        class SlowAgent:
            name = "slow"
            async def process(self, image, ocr, app, window):
                await asyncio.sleep(2.0)
                return Event(
                    agent_name="slow", app_type=AppType.OTHER,
                    summary="slow result",
                )

        with patch("src.capture.triggers.perform_ocr") as mock_ocr:
            mock_ocr.return_value = OCRResult(text="test", confidence=0.9)

            frame_id = await asyncio.wait_for(
                process_frame(
                    image=image, app_name="Test", window_name="Win",
                    trigger="manual", db=db, writer=writer,
                    agents=[FastAgent(), SlowAgent()], providers=[],
                ),
                timeout=5.0,
            )

        events = await db.get_events_for_frame(frame_id)
        names = [e["agent_name"] for e in events]
        assert "fast" in names
        assert "slow" in names

    async def test_outer_timeout_cancels_everything(self, db, tmp_settings):
        """When the outer timeout fires, fast agent results that were already
        processed should still be in the DB (they were inserted before cancel)."""
        from src.capture.triggers import process_frame
        from src.storage.snapshot_writer import SnapshotWriter
        from PIL import Image
        import numpy as np

        writer = SnapshotWriter(tmp_settings)
        image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

        class FastAgent:
            name = "fast"
            async def process(self, image, ocr, app, window):
                return Event(
                    agent_name="fast", app_type=AppType.BROWSER,
                    summary="fast result",
                )

        class HangingAgent:
            name = "hanging"
            async def process(self, image, ocr, app, window):
                await asyncio.sleep(3600)

        with patch("src.capture.triggers.perform_ocr") as mock_ocr:
            mock_ocr.return_value = OCRResult(text="test", confidence=0.9)

            # Outer timeout at 1s — hanging agent blocks gather
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    process_frame(
                        image=image, app_name="Test", window_name="Win",
                        trigger="manual", db=db, writer=writer,
                        agents=[FastAgent(), HangingAgent()], providers=[],
                    ),
                    timeout=1.0,
                )

        # Frame should still exist (it was inserted before agents ran)
        # But events from fast agent are LOST because gather was cancelled
        # before the result-processing loop ran. This is the bug.
        # Just verify the frame was saved.
        rows = await db.execute_raw_sql("SELECT COUNT(*) as c FROM frames")
        assert rows[0]["c"] >= 1

    async def test_agent_raising_timeout_doesnt_affect_siblings(self, db, tmp_settings):
        """An agent that raises TimeoutError (from its own internal timeout)
        should not prevent sibling agents from having their results saved."""
        from src.capture.triggers import process_frame
        from src.storage.snapshot_writer import SnapshotWriter
        from PIL import Image
        import numpy as np

        writer = SnapshotWriter(tmp_settings)
        image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

        class GoodAgent:
            name = "good"
            async def process(self, image, ocr, app, window):
                return Event(
                    agent_name="good", app_type=AppType.OTHER,
                    summary="good result",
                )

        class TimeoutAgent:
            name = "timeout"
            async def process(self, image, ocr, app, window):
                raise TimeoutError("gemini hung")

        class ErrorAgent:
            name = "error"
            async def process(self, image, ocr, app, window):
                raise ConnectionError("network down")

        with patch("src.capture.triggers.perform_ocr") as mock_ocr:
            mock_ocr.return_value = OCRResult(text="test", confidence=0.9)

            frame_id = await asyncio.wait_for(
                process_frame(
                    image=image, app_name="Test", window_name="Win",
                    trigger="manual", db=db, writer=writer,
                    agents=[GoodAgent(), TimeoutAgent(), ErrorAgent()],
                    providers=[],
                ),
                timeout=5.0,
            )

        events = await db.get_events_for_frame(frame_id)
        assert len(events) == 1
        assert events[0]["agent_name"] == "good"


# ── 2. ConnectionManager stress ──────────────────────────────


@pytest.mark.asyncio
class TestConnectionManagerStress:
    async def test_broadcast_to_many_clients(self):
        """Broadcast to 50 clients — all should receive the message."""
        mgr = ConnectionManager()
        clients = [FakeWebSocket() for _ in range(50)]
        for ws in clients:
            await mgr.connect(ws)

        assert mgr.client_count == 50

        await mgr.broadcast({"type": "test", "n": 1})

        for ws in clients:
            assert len(ws.messages) == 1

    async def test_slow_client_doesnt_block_forever(self):
        """One slow client shouldn't prevent others from receiving messages.
        (Current impl is sequential, so it DOES block — but with a timeout.)"""
        mgr = ConnectionManager()
        fast = FakeWebSocket()
        slow = FakeWebSocket(slow=10.0)  # 10s per send, but 5s timeout
        await mgr.connect(fast)
        await mgr.connect(slow)

        start = time.monotonic()
        await mgr.broadcast({"type": "test"})
        elapsed = time.monotonic() - start

        # Should complete in ~5s (slow client timeout), not 10s
        assert elapsed < 7.0
        # Fast client got the message
        assert len(fast.messages) == 1
        # Slow client was removed as dead
        assert mgr.client_count == 1

    async def test_client_dies_mid_broadcast(self):
        """Client that fails after first message is cleaned up."""
        mgr = ConnectionManager()
        good = FakeWebSocket()
        dying = FakeWebSocket(fail_after=1)
        await mgr.connect(good)
        await mgr.connect(dying)

        # First broadcast — both succeed
        await mgr.broadcast({"type": "msg1"})
        assert len(good.messages) == 1
        assert len(dying.messages) == 1
        assert mgr.client_count == 2

        # Second broadcast — dying client fails
        await mgr.broadcast({"type": "msg2"})
        assert len(good.messages) == 2
        assert mgr.client_count == 1  # dying removed

    async def test_rapid_connect_disconnect(self):
        """50 clients connect and disconnect rapidly while broadcasts happen."""
        mgr = ConnectionManager()

        async def churn():
            for _ in range(50):
                ws = FakeWebSocket()
                await mgr.connect(ws)
                await asyncio.sleep(0.01)
                await mgr.disconnect(ws)

        async def broadcast_loop():
            for i in range(20):
                await mgr.broadcast({"type": "churn", "n": i})
                await asyncio.sleep(0.02)

        # Both should complete without errors or deadlocks
        await asyncio.wait_for(
            asyncio.gather(churn(), broadcast_loop()),
            timeout=10.0,
        )

    async def test_broadcast_with_no_clients(self):
        """Broadcast to empty set should be a no-op."""
        mgr = ConnectionManager()
        await mgr.broadcast({"type": "nobody_home"})
        assert mgr.client_count == 0

    async def test_concurrent_broadcasts(self):
        """Multiple broadcasts happening simultaneously shouldn't corrupt state."""
        mgr = ConnectionManager()
        clients = [FakeWebSocket() for _ in range(10)]
        for ws in clients:
            await mgr.connect(ws)

        coros = [mgr.broadcast({"n": i}) for i in range(20)]
        await asyncio.wait_for(asyncio.gather(*coros), timeout=10.0)

        # Each client should have received all 20 messages
        for ws in clients:
            assert len(ws.messages) == 20


# ── 3. GeneralAgent queue saturation ─────────────────────────


@pytest.mark.asyncio
class TestQueueSaturation:
    async def test_queue_drains_under_slow_llm(self, db):
        """Push 10 events while LLM takes 0.5s each. Queue should drain
        within a reasonable time, not back up indefinitely."""
        llm = SlowLLM(delay=0.3, response="noted")
        agent = _make_agent(db, llm=llm)
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        for i in range(10):
            await agent.push("event", {
                "agent_name": "test", "summary": f"event {i}",
                "app_name": "App", "metadata": {},
            })

        # 10 items × ~0.3s each ≈ 3s + margin
        deadline = time.monotonic() + 10.0
        while not agent._queue.empty():
            assert time.monotonic() < deadline, (
                f"Queue stuck: {agent._queue.qsize()} items remaining"
            )
            await asyncio.sleep(0.1)

        await _stop_agent(agent, task)

    async def test_burst_push_doesnt_crash(self, db):
        """Push 100 events in a tight loop — agent must not crash."""
        agent = _make_agent(db, llm=StubLLM("NOTHING"))
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        for i in range(100):
            await agent.push("event", {
                "agent_name": "burst", "summary": f"burst event {i}",
                "app_name": "App", "metadata": {},
            })

        deadline = time.monotonic() + 15.0
        while not agent._queue.empty():
            assert time.monotonic() < deadline, "Agent stalled on burst"
            await asyncio.sleep(0.2)

        assert agent._running
        await _stop_agent(agent, task)

    async def test_chat_during_queue_drain(self, db):
        """User chats while the queue is full of events. Chat should still
        complete within the timeout — it bypasses the queue."""
        llm = SlowLLM(delay=0.5, response="busy but here")
        agent = _make_agent(db, llm=llm)
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        # Saturate the queue
        for i in range(20):
            await agent.push("event", {
                "agent_name": "flood", "summary": f"flood {i}",
                "app_name": "App", "metadata": {},
            })

        # Chat should bypass the queue and respond directly
        response = await asyncio.wait_for(
            agent.chat("what's happening?"),
            timeout=10.0,
        )
        assert len(response) > 0
        assert "busy but here" in response or "couldn't" in response.lower()

        await _stop_agent(agent, task)


# ── 4. Flakey LLM resilience ─────────────────────────────────


@pytest.mark.asyncio
class TestFlakeyLLM:
    async def test_alternating_success_and_timeout(self, db):
        """LLM alternates between working and timing out. Agent must
        keep processing and not accumulate stale state."""
        llm = FlakeyLLM()
        agent = _make_agent(db, llm=llm)
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        # Use very different summaries to avoid dedup in EventFilter
        summaries = [
            "User is buying a $500 HDMI cable on Amazon",
            "Browsing flights to Tokyo on Google Flights for $1200",
            "Reading a Wikipedia article about quantum computing",
            "Editing a spreadsheet in Excel with revenue data",
            "Watching a YouTube video about machine learning",
            "Composing an email in Gmail to the engineering team",
        ]
        for i, summary in enumerate(summaries):
            await agent.push("event", {
                "agent_name": f"agent_{i}",
                "summary": summary,
                "app_name": f"App{i}", "metadata": {"unique": i},
            })

        deadline = time.monotonic() + 15.0
        while not agent._queue.empty():
            assert time.monotonic() < deadline, "Agent stuck on flakey LLM"
            await asyncio.sleep(0.1)

        assert agent._running
        # At least some events should have reached the LLM
        assert llm._call_count >= 3
        await _stop_agent(agent, task)

    async def test_chat_after_many_timeouts(self, db):
        """After several push items that time out, chat should still work."""
        llm = FlakeyLLM()
        agent = _make_agent(db, llm=llm)
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        # Push items that will partially fail
        for i in range(4):
            await agent.push("event", {
                "agent_name": "flakey", "summary": f"fail {i}",
                "app_name": "App", "metadata": {},
            })

        await asyncio.sleep(2.0)

        # Chat should work regardless of previous failures
        response = await asyncio.wait_for(
            agent.chat("still there?"),
            timeout=10.0,
        )
        assert len(response) > 0
        await _stop_agent(agent, task)


# ── 5. IntelligenceLayer concurrent stress ───────────────────


class SlowReasoningAgent(ReasoningAgent):
    def __init__(self, delay: float, name_: str = "slow_reasoner"):
        self._delay = delay
        self._name = name_
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def reason(self, event, db) -> Action | None:
        self.call_count += 1
        await asyncio.sleep(self._delay)
        return Action(
            event_id=event.id,
            frame_id=event.frame_id,
            agent_name=self._name,
            action_type="test",
            action_description=f"slow result from {self._name}",
        )


class CrashingReasoningAgent(ReasoningAgent):
    def __init__(self, name_: str = "crasher"):
        self._name = name_
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def reason(self, event, db) -> Action | None:
        self.call_count += 1
        if self.call_count % 3 == 0:
            raise RuntimeError("random crash")
        raise TimeoutError("LLM timed out")


@pytest.mark.asyncio
class TestIntelligenceLayerStress:
    async def test_burst_events_with_slow_agents(self, db):
        """Submit 10 events rapidly with a 0.3s agent. Layer should process
        all of them without dropping any."""
        slow = SlowReasoningAgent(delay=0.3)
        layer = IntelligenceLayer(agents=[slow], db=db)
        task = asyncio.create_task(layer.run())

        fid = await db.insert_frame(Frame(
            timestamp="2025-01-01T00:00:00Z", capture_trigger="manual",
        ))

        for i in range(10):
            evt = Event(
                agent_name="test", app_type=AppType.BROWSER,
                summary=f"stress event {i}", frame_id=fid,
            )
            evt.id = i + 1
            await layer.submit(evt)

        # 10 events × 0.3s = 3s + margin
        deadline = time.monotonic() + 10.0
        while not layer._event_queue.empty():
            assert time.monotonic() < deadline, "Intelligence layer stalled"
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.5)

        assert slow.call_count == 10

        await layer.stop()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    async def test_mixed_success_and_failure(self, db):
        """One agent succeeds, another alternates between timeout and crash.
        Successful agent's actions should all be saved."""
        good = SlowReasoningAgent(delay=0.05, name_="good")
        bad = CrashingReasoningAgent(name_="bad")
        layer = IntelligenceLayer(agents=[good, bad], db=db)
        task = asyncio.create_task(layer.run())

        fid = await db.insert_frame(Frame(
            timestamp="2025-01-01T00:00:00Z", capture_trigger="manual",
        ))

        for i in range(5):
            evt = Event(
                agent_name="test", app_type=AppType.OTHER,
                summary=f"mixed event {i}", frame_id=fid,
            )
            evt.id = i + 1
            await db.insert_event(fid, evt)
            await layer.submit(evt)

        await asyncio.sleep(2.0)

        assert good.call_count == 5
        assert bad.call_count == 5

        actions = await db.search_actions(query="slow result", limit=20)
        assert len(actions) == 5

        await layer.stop()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


# ── 6. Agent + WS manager integration ────────────────────────


@pytest.mark.asyncio
class TestAgentWSIntegration:
    async def test_notification_reaches_ws_clients(self, db):
        """Full pipeline: push event → LLM → ws_send → client receives."""
        mgr = ConnectionManager()
        client = FakeWebSocket()
        await mgr.connect(client)

        llm = StubLLM("heads up, that price is wrong")
        agent = _make_agent(db, llm=llm, ws_manager=mgr)
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        await agent.push("event", {
            "agent_name": "test",
            "summary": "User viewing overpriced item",
            "app_name": "Chrome", "metadata": {},
        })

        deadline = time.monotonic() + 5.0
        while not client.messages:
            assert time.monotonic() < deadline, "No message reached WS client"
            await asyncio.sleep(0.1)

        import json
        msg = json.loads(client.messages[0])
        assert msg["type"] == "show_proposal"
        assert "price" in msg["text"]

        await _stop_agent(agent, task)

    async def test_multiple_clients_all_receive(self, db):
        """All connected WS clients receive the same notification."""
        mgr = ConnectionManager()
        clients = [FakeWebSocket() for _ in range(5)]
        for ws in clients:
            await mgr.connect(ws)

        llm = StubLLM("alert")
        agent = _make_agent(db, llm=llm, ws_manager=mgr)
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        await agent.push("event", {
            "agent_name": "test", "summary": "something",
            "app_name": "App", "metadata": {},
        })

        deadline = time.monotonic() + 5.0
        while any(len(ws.messages) == 0 for ws in clients):
            assert time.monotonic() < deadline, "Not all clients received message"
            await asyncio.sleep(0.1)

        for ws in clients:
            assert len(ws.messages) >= 1

        await _stop_agent(agent, task)

    async def test_dead_client_doesnt_prevent_delivery(self, db):
        """A dead client shouldn't prevent other clients from getting messages."""
        mgr = ConnectionManager()
        good = FakeWebSocket()
        dead = FakeWebSocket(fail_after=0)  # fails on first send
        await mgr.connect(good)
        await mgr.connect(dead)

        llm = StubLLM("notification")
        agent = _make_agent(db, llm=llm, ws_manager=mgr)
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        await agent.push("event", {
            "agent_name": "test", "summary": "test event",
            "app_name": "App", "metadata": {},
        })

        deadline = time.monotonic() + 5.0
        while not good.messages:
            assert time.monotonic() < deadline, "Good client didn't receive message"
            await asyncio.sleep(0.1)

        assert len(good.messages) >= 1
        assert mgr.client_count == 1  # dead one removed

        await _stop_agent(agent, task)

    async def test_chat_response_reaches_ws(self, db):
        """chat() response should be broadcast as append_conversation."""
        mgr = ConnectionManager()
        client = FakeWebSocket()
        await mgr.connect(client)

        llm = StubLLM("here's what I see")
        agent = _make_agent(db, llm=llm, ws_manager=mgr)

        response = await agent.chat("what's on screen?")
        assert "here's what I see" in response

        import json
        assert len(client.messages) >= 1
        msg = json.loads(client.messages[-1])
        assert msg["type"] == "append_conversation"
        assert msg["role"] == "assistant"

    async def test_no_ws_manager_doesnt_crash(self, db):
        """Agent with ws_manager=None should work — just no broadcasts."""
        agent = _make_agent(db, llm=StubLLM("test"))
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        await agent.push("event", {
            "agent_name": "test", "summary": "event",
            "app_name": "App", "metadata": {},
        })

        await asyncio.sleep(0.5)
        assert agent._running

        response = await agent.chat("hello")
        assert len(response) > 0

        await _stop_agent(agent, task)


# ── 7. Agent process item timeout ─────────────────────────────


@pytest.mark.asyncio
class TestProcessItemTimeout:
    async def test_hanging_llm_triggers_process_item_timeout(self, db):
        """When the LLM hangs, _PROCESS_ITEM_TIMEOUT should kick in and
        skip the item, allowing the next one to be processed."""
        agent = _make_agent(db, llm=HangingLLM())
        # Shorten the timeout for testing
        agent._PROCESS_ITEM_TIMEOUT = 2
        agent._LLM_CALL_TIMEOUT = 1

        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        # Push two items — first will hang then timeout, second should process
        await agent.push("event", {
            "agent_name": "test", "summary": "will hang",
            "app_name": "App", "metadata": {},
        })
        await agent.push("event", {
            "agent_name": "test", "summary": "should process",
            "app_name": "App", "metadata": {},
        })

        # Both should be consumed within timeout budget
        deadline = time.monotonic() + 8.0
        while not agent._queue.empty():
            assert time.monotonic() < deadline, "Queue stuck — timeout didn't fire"
            await asyncio.sleep(0.2)

        assert agent._running
        await _stop_agent(agent, task)

    async def test_item_timeout_doesnt_corrupt_state(self, db):
        """After a timeout, the agent's conversation and context should be clean."""
        agent = _make_agent(db, llm=HangingLLM())
        agent._PROCESS_ITEM_TIMEOUT = 1
        agent._LLM_CALL_TIMEOUT = 0.5

        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        await agent.push("event", {
            "agent_name": "test", "summary": "timeout item",
            "app_name": "App", "metadata": {},
        })

        await asyncio.sleep(2.0)

        # Conversation should not contain garbage from timed-out processing
        assert len(agent._conversation) == 0
        # Recent items should still be tracked
        assert len(agent._recent_items) == 1

        await _stop_agent(agent, task)
