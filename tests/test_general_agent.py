"""Integration tests for GeneralAgent — run-loop resilience, LLM timeout
behavior, concurrent pressure, full push → filter → LLM → broadcast
pipeline, and overlay message handling via ConnectionManager.

Uses the actual ConnectionManager + ASGI WebSocket test client to test
the real broadcast path (no raw websockets needed).
"""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.general_agent.agent import GeneralAgent, PushItem
from src.general_agent.tools import ToolRegistry
from src.general_agent.ws_manager import ConnectionManager
from src.llm.types import Message, ToolCall, ToolSpec
from src.storage.database import DatabaseManager
from src.storage.models import Event, AppType, Frame


# ── Stubs ─────────────────────────────────────────────────────


class StubLLM:
    """Returns a canned response. Records calls for assertion."""

    def __init__(self, response: str = "test response"):
        self._response = response
        self.calls: list[list[Message]] = []

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Message:
        self.calls.append(messages)
        return Message(role="assistant", content=self._response)


class SlowLLM:
    """Takes `delay` seconds per completion — simulates a slow provider."""

    def __init__(self, delay: float, response: str = "slow answer"):
        self._delay = delay
        self._response = response

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Message:
        await asyncio.sleep(self._delay)
        return Message(role="assistant", content=self._response)


class HangingLLM:
    """Hangs forever — simulates a provider that accepts the connection
    but never sends a response (TCP half-open, load-balancer timeout, etc.)."""

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Message:
        await asyncio.Event().wait()  # blocks forever
        return Message(role="assistant", content="")


class ToolCallingLLM:
    """First call returns a tool_call, second call returns text."""

    def __init__(self):
        self._call_count = 0

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Message:
        self._call_count += 1
        if self._call_count == 1:
            return Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="tc1", name="db_query", arguments={
                    "table": "events", "query": "test", "limit": "3",
                })],
            )
        return Message(role="assistant", content="found 3 events")


class ExplodingLLM:
    """Raises on every call."""

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Message:
        raise ConnectionError("LLM provider unreachable")


class SpyConnectionManager(ConnectionManager):
    """ConnectionManager that records all broadcast messages for assertion."""

    def __init__(self):
        super().__init__()
        self.broadcast_log: list[dict] = []

    async def broadcast(self, message: dict) -> None:
        self.broadcast_log.append(message)
        await super().broadcast(message)


# ── Helpers ───────────────────────────────────────────────────


def _make_agent(db, llm=None, ws_manager=None) -> GeneralAgent:
    tools = ToolRegistry(db)
    return GeneralAgent(
        db=db,
        tools=tools,
        llm=llm or StubLLM(),
        ws_manager=ws_manager or SpyConnectionManager(),
    )


async def _run_agent_briefly(agent, duration=0.3):
    """Start agent, let it run, then return the task."""
    task = asyncio.create_task(agent.run())
    await asyncio.sleep(duration)
    return task


async def _stop_agent(agent, task):
    await agent.stop()
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


async def _wait_queue_empty(agent, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not agent._queue.empty():
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Queue not empty after {timeout}s "
                f"(depth={agent._queue.qsize()})"
            )
        await asyncio.sleep(0.05)


def _proposals(ws: SpyConnectionManager) -> list[dict]:
    """Extract show_proposal messages from the broadcast log."""
    return [m for m in ws.broadcast_log if m.get("type") == "show_proposal"]


def _conversations(ws: SpyConnectionManager) -> list[dict]:
    """Extract append_conversation messages from the broadcast log."""
    return [m for m in ws.broadcast_log if m.get("type") == "append_conversation"]


# ── 1. Full pipeline: push → filter → LLM → broadcast ────────


@pytest.mark.asyncio
class TestFullPipeline:
    async def test_event_flows_to_broadcast(self, db):
        """An event pushed into the agent goes through the filter, hits the
        LLM, and the notification is broadcast via ConnectionManager."""
        ws = SpyConnectionManager()
        llm = StubLLM("that price seems high")
        agent = _make_agent(db, llm=llm, ws_manager=ws)
        task = await _run_agent_briefly(agent, 0.2)

        await agent.push("event", {
            "agent_name": "gemini_vision",
            "app_type": "browser",
            "app_name": "Chrome",
            "summary": "User viewing a $500 HDMI cable on Amazon",
            "metadata": {"url": "https://amazon.com/dp/123"},
        })

        await _wait_queue_empty(agent, timeout=5.0)
        await asyncio.sleep(0.1)

        found = _proposals(ws)
        assert len(found) == 1
        assert "that price seems high" in found[0]["text"]

        await _stop_agent(agent, task)

    async def test_duplicate_events_are_filtered(self, db):
        """Pushing the same event twice — only one should be broadcast
        (fuzzy dedup in EventFilter)."""
        ws = SpyConnectionManager()
        llm = StubLLM("interesting")
        agent = _make_agent(db, llm=llm, ws_manager=ws)
        task = await _run_agent_briefly(agent, 0.2)

        for _ in range(2):
            await agent.push("event", {
                "agent_name": "gemini_vision",
                "app_name": "Safari",
                "summary": "User reading the same article about Python asyncio",
                "metadata": {},
            })

        await _wait_queue_empty(agent, timeout=5.0)
        await asyncio.sleep(0.2)

        assert len(_proposals(ws)) <= 1

        await _stop_agent(agent, task)

    async def test_nothing_response_suppressed(self, db):
        """When the LLM says NOTHING, no notification is broadcast."""
        ws = SpyConnectionManager()
        llm = StubLLM("NOTHING")
        agent = _make_agent(db, llm=llm, ws_manager=ws)
        task = await _run_agent_briefly(agent, 0.2)

        await agent.push("event", {
            "agent_name": "gemini_vision",
            "app_name": "Finder",
            "summary": "User browsing files",
            "metadata": {},
        })

        await _wait_queue_empty(agent, timeout=5.0)
        await asyncio.sleep(0.2)

        assert len(_proposals(ws)) == 0

        await _stop_agent(agent, task)

    async def test_chat_broadcasts_response(self, db):
        """chat() returns the LLM response and broadcasts it."""
        ws = SpyConnectionManager()
        llm = StubLLM("hey there")
        agent = _make_agent(db, llm=llm, ws_manager=ws)

        response = await agent.chat("hello")
        assert response == "hey there"

        convos = _conversations(ws)
        assert len(convos) == 1
        assert convos[0]["text"] == "hey there"
        assert convos[0]["role"] == "assistant"

    async def test_chat_records_conversation_history(self, db):
        """chat() appends both user and assistant turns."""
        ws = SpyConnectionManager()
        llm = StubLLM("answer")
        agent = _make_agent(db, llm=llm, ws_manager=ws)

        await agent.chat("question")

        assert len(agent._conversation) == 2
        assert agent._conversation[0].role == "user"
        assert agent._conversation[0].text == "question"
        assert agent._conversation[1].role == "assistant"
        assert agent._conversation[1].text == "answer"


# ── 2. LLM failures & timeouts ───────────────────────────────


@pytest.mark.asyncio
class TestLLMFailures:
    async def test_llm_crash_doesnt_kill_run_loop(self, db):
        """If the LLM throws on every call, the run loop should keep
        consuming items, not crash or stall."""
        agent = _make_agent(db, llm=ExplodingLLM())
        task = await _run_agent_briefly(agent, 0.2)

        for i in range(3):
            await agent.push("event", {
                "agent_name": f"crash{i}",
                "summary": f"event {i} that will fail",
                "app_name": "X", "metadata": {},
            })

        await _wait_queue_empty(agent, timeout=5.0)
        assert agent._running
        await _stop_agent(agent, task)

    async def test_chat_with_llm_failure_returns_fallback(self, db):
        """chat() must return a fallback string, not raise."""
        agent = _make_agent(db, llm=ExplodingLLM())
        response = await agent.chat("hello")
        assert "couldn't generate" in response.lower()

    async def test_hanging_llm_is_killed_by_process_item_timeout(self, db):
        """An LLM that hangs forever should be killed by _PROCESS_ITEM_TIMEOUT.
        The run loop must continue processing the next item."""
        agent = _make_agent(db, llm=HangingLLM())
        agent._PROCESS_ITEM_TIMEOUT = 2
        agent._LLM_CALL_TIMEOUT = 1

        task = await _run_agent_briefly(agent, 0.2)

        # Push 2 items: the first will hang and get killed, the second
        # should still be consumed
        await agent.push("event", {
            "agent_name": "hang1", "summary": "hanging event one",
            "app_name": "X", "metadata": {},
        })
        await agent.push("event", {
            "agent_name": "hang2", "summary": "hanging event two",
            "app_name": "Y", "metadata": {},
        })

        # Both items should be consumed within timeout budget:
        # 2 items × 2s _PROCESS_ITEM_TIMEOUT + margin
        await _wait_queue_empty(agent, timeout=8.0)
        assert agent._running
        assert len(agent._recent_items) == 2

        await _stop_agent(agent, task)

    async def test_hanging_llm_chat_returns_within_timeout(self, db):
        """chat() with a hanging LLM should return within _CHAT_TIMEOUT,
        not hang forever."""
        agent = _make_agent(db, llm=HangingLLM())
        agent._CHAT_TIMEOUT = 2
        agent._LLM_CALL_TIMEOUT = 1

        start = time.monotonic()
        response = await agent.chat("hello?")
        elapsed = time.monotonic() - start

        assert elapsed < 5.0
        assert "couldn't generate" in response.lower() or "too long" in response.lower()

    async def test_slow_llm_doesnt_block_queue_forever(self, db):
        """A slow (but finishing) LLM call should not prevent subsequent
        items from eventually being processed."""
        llm = SlowLLM(delay=0.5, response="slow but ok")
        agent = _make_agent(db, llm=llm)
        task = await _run_agent_briefly(agent, 0.2)

        for i in range(2):
            await agent.push("event", {
                "agent_name": f"slow{i}",
                "summary": f"slow event {i}",
                "app_name": "X", "metadata": {},
            })

        await _wait_queue_empty(agent, timeout=5.0)
        await _stop_agent(agent, task)


# ── 3. Concurrent pressure ───────────────────────────────────


@pytest.mark.asyncio
class TestConcurrentPressure:
    async def test_chat_and_push_concurrent(self, db):
        """chat() and push items arriving simultaneously — neither should
        starve or deadlock the other."""
        ws = SpyConnectionManager()
        llm = StubLLM("reply")
        agent = _make_agent(db, llm=llm, ws_manager=ws)
        task = await _run_agent_briefly(agent, 0.2)

        async def do_chats():
            for _ in range(3):
                await agent.chat("hello")

        async def do_pushes():
            for i in range(5):
                await agent.push("event", {
                    "agent_name": "concurrent", "summary": f"concurrent event {i}",
                    "app_name": "App", "metadata": {},
                })

        await asyncio.wait_for(
            asyncio.gather(do_chats(), do_pushes()),
            timeout=15.0,
        )

        # All 3 chats should have produced conversation broadcasts
        assert len(_conversations(ws)) == 3

        await _stop_agent(agent, task)

    async def test_rapid_pushes_all_consumed(self, db):
        """10 pushes in rapid succession — all should be consumed."""
        agent = _make_agent(db, llm=StubLLM("ok"))
        task = await _run_agent_briefly(agent, 0.2)

        for i in range(10):
            await agent.push("event", {
                "agent_name": f"rapid{i}",
                "summary": f"rapid unique event {i} about topic {i * 7}",
                "app_name": f"App{i}", "metadata": {},
            })

        await _wait_queue_empty(agent, timeout=15.0)
        assert len(agent._recent_items) == 10

        await _stop_agent(agent, task)


# ── 4. Tool-calling loop ─────────────────────────────────────


@pytest.mark.asyncio
class TestToolLoop:
    async def test_generate_response_with_tool_calls(self, db):
        """_generate_response should execute tool calls and return the
        final text response."""
        llm = ToolCallingLLM()
        agent = _make_agent(db, llm=llm)

        fid = await db.insert_frame(Frame(
            timestamp="2025-01-01T00:00:00Z", capture_trigger="manual",
        ))
        await db.insert_event(fid, Event(
            agent_name="test", app_type=AppType.BROWSER, summary="test event",
        ))

        response = await agent._generate_response("what happened?", "")
        assert response == "found 3 events"

    async def test_hanging_tool_is_killed_by_timeout(self, db):
        """A tool that hangs should be timed out, not block the loop."""
        llm = ToolCallingLLM()
        agent = _make_agent(db, llm=llm)
        agent._TOOL_CALL_TIMEOUT = 1

        # Replace db_query tool with a hanging one
        from src.general_agent.tools import RegisteredTool
        from src.llm.types import ToolSpec as TS

        async def hanging_tool(**kwargs):
            await asyncio.Event().wait()
            return "{}"

        agent._tools.register(RegisteredTool(
            spec=TS(name="db_query", description="test", parameters={}),
            fn=hanging_tool,
        ))

        response = await asyncio.wait_for(
            agent._generate_response("what happened?", ""),
            timeout=10.0,
        )
        # Should still get a response (tool timed out, LLM got error, replied)
        assert response == "found 3 events"


# ── 5. Overlay message handling ──────────────────────────────


@pytest.mark.asyncio
class TestOverlayMessages:
    async def test_action_more_triggers_chat(self, db):
        """'more' action from overlay fires a chat() task and broadcasts
        the response."""
        ws = SpyConnectionManager()
        llm = StubLLM("more details here")
        agent = _make_agent(db, llm=llm, ws_manager=ws)

        await agent._handle_overlay_message({
            "type": "action",
            "action": "more",
            "text": "that price thing",
        })

        # chat() is fire-and-forget via create_task — let it complete
        await asyncio.sleep(0.3)

        convos = _conversations(ws)
        assert len(convos) == 1
        assert "more details here" in convos[0]["text"]

    async def test_action_accept_does_not_crash(self, db):
        """'accept' action is logged but doesn't cause errors."""
        agent = _make_agent(db)
        await agent._handle_overlay_message({
            "type": "action",
            "action": "accept",
        })
        # No crash = success

    async def test_action_dismiss_does_not_crash(self, db):
        """'dismiss' action is logged but doesn't cause errors."""
        agent = _make_agent(db)
        await agent._handle_overlay_message({
            "type": "action",
            "action": "dismiss",
        })

    async def test_unknown_message_type_ignored(self, db):
        """Unknown message types are silently ignored."""
        agent = _make_agent(db)
        await agent._handle_overlay_message({
            "type": "unknown_thing",
            "data": "whatever",
        })


# ── 6. Broadcast with no ws_manager ──────────────────────────


@pytest.mark.asyncio
class TestNoWebSocket:
    async def test_ws_send_with_no_manager_is_noop(self, db):
        """_ws_send with no ws_manager should silently do nothing."""
        tools = ToolRegistry(db)
        agent = GeneralAgent(
            db=db, tools=tools, llm=StubLLM(),
            ws_manager=None,
        )

        # Should not raise
        await agent._ws_send({"type": "test"})

    async def test_chat_works_without_ws_manager(self, db):
        """chat() should still return a response even with no ws_manager."""
        tools = ToolRegistry(db)
        agent = GeneralAgent(
            db=db, tools=tools, llm=StubLLM("no ws reply"),
            ws_manager=None,
        )

        response = await agent.chat("hello")
        assert response == "no ws reply"

    async def test_run_loop_works_without_ws_manager(self, db):
        """The run loop should process items even with no ws_manager."""
        tools = ToolRegistry(db)
        agent = GeneralAgent(
            db=db, tools=tools, llm=StubLLM("noted"),
            ws_manager=None,
        )
        task = await _run_agent_briefly(agent, 0.2)

        await agent.push("event", {
            "agent_name": "test",
            "summary": "something to process",
            "app_name": "X", "metadata": {},
        })

        await _wait_queue_empty(agent, timeout=5.0)
        await _stop_agent(agent, task)


# ── 7. Lifecycle ─────────────────────────────────────────────


@pytest.mark.asyncio
class TestLifecycle:
    async def test_stop_is_idempotent(self, db):
        """Calling stop() twice should not raise."""
        agent = _make_agent(db)
        task = await _run_agent_briefly(agent, 0.2)

        await agent.stop()
        await agent.stop()

        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    async def test_status_fields(self, db):
        """status() returns expected fields."""
        agent = _make_agent(db)
        s = agent.status()
        assert s["running"] is False
        assert s["queue_depth"] == 0
        assert "ws_clients" in s
        assert "conversation_turns" in s

    async def test_empty_summary_skipped(self, db):
        """Items with no summary are dropped without LLM call."""
        llm = StubLLM()
        agent = _make_agent(db, llm=llm)
        task = await _run_agent_briefly(agent, 0.2)

        await agent.push("event", {
            "agent_name": "test",
            "summary": "",
            "metadata": {},
        })

        await _wait_queue_empty(agent, timeout=3.0)
        assert len(llm.calls) == 0

        await _stop_agent(agent, task)


# ── 8. ConnectionManager unit tests ──────────────────────────


@pytest.mark.asyncio
class TestConnectionManager:
    async def test_broadcast_with_no_clients(self):
        """broadcast() with no clients is a no-op."""
        mgr = ConnectionManager()
        await mgr.broadcast({"type": "test"})
        assert mgr.client_count == 0

    async def test_dead_client_removed_on_broadcast(self):
        """A client that errors on send is removed from the pool."""
        mgr = ConnectionManager()

        # Fake a WebSocket that raises on send
        dead_ws = AsyncMock()
        dead_ws.send_text = AsyncMock(side_effect=ConnectionError("gone"))
        dead_ws.accept = AsyncMock()
        await mgr.connect(dead_ws)
        assert mgr.client_count == 1

        await mgr.broadcast({"type": "test"})
        assert mgr.client_count == 0


# ── 9. /ws endpoint error boundary ──────────────────────────


@pytest.mark.asyncio
class TestWSEndpointErrorBoundary:
    async def test_handle_overlay_message_error_doesnt_kill_connection(self, db):
        """If _handle_overlay_message raises, the WS endpoint should catch it
        and keep the connection alive — not disconnect the overlay."""
        ws = SpyConnectionManager()
        agent = _make_agent(db, llm=ExplodingLLM(), ws_manager=ws)

        # "more" action calls chat() which calls the LLM which explodes.
        # _handle_overlay_message should NOT propagate the error.
        # (The error boundary is in server.py, but we test the agent side here:
        #  chat() returns a fallback, so _handle_overlay_message won't raise.)
        await agent._handle_overlay_message({
            "type": "action",
            "action": "more",
            "text": "tell me more",
        })

        # chat() is fire-and-forget via create_task — let it complete
        await asyncio.sleep(0.3)

        # Should have broadcast a fallback response, not crashed
        convos = _conversations(ws)
        assert len(convos) == 1
        assert "couldn't generate" in convos[0]["text"].lower()


# ── 10. IntelligenceLayer reasoning timeout ──────────────────


@pytest.mark.asyncio
class TestIntelligenceLayerTimeout:
    async def test_hanging_reasoning_agent_is_timed_out(self, db):
        """A reasoning agent that hangs forever should be killed by
        _REASON_TIMEOUT, not block the intelligence queue."""
        from src.intelligence.reasoning_agent import ReasoningAgent
        from src.storage.models import Action, AppType

        class HangingReasoningAgent(ReasoningAgent):
            @property
            def name(self) -> str:
                return "hanger"

            async def reason(self, event, db):
                await asyncio.Event().wait()

        from src.intelligence.intelligence_layer import IntelligenceLayer

        layer = IntelligenceLayer(agents=[HangingReasoningAgent()], db=db)
        layer._REASON_TIMEOUT = 1  # 1s timeout for testing

        task = asyncio.create_task(layer.run())

        # Seed a frame + event
        fid = await db.insert_frame(Frame(
            timestamp="2025-01-01T00:00:00Z", capture_trigger="manual",
        ))
        eid = await db.insert_event(fid, Event(
            agent_name="test", app_type=AppType.IDE, summary="test",
        ))

        evt = Event(id=eid, frame_id=fid, agent_name="test",
                    app_type=AppType.IDE, summary="test")
        await layer.submit(evt)

        # Should process (and timeout) within 3s, not hang
        await asyncio.sleep(2.0)

        # Push a second event — if the layer is stuck, the queue won't drain
        await layer.submit(evt)
        await asyncio.sleep(2.0)

        # Queue should be empty — both events consumed despite timeouts
        assert layer._event_queue.empty()

        await layer.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_timeout_doesnt_break_working_agents(self, db):
        """A fast reasoning agent should still work normally with timeouts."""
        from src.intelligence.reasoning_agent import ReasoningAgent
        from src.intelligence.intelligence_layer import IntelligenceLayer
        from src.storage.models import Action, AppType

        class FastReasoningAgent(ReasoningAgent):
            @property
            def name(self) -> str:
                return "fast"

            async def reason(self, event, db):
                return Action(
                    event_id=event.id, frame_id=event.frame_id,
                    agent_name="fast", action_type="log",
                    action_description="done fast",
                )

        layer = IntelligenceLayer(agents=[FastReasoningAgent()], db=db)
        task = asyncio.create_task(layer.run())

        fid = await db.insert_frame(Frame(
            timestamp="2025-01-01T00:00:00Z", capture_trigger="manual",
        ))
        eid = await db.insert_event(fid, Event(
            agent_name="test", app_type=AppType.IDE, summary="test",
        ))

        await layer.submit(Event(
            id=eid, frame_id=fid, agent_name="test",
            app_type=AppType.IDE, summary="test",
        ))
        await asyncio.sleep(0.5)

        actions = await db.get_actions_for_event(eid)
        assert len(actions) == 1
        assert actions[0]["action_type"] == "log"

        await layer.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
