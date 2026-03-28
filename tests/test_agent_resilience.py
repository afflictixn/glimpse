"""Integration tests for agent resilience — WebSocket disconnects, LLM timeouts,
tool hangs, and broadcast failures.

These tests verify that the timeout and error-handling fixes actually prevent
hangs under real failure scenarios. Each test injects a specific failure and
asserts the agent recovers within bounded time.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.general_agent.agent import GeneralAgent, PushItem
from src.general_agent.tools import ToolRegistry
from src.general_agent.ws_manager import ConnectionManager
from src.llm.types import Message, ToolCall, ToolSpec
from src.storage.database import DatabaseManager
from src.config import Settings


# ── Helpers ──────────────────────────────────────────────────


class FakeLLM:
    """Controllable fake LLM that can hang, crash, or return canned responses."""

    def __init__(self) -> None:
        self.response: Message = Message(role="assistant", content="test response")
        self.hang_seconds: float = 0
        self.raise_on_call: Exception | None = None
        self.call_count: int = 0

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Message:
        self.call_count += 1
        if self.raise_on_call:
            raise self.raise_on_call
        if self.hang_seconds:
            await asyncio.sleep(self.hang_seconds)
        return self.response


class HangingWebSocket:
    """Fake FastAPI WebSocket that hangs on send_text."""

    async def send_text(self, data: str) -> None:
        await asyncio.sleep(3600)  # hang forever


class CrashingWebSocket:
    """Fake FastAPI WebSocket that raises on send_text."""

    async def send_text(self, data: str) -> None:
        raise ConnectionError("peer gone")


class SlowWebSocket:
    """Fake FastAPI WebSocket that takes just over the timeout to send."""

    def __init__(self, delay: float = 6.0) -> None:
        self.delay = delay
        self.sent: list[str] = []

    async def send_text(self, data: str) -> None:
        await asyncio.sleep(self.delay)
        self.sent.append(data)


class GoodWebSocket:
    """Fake FastAPI WebSocket that works fine."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, data: str) -> None:
        self.sent.append(data)


@pytest_asyncio.fixture
async def db(tmp_path):
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    database = DatabaseManager(settings)
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def tools(db):
    return ToolRegistry(db)


def make_agent(
    db: DatabaseManager,
    tools: ToolRegistry,
    llm: FakeLLM,
    ws_manager: ConnectionManager | None = None,
) -> GeneralAgent:
    return GeneralAgent(
        db=db,
        tools=tools,
        llm=llm,
        ws_manager=ws_manager,
    )


# ── ConnectionManager tests ─────────────────────────────────


class TestConnectionManager:
    """Test the WS broadcast layer handles failures gracefully."""

    @pytest.mark.asyncio
    async def test_broadcast_to_healthy_client(self):
        mgr = ConnectionManager()
        ws = GoodWebSocket()
        mgr._connections.add(ws)

        await mgr.broadcast({"type": "test", "text": "hello"})

        assert len(ws.sent) == 1
        assert json.loads(ws.sent[0])["text"] == "hello"

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_client(self):
        mgr = ConnectionManager()
        dead = CrashingWebSocket()
        alive = GoodWebSocket()
        mgr._connections.add(dead)
        mgr._connections.add(alive)

        await mgr.broadcast({"type": "test"})

        assert dead not in mgr._connections
        assert alive in mgr._connections
        assert len(alive.sent) == 1

    @pytest.mark.asyncio
    async def test_broadcast_times_out_hanging_client(self):
        """A hanging WS client must not block broadcast indefinitely."""
        mgr = ConnectionManager()
        hanging = HangingWebSocket()
        alive = GoodWebSocket()
        mgr._connections.add(hanging)
        mgr._connections.add(alive)

        start = time.monotonic()
        await mgr.broadcast({"type": "test"})
        elapsed = time.monotonic() - start

        # Must complete within the send timeout (5s) + margin
        assert elapsed < 7.0, f"Broadcast took {elapsed:.1f}s — hanging client blocked it"
        assert hanging not in mgr._connections
        assert len(alive.sent) == 1

    @pytest.mark.asyncio
    async def test_broadcast_no_clients_is_noop(self):
        mgr = ConnectionManager()
        await mgr.broadcast({"type": "test"})  # should not raise

    @pytest.mark.asyncio
    async def test_broadcast_all_dead(self):
        mgr = ConnectionManager()
        mgr._connections.add(CrashingWebSocket())
        mgr._connections.add(CrashingWebSocket())

        await mgr.broadcast({"type": "test"})
        assert mgr.client_count == 0


# ── Agent chat timeout tests ────────────────────────────────


class TestChatTimeout:
    """Verify chat() returns a fallback instead of hanging."""

    @pytest.mark.asyncio
    async def test_chat_returns_on_llm_timeout(self, db, tools, fake_llm):
        fake_llm.hang_seconds = 3600  # hang forever
        agent = make_agent(db, tools, fake_llm)
        agent._CHAT_TIMEOUT = 2  # short timeout for test
        agent._LLM_CALL_TIMEOUT = 1

        start = time.monotonic()
        result = await agent.chat("hello")
        elapsed = time.monotonic() - start

        assert elapsed < 4.0
        assert "sorry" in result.lower() or "too long" in result.lower() or "couldn't" in result.lower()

    @pytest.mark.asyncio
    async def test_chat_returns_on_llm_crash(self, db, tools, fake_llm):
        fake_llm.raise_on_call = RuntimeError("LLM is down")
        agent = make_agent(db, tools, fake_llm)

        result = await agent.chat("hello")
        assert "sorry" in result.lower() or "couldn't" in result.lower()

    @pytest.mark.asyncio
    async def test_chat_succeeds_normally(self, db, tools, fake_llm):
        fake_llm.response = Message(role="assistant", content="hey there")
        agent = make_agent(db, tools, fake_llm)

        result = await agent.chat("hello")
        assert result == "hey there"


# ── Agent tool timeout tests ────────────────────────────────


class TestToolTimeout:
    """Verify tool calls are bounded by _TOOL_CALL_TIMEOUT."""

    @pytest.mark.asyncio
    async def test_hanging_tool_does_not_block_agent(self, db, tools, fake_llm):
        # Register a tool that hangs forever
        from src.general_agent.tools import RegisteredTool

        async def hanging_tool(**kwargs: Any) -> str:
            await asyncio.sleep(3600)
            return "never"

        tools.register(RegisteredTool(
            spec=ToolSpec(name="hang", description="hangs", parameters={}),
            fn=hanging_tool,
        ))

        # LLM calls the hanging tool, then returns text
        call_count = 0

        async def llm_with_tool_call(messages, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="t1", name="hang", arguments={})],
                )
            return Message(role="assistant", content="recovered")

        fake_llm.complete = llm_with_tool_call
        agent = make_agent(db, tools, fake_llm)
        agent._TOOL_CALL_TIMEOUT = 1
        agent._CHAT_TIMEOUT = 10

        start = time.monotonic()
        result = await agent.chat("use the hang tool")
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"Tool hang blocked chat for {elapsed:.1f}s"
        assert result == "recovered"

    @pytest.mark.asyncio
    async def test_crashing_tool_returns_error_to_llm(self, db, tools, fake_llm):
        from src.general_agent.tools import RegisteredTool

        async def crashing_tool(**kwargs: Any) -> str:
            raise ValueError("boom")

        tools.register(RegisteredTool(
            spec=ToolSpec(name="crash", description="crashes", parameters={}),
            fn=crashing_tool,
        ))

        call_count = 0

        async def llm_with_tool_call(messages, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="t1", name="crash", arguments={})],
                )
            # Check that the tool error was passed back
            tool_msg = [m for m in messages if m.role == "tool"]
            assert tool_msg, "Tool result should be in messages"
            assert "error" in tool_msg[-1].content.lower()
            return Message(role="assistant", content="handled")

        fake_llm.complete = llm_with_tool_call
        agent = make_agent(db, tools, fake_llm)

        result = await agent.chat("use the crash tool")
        assert result == "handled"


# ── Process item timeout tests ───────────────────────────────


class TestProcessItemTimeout:
    """Verify the main loop doesn't get stuck on a single item."""

    @pytest.mark.asyncio
    async def test_slow_item_is_skipped(self, db, tools, fake_llm):
        """Push an item that causes a slow LLM call — the loop should skip it."""
        fake_llm.hang_seconds = 3600
        agent = make_agent(db, tools, fake_llm)
        agent._PROCESS_ITEM_TIMEOUT = 1
        agent._LLM_CALL_TIMEOUT = 1

        # Disable the filter so it always processes
        agent._filter.should_process = lambda item, summary: (True, 1.0)

        # Start the agent loop
        task = asyncio.create_task(agent.run())

        # Push an item that will hang on LLM
        await agent.push("event", {"summary": "slow event", "agent_name": "test"})

        # Wait for it to be processed (should time out in ~1s)
        await asyncio.sleep(3)

        # Push another item — the loop should still be alive
        await agent.push("event", {"summary": "fast event", "agent_name": "test"})
        await asyncio.sleep(0.5)

        # The loop processed both items (didn't hang on the first)
        assert agent._queue.qsize() == 0

        await agent.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_run_loop_survives_exception(self, db, tools, fake_llm):
        """An exception in _process_item should not kill the loop."""
        fake_llm.raise_on_call = RuntimeError("kaboom")
        agent = make_agent(db, tools, fake_llm)
        agent._filter.should_process = lambda item, summary: (True, 1.0)

        task = asyncio.create_task(agent.run())

        await agent.push("event", {"summary": "bad event", "agent_name": "test"})
        await asyncio.sleep(0.5)

        # Loop is still alive
        assert agent._running is True

        await agent.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ── WS send with no manager ─────────────────────────────────


class TestWsSendWithoutManager:
    """Agent with ws_manager=None should not crash on _ws_send."""

    @pytest.mark.asyncio
    async def test_ws_send_no_manager_is_noop(self, db, tools, fake_llm):
        agent = make_agent(db, tools, fake_llm, ws_manager=None)
        # Should not raise
        await agent._ws_send({"type": "test"})

    @pytest.mark.asyncio
    async def test_chat_works_without_ws_manager(self, db, tools, fake_llm):
        fake_llm.response = Message(role="assistant", content="no ws")
        agent = make_agent(db, tools, fake_llm, ws_manager=None)

        result = await agent.chat("hello")
        assert result == "no ws"


# ── End-to-end: WS broadcast during chat ────────────────────


class TestChatBroadcast:
    """Chat response should be broadcast to connected WS clients."""

    @pytest.mark.asyncio
    async def test_chat_broadcasts_response(self, db, tools, fake_llm):
        fake_llm.response = Message(role="assistant", content="broadcast me")
        mgr = ConnectionManager()
        ws = GoodWebSocket()
        mgr._connections.add(ws)

        agent = make_agent(db, tools, fake_llm, ws_manager=mgr)
        result = await agent.chat("hello")

        assert result == "broadcast me"
        assert len(ws.sent) == 1
        msg = json.loads(ws.sent[0])
        assert msg["type"] == "append_conversation"
        assert msg["text"] == "broadcast me"

    @pytest.mark.asyncio
    async def test_chat_survives_dead_ws_client(self, db, tools, fake_llm):
        """Chat must return a response even if all WS clients are dead."""
        fake_llm.response = Message(role="assistant", content="still works")
        mgr = ConnectionManager()
        mgr._connections.add(CrashingWebSocket())

        agent = make_agent(db, tools, fake_llm, ws_manager=mgr)
        result = await agent.chat("hello")

        assert result == "still works"
        assert mgr.client_count == 0

    @pytest.mark.asyncio
    async def test_chat_survives_hanging_ws_client(self, db, tools, fake_llm):
        """Chat must not hang if a WS client is unresponsive."""
        fake_llm.response = Message(role="assistant", content="not blocked")
        mgr = ConnectionManager()
        mgr._connections.add(HangingWebSocket())

        agent = make_agent(db, tools, fake_llm, ws_manager=mgr)
        agent._CHAT_TIMEOUT = 10

        start = time.monotonic()
        result = await agent.chat("hello")
        elapsed = time.monotonic() - start

        assert result == "not blocked"
        assert elapsed < 8.0, f"Hanging WS client blocked chat for {elapsed:.1f}s"
