"""Tests for WebSocket reconnection in GeneralAgent."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


@pytest.mark.asyncio
class TestWSReconnect:
    """Test the WS send/connect/keepalive logic."""

    async def _make_agent(self):
        """Create a GeneralAgent with mocked dependencies."""
        from src.general_agent.agent import GeneralAgent

        db = MagicMock()
        tools = MagicMock()
        tools.list_specs.return_value = []
        llm = AsyncMock()

        agent = GeneralAgent(
            db=db,
            tools=tools,
            llm=llm,
            overlay_ws_url="ws://localhost:19999",
        )
        return agent

    async def test_ws_send_connects_on_first_call(self):
        agent = await self._make_agent()
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()

        with patch("websockets.connect", AsyncMock(return_value=mock_ws)):
            await agent._ws_send({"type": "test", "text": "hello"})

        mock_ws.send.assert_called_once()
        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["text"] == "hello"

    async def test_ws_send_retries_on_failure(self):
        agent = await self._make_agent()

        call_count = 0
        mock_ws_good = AsyncMock()
        mock_ws_good.send = AsyncMock()

        async def mock_connect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionRefusedError("nope")
            return mock_ws_good

        with patch("websockets.connect", mock_connect):
            await agent._ws_send({"type": "test"})

        assert call_count == 2  # failed once, succeeded on retry
        mock_ws_good.send.assert_called_once()

    async def test_ws_send_reconnects_after_broken_connection(self):
        agent = await self._make_agent()

        # First connection works
        mock_ws1 = AsyncMock()
        mock_ws1.send = AsyncMock()
        mock_ws1.close = AsyncMock()

        mock_ws2 = AsyncMock()
        mock_ws2.send = AsyncMock()
        mock_ws2.close = AsyncMock()

        call_count = 0

        async def mock_connect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_ws1
            return mock_ws2

        with patch("websockets.connect", mock_connect):
            # First send — connects and sends
            await agent._ws_send({"type": "msg1"})
            assert call_count == 1
            mock_ws1.send.assert_called_once()

            # Simulate connection drop
            mock_ws1.send.side_effect = Exception("connection lost")
            agent._ws_connection = mock_ws1

            # Second send — should reconnect and send
            await agent._ws_send({"type": "msg2"})
            assert call_count == 2
            mock_ws2.send.assert_called_once()

    async def test_ws_send_drops_message_after_3_failures(self):
        agent = await self._make_agent()

        async def always_fail(*args, **kwargs):
            raise ConnectionRefusedError("overlay down")

        with patch("websockets.connect", always_fail):
            # Should not raise, just log and drop
            await agent._ws_send({"type": "test"})

        assert agent._ws_connection is None

    async def test_ws_keepalive_reconnects(self):
        agent = await self._make_agent()
        agent._running = True

        connect_count = 0
        mock_ws = AsyncMock()
        mock_ws.ping = AsyncMock()
        mock_ws.close = AsyncMock()

        async def mock_connect(*args, **kwargs):
            nonlocal connect_count
            connect_count += 1
            return mock_ws

        with patch("websockets.connect", mock_connect):
            # Run keepalive for a short time
            task = asyncio.create_task(agent._ws_keepalive())
            await asyncio.sleep(0.5)
            agent._running = False
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Should have connected at least once
        assert connect_count >= 1

    async def test_ws_keepalive_handles_ping_failure(self):
        agent = await self._make_agent()
        agent._running = True

        mock_ws = AsyncMock()
        mock_ws.ping = AsyncMock(side_effect=Exception("ping failed"))
        mock_ws.close = AsyncMock()
        agent._ws_connection = mock_ws

        connect_count = 0
        mock_ws_new = AsyncMock()
        mock_ws_new.ping = AsyncMock()
        mock_ws_new.close = AsyncMock()

        async def mock_connect(*args, **kwargs):
            nonlocal connect_count
            connect_count += 1
            return mock_ws_new

        with patch("websockets.connect", mock_connect):
            task = asyncio.create_task(agent._ws_keepalive())
            await asyncio.sleep(0.5)
            agent._running = False
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Should have reconnected after ping failure
        assert connect_count >= 1

    async def test_ws_disconnect(self):
        agent = await self._make_agent()
        mock_ws = AsyncMock()
        mock_ws.close = AsyncMock()
        agent._ws_connection = mock_ws

        await agent._ws_disconnect()
        mock_ws.close.assert_called_once()
        assert agent._ws_connection is None

    async def test_ws_disconnect_handles_error(self):
        agent = await self._make_agent()
        mock_ws = AsyncMock()
        mock_ws.close = AsyncMock(side_effect=Exception("already closed"))
        agent._ws_connection = mock_ws

        await agent._ws_disconnect()
        assert agent._ws_connection is None
