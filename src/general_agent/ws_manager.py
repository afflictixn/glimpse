"""WebSocket connection manager — broadcasts messages to all connected clients.

The FastAPI `/ws` endpoint registers clients here. The GeneralAgent calls
``broadcast()`` instead of maintaining its own outbound WS connection.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages a set of WebSocket clients and broadcasts JSON messages."""

    _SEND_TIMEOUT = 5  # max seconds per client send

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)
        logger.info("WS client connected (%d total)", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(ws)
        logger.info("WS client disconnected (%d remaining)", len(self._connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send a JSON message to every connected client.

        Dead connections are silently removed.
        """
        payload = json.dumps(message)
        dead: list[WebSocket] = []

        for ws in list(self._connections):
            try:
                await asyncio.wait_for(ws.send_text(payload), timeout=self._SEND_TIMEOUT)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)
            logger.debug("Removed %d dead WS clients", len(dead))

    @property
    def client_count(self) -> int:
        return len(self._connections)
