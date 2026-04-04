"""WebSocket log handler — streams log records to overlay debug panel.

Attached to ``src.*`` / ``zexp`` loggers when ``--debug`` is active.
Each record becomes a ``{"type": "debug_log", ...}`` WS broadcast.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.general_agent.ws_manager import ConnectionManager

_MAX_MSG_LEN = 300


class WsLogHandler(logging.Handler):
    """Non-blocking handler that queues log records for WS broadcast."""

    def __init__(self, ws_manager: ConnectionManager, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(level=logging.DEBUG)
        self._ws = ws_manager
        self._loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            if len(msg) > _MAX_MSG_LEN:
                msg = msg[:_MAX_MSG_LEN] + "…"

            payload = {
                "type": "debug_log",
                "ts": time.strftime("%H:%M:%S", time.localtime(record.created)),
                "level": record.levelname,
                "source": record.module,
                "msg": msg,
            }
            asyncio.run_coroutine_threadsafe(self._ws.broadcast(payload), self._loop)
        except Exception:
            pass  # never break the app for debug logging
