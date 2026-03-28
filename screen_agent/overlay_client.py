"""Async WebSocket client that talks to the Swift overlay server."""

from __future__ import annotations

import asyncio
import json
import logging
from enum import Enum, auto
from typing import Any, Callable, Optional

import websockets
from websockets.asyncio.client import ClientConnection

log = logging.getLogger(__name__)

RECONNECT_DELAY = 2.0


class UserAction(Enum):
    ACCEPT = auto()
    DISMISS = auto()
    ESCALATE = auto()
    FOLLOW_UP = auto()


class OverlayClient:
    """WebSocket client bridging the agent to the Swift overlay server.

    All public ``send_*`` methods are thread-safe: they schedule the send on
    the asyncio event loop regardless of which thread calls them.
    """

    def __init__(self, url: str = "ws://localhost:9321") -> None:
        self._url = url
        self._ws: Optional[ClientConnection] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stopped = False

        self.on_action: Optional[Callable[[UserAction, str], None]] = None
        self.on_pause: Optional[Callable[[bool], None]] = None

    async def run(self) -> None:
        """Connect (with auto-reconnect) and listen for inbound events."""
        self._loop = asyncio.get_running_loop()
        while not self._stopped:
            try:
                async with websockets.connect(self._url) as ws:
                    self._ws = ws
                    log.info("Connected to overlay server at %s", self._url)
                    await self._recv_loop(ws)
            except asyncio.CancelledError:
                break
            except (ConnectionRefusedError, OSError) as exc:
                log.warning(
                    "Overlay server unreachable (%s), retrying in %.0fs…",
                    exc, RECONNECT_DELAY,
                )
            except websockets.ConnectionClosed:
                log.warning("Overlay connection closed, reconnecting…")
            finally:
                self._ws = None
            if self._stopped:
                break
            await asyncio.sleep(RECONNECT_DELAY)

    def request_stop(self) -> None:
        """Signal the run loop to exit. Thread-safe."""
        self._stopped = True
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._cancel_all_tasks)

    def _cancel_all_tasks(self) -> None:
        for task in asyncio.all_tasks(self._loop):
            task.cancel()

    # -- inbound ---------------------------------------------------------

    async def _recv_loop(self, ws: ClientConnection) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            self._dispatch(msg)

    def _dispatch(self, msg: dict[str, Any]) -> None:
        msg_type = msg.get("type")
        if msg_type == "action":
            action_str = msg.get("action", "")
            text = msg.get("text", "")
            try:
                action = {
                    "dismiss": UserAction.DISMISS,
                    "accept": UserAction.ACCEPT,
                    "escalate": UserAction.ESCALATE,
                    "follow_up": UserAction.FOLLOW_UP,
                }[action_str]
            except KeyError:
                log.warning("Unknown action: %s", action_str)
                return
            if self.on_action:
                self.on_action(action, text)

        elif msg_type == "pause_toggle":
            if self.on_pause:
                self.on_pause(msg.get("paused", False))

    # -- outbound (thread-safe) ------------------------------------------

    def send_show_proposal(self, text: str, proposal_id: Optional[int] = None) -> None:
        self._send({"type": "show_proposal", "text": text, "proposalId": proposal_id})

    def send_show_conversation(self, text: str) -> None:
        self._send({"type": "show_conversation", "text": text})

    def send_append_conversation(self, role: str, text: str) -> None:
        self._send({"type": "append_conversation", "role": role, "text": text})

    def send_set_assistant_label(self, label: str) -> None:
        self._send({"type": "set_assistant_label", "label": label})

    def send_hide(self) -> None:
        self._send({"type": "hide"})

    def _send(self, payload: dict[str, Any]) -> None:
        if self._loop is None or self._ws is None:
            log.debug("Not connected; dropping message %s", payload.get("type"))
            return
        data = json.dumps(payload)
        asyncio.run_coroutine_threadsafe(self._ws.send(data), self._loop)
