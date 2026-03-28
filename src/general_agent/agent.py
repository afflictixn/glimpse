"""Core general agent — queue consumer, conversation state, tool dispatch, overlay push.

The agent runs as a continuous asyncio.Task:
1. Consumes items from the processing queue (events + actions pushed via /push)
2. Evaluates each against current context (recent events, conversation history, user prefs)
3. Calls tools to gather more info if needed
4. Formulates a response and pushes to the overlay UI via WebSocket
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from src.general_agent.ollama_client import sanitize_response
from src.general_agent.tools import ToolRegistry
from src.llm.client import LLMClient
from src.llm.types import Message
from src.storage.database import DatabaseManager

logger = logging.getLogger(__name__)

# How many recent items to keep in the sliding context window
MAX_CONTEXT_ITEMS = 50
MAX_CONVERSATION_TURNS = 30

# Suppress duplicate notifications within this window (seconds)
DEDUP_WINDOW_S = 30.0

MAX_TOOL_ROUNDS = 10


SYSTEM_PROMPT = """\
You are Glimpse, a helpful assistant that watches the user's screen and \
proactively surfaces useful information. You have access to the user's \
screen capture history, OCR text, events, and actions stored in a local database.

Use tools when you need to look up information. Prefer the database tools for \
anything the user has seen on screen. Use web_search or price_lookup for \
external information.

Keep responses concise and helpful. If you have no relevant information, say so \
briefly rather than making things up.

{context}\
"""


@dataclass
class PushItem:
    """An event or action pushed into the agent's queue."""
    kind: str  # "event" or "action"
    data: dict[str, Any]
    received_at: float = field(default_factory=time.time)


@dataclass
class ConversationTurn:
    role: str  # "user" or "assistant"
    text: str
    timestamp: float = field(default_factory=time.time)


class GeneralAgent:
    """Long-running conversational agent that sits at the end of the pipeline."""

    def __init__(
        self,
        db: DatabaseManager,
        tools: ToolRegistry,
        llm: LLMClient,
        overlay_ws_url: str = "ws://localhost:9321",
    ) -> None:
        self._db = db
        self._tools = tools
        self._llm = llm
        self._overlay_ws_url = overlay_ws_url

        # Processing queue — events and actions arrive here via /push
        self._queue: asyncio.Queue[PushItem] = asyncio.Queue()

        # Sliding context window of recent pushes
        self._recent_items: deque[PushItem] = deque(maxlen=MAX_CONTEXT_ITEMS)

        # Conversation history with the user
        self._conversation: deque[ConversationTurn] = deque(maxlen=MAX_CONVERSATION_TURNS)

        # Track what we've already surfaced to avoid spam
        self._recent_notifications: deque[tuple[str, float]] = deque(maxlen=100)

        self._running = False
        self._ws_connection: Any | None = None
        self._ws_lock = asyncio.Lock()

    # ── Public interface ───────────────────────────────────────

    async def push(self, kind: str, data: dict[str, Any]) -> None:
        """Called by /push endpoint — drop an event or action into the queue."""
        item = PushItem(kind=kind, data=data)
        await self._queue.put(item)

    async def chat(self, user_message: str) -> str:
        """Called by /chat endpoint — user sends a message, get a response."""
        self._conversation.append(ConversationTurn(role="user", text=user_message))

        # Build context for the response
        context = self._build_context()

        # LLM-driven tool-augmented response
        response = sanitize_response(await self._generate_response(user_message, context))

        self._conversation.append(ConversationTurn(role="assistant", text=response))

        return response

    def status(self) -> dict[str, Any]:
        """Return current agent status for GET /status."""
        return {
            "running": self._running,
            "queue_depth": self._queue.qsize(),
            "recent_items": len(self._recent_items),
            "conversation_turns": len(self._conversation),
            "ws_connected": self._ws_connection is not None,
        }

    # ── Main loop ──────────────────────────────────────────────

    async def run(self) -> None:
        """Background loop: consume queue items, evaluate, act."""
        self._running = True
        logger.info("General agent started (overlay WS: %s)", self._overlay_ws_url)

        while self._running:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            self._recent_items.append(item)

            try:
                await self._process_item(item)
            except Exception:
                logger.error("General agent failed processing item", exc_info=True)

        logger.info("General agent stopped")

    async def stop(self) -> None:
        self._running = False
        await self._ws_disconnect()

    # ── Item processing ────────────────────────────────────────

    async def _process_item(self, item: PushItem) -> None:
        """Evaluate a pushed event/action and decide whether to surface it."""
        summary = self._extract_summary(item)
        if not summary:
            return

        # Dedup: skip if we recently surfaced something very similar
        if self._is_duplicate(summary):
            logger.debug("Skipping duplicate: %s", summary[:80])
            return

        # Decide importance — actions from the intelligence layer are higher signal
        importance = self._assess_importance(item)

        if importance < 0.3:
            logger.debug("Skipping low-importance item: %s", summary[:80])
            return

        # For high-importance items, use LLM with tools to enrich
        enrichment = ""
        if importance >= 0.7:
            enrichment = await self._enrich(item)

        # Build the notification text
        notification = self._format_notification(item, summary, enrichment)

        # Push to overlay
        await self._ws_send({
            "type": "show_proposal",
            "text": notification,
        })

        self._recent_notifications.append((summary, time.time()))
        logger.info("Surfaced to overlay: %s", notification[:100])

    def _extract_summary(self, item: PushItem) -> str:
        data = item.data
        if item.kind == "event":
            return data.get("summary", "")
        elif item.kind == "action":
            return data.get("action_description", "")
        return ""

    def _is_duplicate(self, summary: str) -> bool:
        now = time.time()
        normalized = summary.strip().lower()
        for prev_summary, ts in self._recent_notifications:
            if now - ts < DEDUP_WINDOW_S and prev_summary.strip().lower() == normalized:
                return True
        return False

    def _assess_importance(self, item: PushItem) -> float:
        """Score 0.0–1.0 based on heuristics. Refine over time."""
        score = 0.5

        # Actions from reasoning agents are higher signal than raw events
        if item.kind == "action":
            score += 0.2

        data = item.data

        # Certain action types are inherently important
        high_importance_types = {"flag", "escalate", "warn", "alert"}
        if data.get("action_type", "").lower() in high_importance_types:
            score += 0.2

        # Longer summaries tend to carry more substance
        summary = data.get("summary", "") or data.get("action_description", "")
        if len(summary) > 100:
            score += 0.1

        return min(score, 1.0)

    async def _enrich(self, item: PushItem) -> str:
        """Use tools to gather additional context for a high-importance item."""
        data = item.data
        summary = data.get("summary", "") or data.get("action_description", "")

        # Pull recent related context from the DB
        try:
            result = await self._tools.call(
                "db_query",
                table="events",
                query=summary[:100],
                limit="3",
            )
            parsed = json.loads(result)
            if parsed:
                return f"Related history: {json.dumps(parsed[:2], default=str)}"
        except Exception:
            logger.debug("Enrichment failed", exc_info=True)

        return ""

    def _format_notification(self, item: PushItem, summary: str, enrichment: str) -> str:
        parts = [summary]
        if enrichment:
            parts.append(f"\n\n{enrichment}")
        return "\n".join(parts)

    # ── Conversation / response generation ─────────────────────

    def _build_context(self) -> str:
        """Build context string from recent items + conversation for response generation."""
        parts: list[str] = []

        # Recent screen activity
        if self._recent_items:
            parts.append("Recent screen activity:")
            for item in list(self._recent_items)[-5:]:
                summary = self._extract_summary(item)
                if summary:
                    parts.append(f"  [{item.kind}] {summary[:200]}")

        return "\n".join(parts)

    async def _generate_response(self, user_message: str, context: str) -> str:
        """Run an agentic loop: LLM may call tools, we execute them, repeat until text reply."""
        system = SYSTEM_PROMPT.format(context=f"\n\nCurrent context:\n{context}" if context else "")

        messages: list[Message] = [Message(role="system", content=system)]

        for turn in list(self._conversation)[-10:]:
            messages.append(Message(role=turn.role, content=turn.text))

        messages.append(Message(role="user", content=user_message))

        tool_specs = self._tools.list_specs()

        for _ in range(MAX_TOOL_ROUNDS):
            try:
                response = await self._llm.complete(messages, tools=tool_specs)
            except Exception:
                logger.error("LLM completion failed", exc_info=True)
                return "Sorry, I couldn't generate a response right now."

            if not response.tool_calls:
                return response.content or ""

            messages.append(response)

            for tc in response.tool_calls:
                result = await self._tools.call(tc.name, **tc.arguments)
                messages.append(Message(
                    role="tool",
                    content=result,
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                ))

        return response.content or "I ran out of steps processing your request."

    # ── WebSocket to overlay ───────────────────────────────────

    async def _ws_send(self, message: dict[str, Any]) -> None:
        """Send a JSON message to the overlay via WebSocket."""
        import websockets

        async with self._ws_lock:
            try:
                if self._ws_connection is None:
                    self._ws_connection = await websockets.connect(self._overlay_ws_url)
                    logger.info("Connected to overlay WebSocket at %s", self._overlay_ws_url)

                await self._ws_connection.send(json.dumps(message))
            except Exception:
                logger.debug("WebSocket send failed, will reconnect next time", exc_info=True)
                self._ws_connection = None

    async def _ws_disconnect(self) -> None:
        async with self._ws_lock:
            if self._ws_connection is not None:
                try:
                    await self._ws_connection.close()
                except Exception:
                    pass
                self._ws_connection = None

    # ── WebSocket listener (overlay → agent) ───────────────────

    async def _ws_listen(self) -> None:
        """Listen for messages from the overlay (user actions like 'yes do it', 'tell me more')."""
        import websockets

        while self._running:
            try:
                async with websockets.connect(self._overlay_ws_url) as ws:
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                            await self._handle_overlay_message(msg)
                        except json.JSONDecodeError:
                            logger.debug("Invalid JSON from overlay: %s", raw[:100])
            except Exception:
                logger.debug("Overlay WS listener disconnected, retrying in 5s")
                await asyncio.sleep(5.0)

    async def _handle_overlay_message(self, msg: dict[str, Any]) -> None:
        """Handle a message from the overlay UI."""
        msg_type = msg.get("type", "")
        if msg_type == "action":
            action = msg.get("action", "")
            text = msg.get("text", "")
            if action == "accept":
                logger.info("User accepted proposal")
            elif action == "more":
                logger.info("User wants more details")
                if text:
                    response = await self.chat(f"Tell me more about: {text}")
            elif action == "dismiss":
                logger.info("User dismissed proposal")
        elif msg_type == "pause_toggle":
            paused = msg.get("paused", False)
            logger.info("Overlay pause toggled: %s", paused)
