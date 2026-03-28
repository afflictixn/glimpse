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

from src.general_agent.event_filter import EventFilter
from src.general_agent.ollama_client import sanitize_response
from src.general_agent.tools import ToolRegistry
from src.llm.client import LLMClient
from src.llm.types import Message
from src.storage.database import DatabaseManager
from src.voice.tts import VoiceClient

logger = logging.getLogger(__name__)

# How many recent items to keep in the sliding context window
MAX_CONTEXT_ITEMS = 50
MAX_CONVERSATION_TURNS = 30

MAX_TOOL_ROUNDS = 10


SYSTEM_PROMPT = """\
You are Z, the user's ambient assistant. You watch their screen and chat \
naturally — like a knowledgeable friend sitting next to them.

Everything you say is read aloud, so keep it to the point and conversational:
- Talk like a person, not a document. Contractions are good.
- Never pad with filler ("Sure!", "Great question!", "Of course!").
- If you don't know, say so.

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
        voice: VoiceClient | None = None,
        importance_filter_enabled: bool = False,
    ) -> None:
        self._db = db
        self._tools = tools
        self._llm = llm
        self._overlay_ws_url = overlay_ws_url
        self._voice = voice
        self._voice_enabled = voice is not None  # mirrors overlay setting

        # Processing queue — events and actions arrive here via /push
        self._queue: asyncio.Queue[PushItem] = asyncio.Queue()

        # Sliding context window of recent pushes
        self._recent_items: deque[PushItem] = deque(maxlen=MAX_CONTEXT_ITEMS)

        # Conversation history with the user
        self._conversation: deque[ConversationTurn] = deque(maxlen=MAX_CONVERSATION_TURNS)

        self._filter = EventFilter(importance_filter_enabled=importance_filter_enabled)

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

        # Push the conversation to the overlay
        await self._ws_send({
            "type": "append_conversation",
            "role": "assistant",
            "text": response,
        })

        # Speak the response aloud
        if self._voice and self._voice_enabled:
            asyncio.create_task(self._voice.speak(response))


        return response

    def status(self) -> dict[str, Any]:
        """Return current agent status for GET /status."""
        return {
            "running": self._running,
            "queue_depth": self._queue.qsize(),
            "recent_items": len(self._recent_items),
            "conversation_turns": len(self._conversation),
            "ws_connected": self._ws_connection is not None,
            "tts_enabled": self._voice is not None,
            "voice_enabled": self._voice_enabled,
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
        if self._voice:
            await self._voice.close()

    # ── Item processing ────────────────────────────────────────

    async def _process_item(self, item: PushItem) -> None:
        """Evaluate a pushed event/action and decide whether to surface it."""
        summary = self._extract_summary(item)
        if not summary:
            return

        should_process, importance = self._filter.should_process(item, summary)
        if not should_process:
            return

        notification = await self._analyze_screen_context(item, summary)

        if not notification:
            logger.debug("LLM found nothing noteworthy for %s event", item.data.get("agent_name", "unknown"))
            return

        if self._filter.is_duplicate_notification(notification):
            logger.debug("Filter: duplicate notification — %s", notification[:80])
            return

        await self._ws_send({
            "type": "show_proposal",
            "text": notification,
        })

        # Speak high-importance notifications aloud
        if self._voice and self._voice_enabled and importance >= 0.7:
            asyncio.create_task(self._voice.speak(notification))

        self._filter.record_notification(notification)
        logger.info("Surfaced to overlay: %s", notification[:100])

    async def _analyze_screen_context(self, item: PushItem, summary: str) -> str:
        """Let the LLM decide if the screen context is worth a proactive notification."""
        from datetime import date
        today = date.today().isoformat()

        enrichment = await self._enrich(item)

        metadata = item.data.get("metadata", {})
        app_name = item.data.get("app_name", "")
        app_type = item.data.get("app_type", "")

        context_parts = [f"Screen activity: {summary}"]
        if app_name:
            context_parts.append(f"App: {app_name}")
        if app_type:
            context_parts.append(f"Type: {app_type}")
        if metadata:
            context_parts.append(f"Details: {json.dumps(metadata, default=str)[:500]}")
        if enrichment:
            context_parts.append(f"Related history: {enrichment}")

        context_block = "\n".join(context_parts)

        system = (
            f"You are Z, the user's ambient assistant. Today is {today}. "
            "Everything you say is spoken aloud, so talk like a friend — "
            "short, casual, no filler.\n\n"
            "You just got a screen update. Chime in ONLY if you'd actually "
            "tap a friend on the shoulder for it. Never narrate what's on screen.\n\n"
            "Max one sentence. "
            "Don't repeat yourself. If you've already mentioned something, don't mention it again."
            "If nothing worth saying, respond: NOTHING\n\n"
            "Good: \"Heads up, that book's only 20 pages with bad reviews — "
            "the original paper's free on arXiv.\"\n"
            "Good: \"That header needs more contrast — try #E2E2E2.\"\n"
            "Bad: \"You are viewing a product page on Amazon.\""
        )

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=context_block),
        ]
        try:
            response = await self._llm.complete(messages)
            result = response.content.strip()
            if "NOTHING" in result.upper() or not result:
                return ""
            return result
        except Exception:
            logger.error("Screen context LLM analysis failed", exc_info=True)
            return ""

    def _extract_summary(self, item: PushItem) -> str:
        data = item.data
        if item.kind == "event":
            return data.get("summary", "")
        elif item.kind == "action":
            return data.get("action_description", "")
        return ""

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
            if isinstance(parsed, list) and parsed:
                return f"Related history: {json.dumps(parsed[:2], default=str)}"
        except Exception:
            logger.debug("Enrichment failed", exc_info=True)

        return ""

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

        last_text = ""
        for round_num in range(MAX_TOOL_ROUNDS):
            try:
                response = await self._llm.complete(messages, tools=tool_specs)
            except Exception:
                logger.error("LLM completion failed", exc_info=True)
                return last_text or "Sorry, I couldn't generate a response right now."

            if response.content:
                last_text = response.content

            if not response.tool_calls:
                return response.content or last_text or ""

            logger.info("Tool round %d: calling %s", round_num + 1,
                        ", ".join(tc.name for tc in response.tool_calls))

            messages.append(response)

            for tc in response.tool_calls:
                result = await self._tools.call(tc.name, **tc.arguments)
                messages.append(Message(
                    role="tool",
                    content=result,
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                ))

        return last_text or "I ran out of steps processing your request."

    # ── WebSocket to overlay ───────────────────────────────────

    _WS_SEND_RETRIES = 2
    _WS_CONNECT_TIMEOUT = 5

    async def _ws_ensure_connected(self) -> None:
        """Establish (or re-establish) the overlay WebSocket connection.

        Must be called while holding ``_ws_lock``.
        """
        import websockets

        if self._ws_connection is not None:
            return

        self._ws_connection = await asyncio.wait_for(
            websockets.connect(
                self._overlay_ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ),
            timeout=self._WS_CONNECT_TIMEOUT,
        )
        logger.info("Connected to overlay WebSocket at %s", self._overlay_ws_url)

    async def _ws_close_connection(self) -> None:
        """Close the current connection (if any) without acquiring the lock."""
        if self._ws_connection is not None:
            try:
                await self._ws_connection.close()
            except Exception:
                pass
            self._ws_connection = None

    async def _ws_send(self, message: dict[str, Any]) -> None:
        """Send a JSON message to the overlay via WebSocket.

        Retries once with a fresh connection on failure so that transient
        disconnects (overlay restart, stale socket) don't silently drop
        messages.
        """
        payload = json.dumps(message)

        async with self._ws_lock:
            for attempt in range(1, self._WS_SEND_RETRIES + 1):
                try:
                    await self._ws_ensure_connected()
                    await self._ws_connection.send(payload)
                    return
                except Exception:
                    await self._ws_close_connection()
                    if attempt < self._WS_SEND_RETRIES:
                        logger.debug("WebSocket send failed (attempt %d), reconnecting", attempt)
                    else:
                        logger.warning(
                            "WebSocket send failed after %d attempts, message dropped: %s",
                            self._WS_SEND_RETRIES,
                            message.get("type", "?"),
                        )

    async def _ws_disconnect(self) -> None:
        async with self._ws_lock:
            await self._ws_close_connection()

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
