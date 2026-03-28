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
You are Z, a helpful assistant that watches the user's screen and \
proactively surfaces useful information. You have access to the user's \
screen capture history, OCR text, events, and actions stored in a local database.

Use tools when you need to look up information. Prefer the database tools for \
anything the user has seen on screen. Use web_search for external information.

CRITICAL RULES:
- NEVER invent, fabricate, or hallucinate facts. No made-up birthdays, dates, \
trips, plans, or suggestions unless the data explicitly contains them.
- Only mention a birthday if you see an actual date in the tool results.
- Only mention trips or plans if the user's messages or calendar explicitly mention them.
- If you have no relevant context, say nothing or say you have no info. \
Silence is better than a hallucinated suggestion.
- Keep responses concise — one or two sentences max.

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

        self._filter = EventFilter()

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

        # Start WS keepalive in background
        keepalive_task = asyncio.create_task(self._ws_keepalive())

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

        keepalive_task.cancel()
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

        agent_name = item.data.get("agent_name", "")

        # If this looks like a messaging app, enrich with iMessage history
        if self._is_messaging_event(item):
            notification = await self._handle_messaging_event(item, summary)
        elif agent_name == "social_context":
            notification = await self._analyze_social_context(summary)
        else:
            notification = await self._analyze_screen_context(item, summary)

        if not notification:
            logger.debug("LLM found nothing noteworthy for %s event", agent_name or "unknown")
            return

        await self._ws_send({
            "type": "show_proposal",
            "text": notification,
        })

        # Speak high-importance notifications aloud
        if self._voice and self._voice_enabled and importance >= 0.7:
            asyncio.create_task(self._voice.speak(notification))

        self._filter.record_notification(summary)
        logger.info("Surfaced to overlay: %s", notification[:100])

    _WALLPAPER_SIGNALS = [
        "wallpaper", "screensaver", "desktop background", "lock screen",
        "scenic image", "landscape photo", "moment of relaxation",
        "visual distraction", "break during",
    ]

    async def _analyze_screen_context(self, item: PushItem, summary: str) -> str:
        """Let the LLM decide if the screen context is worth a proactive notification."""
        # Pre-filter: skip wallpaper/idle events before wasting an LLM call
        summary_lower = summary.lower()
        if any(sig in summary_lower for sig in self._WALLPAPER_SIGNALS):
            logger.debug("Skipping wallpaper/idle event: %s", summary[:80])
            return ""

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
            "You are Glimpse, an ambient assistant watching the user's screen. "
            f"Today is {today}.\n\n"
            "You just received a screen activity update. Your job is to decide "
            "whether there is something genuinely USEFUL to tell the user — "
            "an insight, a tip, a warning, a deal, a reminder, or context they "
            "might not have.\n\n"
            "Rules:\n"
            "- Do NOT describe what the user is looking at — they already know.\n"
            "- Do NOT say 'I see you are viewing…' or 'You are currently on…'.\n"
            "- Only speak up if you have something actionable or genuinely helpful.\n"
            "- If you have nothing useful to add, respond with exactly: NOTHING\n"
            "- Keep responses to 1-2 sentences, friendly and concise.\n"
            "- IGNORE desktop wallpapers, screensavers, and background images. "
            "A scenic landscape on screen does NOT mean the user is planning a trip. "
            "NEVER suggest trips, travel, or vacation based on what looks like a wallpaper.\n"
            "- IGNORE idle screens, lock screens, and desktop with no active app.\n"
            "- NEVER invent or hallucinate facts, dates, plans, or suggestions.\n\n"
            "Good examples:\n"
            '- "Friendly reminder: your friend Walter\'s birthday is on April 2nd. That\'s in 5 days. Consider getting a gift."\n'
            '- "The color used for the header of the presentation doesn\'t contrast well with the background. Try using <hex code> instead."\n'
            '- "This book is only 20 pages, has 10 reviews and most of them are negative. Take a look at <book title> instead."\n\n'
            "Bad examples (NEVER do these):\n"
            '- "You are viewing a product page on Amazon."\n'
            '- "I see you\'re browsing Reddit."\n'
            '- "You are currently reading an article about AI."\n'
            '- "Beautiful scenery! Have you considered visiting Lake Tahoe?"\n'
            '- "Looks like you might be planning a trip!"'
        )

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=context_block),
        ]
        try:
            response = await self._llm.complete(messages)
            result = response.content.strip()
            logger.info("Screen context LLM response: %.100s", result)
            if "NOTHING" in result.upper() or not result:
                return ""
            return result
        except Exception:
            logger.error("Screen context LLM analysis failed", exc_info=True)
            return ""

    async def _analyze_social_context(self, raw_context: str) -> str:
        """Let the big LLM analyze social/chat context and produce a notification."""
        from datetime import date
        today = date.today().isoformat()

        system = (
            "You are Z, an ambient assistant. The user is chatting with someone. "
            "You have been given their recent chat history and contact info.\n\n"
            f"Today's date is {today}.\n\n"
            "Your HIGHEST priority: if ANY birthday, anniversary, or important date is "
            "mentioned in the conversation (even by the user themselves), calculate how "
            "soon it is and remind the user. For example: 'Walter's birthday is April 2nd — "
            "that's in 5 days! Consider getting a gift.'\n\n"
            "Secondary: action items, unanswered questions, or promises made.\n\n"
            "Write a SHORT (1-2 sentence) friendly notification. "
            "If there is truly nothing worth notifying about, respond with exactly: NOTHING"
        )
        messages = [
            Message(role="system", content=system),
            Message(role="user", content=raw_context),
        ]
        try:
            response = await self._llm.complete(messages)
            result = response.content.strip()
            if "NOTHING" in result.upper() or not result:
                return ""
            return result
        except Exception:
            logger.error("Social context LLM analysis failed", exc_info=True)
            return ""

    _MESSAGING_KEYWORDS = {
        "messages", "imessage", "whatsapp", "telegram", "signal",
        "slack", "discord", "messenger", "wechat",
    }

    def _is_messaging_event(self, item: PushItem) -> bool:
        """Check if this event is from a messaging app based on GemmaAgent output."""
        data = item.data
        app_name = (data.get("app_name") or "").lower()
        summary = (data.get("summary") or "").lower()
        for kw in self._MESSAGING_KEYWORDS:
            if kw in app_name or kw in summary:
                return True
        return False

    async def _handle_messaging_event(self, item: PushItem, summary: str) -> str:
        """When user opens a messaging app, fetch recent chats and look for insights."""
        logger.info("Messaging event detected — fetching recent conversations")

        context_parts = ["User is in a messaging app."]

        try:
            recent_result = await self._tools.call(
                "imessage_recent", limit="5",
            )
            parsed = json.loads(recent_result)
            if isinstance(parsed, list) and parsed:
                for conv in parsed[:3]:
                    contact = conv.get("chat_identifier") or conv.get("display_name") or ""
                    display = conv.get("display_name") or contact
                    if not contact:
                        continue

                    context_parts.append(f"--- Conversation with {display} ({contact}) ---")

                    # Fetch recent messages
                    try:
                        msgs_result = await self._tools.call(
                            "imessage_conversation", contact=contact, limit="20",
                        )
                        msgs = json.loads(msgs_result)
                        if isinstance(msgs, list) and msgs:
                            msg_lines = []
                            for m in msgs[:20]:
                                sender = "me" if m.get("is_from_me") else display
                                text = (m.get("text") or "")[:200]
                                ts = m.get("msg_date") or m.get("timestamp") or ""
                                if text:
                                    msg_lines.append(f"[{ts}] {sender}: {text}")
                            if msg_lines:
                                msg_lines.reverse()
                                context_parts.append("\n".join(msg_lines))
                    except Exception:
                        logger.debug("iMessage conversation fetch failed for %s", contact)

                    # Search this contact's ENTIRE message history for birthday mentions
                    try:
                        bday_result = await self._tools.call(
                            "imessage_search", query="birthday", limit="10",
                        )
                        bday_msgs = json.loads(bday_result)
                        if isinstance(bday_msgs, list) and bday_msgs:
                            # Filter to this contact's messages
                            relevant = [
                                m for m in bday_msgs
                                if contact.lower() in (m.get("chat_identifier") or "").lower()
                                or contact.lower() in (m.get("display_name") or "").lower()
                            ]
                            if relevant:
                                bday_lines = []
                                for m in relevant[:5]:
                                    sender = "me" if m.get("is_from_me") else display
                                    text = (m.get("text") or "")[:200]
                                    ts = m.get("msg_date") or m.get("timestamp") or ""
                                    bday_lines.append(f"[{ts}] {sender}: {text}")
                                context_parts.append(
                                    f"BIRTHDAY-RELATED MESSAGES with {display}:\n"
                                    + "\n".join(bday_lines)
                                )

                        # Also search for "bday" and "born"
                        for keyword in ("bday", "born", "turning"):
                            extra = await self._tools.call(
                                "imessage_search", query=keyword, limit="5",
                            )
                            extra_msgs = json.loads(extra)
                            if isinstance(extra_msgs, list):
                                for m in extra_msgs:
                                    if contact.lower() in (m.get("chat_identifier") or "").lower():
                                        text = (m.get("text") or "")[:200]
                                        ts = m.get("msg_date") or ""
                                        if text:
                                            context_parts.append(f"[{ts}] {display}: {text}")
                    except Exception:
                        logger.debug("Birthday search failed for %s", contact)

                    # Check stored memories about this person
                    try:
                        mem_result = await self._tools.call(
                            "memory_query", query=display or contact, entity_type="person",
                        )
                        mems = json.loads(mem_result)
                        if isinstance(mems, list):
                            for mem in mems[:2]:
                                if isinstance(mem, dict) and mem.get("fact"):
                                    context_parts.append(f"Remembered about {display}: {mem['fact']}")
                    except Exception:
                        pass

        except Exception:
            logger.debug("iMessage recent fetch failed")
            return await self._analyze_screen_context(item, summary)

        if len(context_parts) <= 1:
            return ""

        raw_context = "\n\n".join(context_parts)
        return await self._analyze_social_context(raw_context)

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

    async def _ws_connect(self) -> bool:
        """Establish WS connection. Returns True on success."""
        import websockets

        try:
            if self._ws_connection is not None:
                try:
                    await self._ws_connection.close()
                except Exception:
                    pass
                self._ws_connection = None

            self._ws_connection = await asyncio.wait_for(
                websockets.connect(self._overlay_ws_url, ping_interval=20, ping_timeout=10),
                timeout=5,
            )
            logger.info("WS connected to overlay at %s", self._overlay_ws_url)
            return True
        except Exception:
            self._ws_connection = None
            return False

    async def _ws_keepalive(self) -> None:
        """Background task: maintain WS connection, reconnect on drop."""
        while self._running:
            try:
                if self._ws_connection is None:
                    await self._ws_connect()

                # Check if connection is still alive
                if self._ws_connection is not None:
                    try:
                        await self._ws_connection.ping()
                    except Exception:
                        logger.debug("WS ping failed, reconnecting")
                        self._ws_connection = None
                        await self._ws_connect()

                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(5)

    async def _ws_send(self, message: dict[str, Any]) -> None:
        """Send a JSON message to the overlay via WebSocket."""
        for attempt in range(3):
            try:
                if self._ws_connection is None:
                    if not await self._ws_connect():
                        if attempt < 2:
                            await asyncio.sleep(0.5)
                        continue

                await self._ws_connection.send(json.dumps(message))
                return
            except Exception:
                logger.debug("WS send failed (attempt %d/3)", attempt + 1)
                self._ws_connection = None
                if attempt < 2:
                    await asyncio.sleep(0.5)

        logger.warning("WS send failed after 3 attempts, message dropped")

    async def _ws_disconnect(self) -> None:
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
