"""Core general agent — queue consumer, conversation state, tool dispatch, overlay push.

The agent runs as a continuous asyncio.Task:
1. Consumes items from the processing queue (events + actions pushed via /push)
2. Evaluates each against current context (recent events, conversation history, user prefs)
3. Calls tools to gather more info if needed
4. Formulates a response and pushes to the overlay UI via WebSocket
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from src.general_agent.event_filter import EventFilter
from src.general_agent.ollama_client import sanitize_response
from src.general_agent.tools import ToolRegistry
from src.general_agent.ws_manager import ConnectionManager
from src.llm.client import LLMClient
from src.llm.types import ContentPart, Message, text_content
from src.storage.database import DatabaseManager
from src.voice.tts import VoiceClient

logger = logging.getLogger(__name__)

# How many recent items to keep in the sliding context window
MAX_CONTEXT_ITEMS = 50
MAX_CONVERSATION_TURNS = 30

MAX_TOOL_ROUNDS = 10

MAX_SCREENSHOT_WIDTH = 1280
SCREENSHOT_JPEG_QUALITY = 70

FRAME_PROMPTS: dict[str, str] = {
    "Amazon product": (
        "The user is viewing a product page. Analyze for red flags:\n"
        "- Low review count or suspicious review patterns\n"
        "- Specific bad reviews mentioning defects even if overall rating is good\n"
        "- Inconsistencies in the product description\n"
        "- Pricing concerns (unusually high/low)\n"
        "If you find something worth mentioning, say it in one sentence.\n"
        "If the page looks fine, respond: NOTHING"
    ),
    "": (
        "The user is browsing a web page. Only speak up if you notice:\n"
        "- Misleading claims or inconsistencies\n"
        "- Scam/phishing indicators\n"
        "- Something genuinely useful or noteworthy\n"
        "If nothing stands out, respond: NOTHING"
    ),
}


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
        ws_manager: ConnectionManager | None = None,
        voice: VoiceClient | None = None,
        importance_filter_enabled: bool = False,
    ) -> None:
        self._db = db
        self._tools = tools
        self._llm = llm
        self._ws_manager = ws_manager
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
        self._chat_lock = asyncio.Lock()

    # ── Public interface ───────────────────────────────────────

    async def push(self, kind: str, data: dict[str, Any]) -> None:
        """Called by /push endpoint — drop an event or action into the queue."""
        item = PushItem(kind=kind, data=data)
        await self._queue.put(item)

    _CHAT_TIMEOUT = 60  # max seconds for entire chat round-trip

    async def chat(self, user_message: str) -> str:
        """Called by /chat endpoint — user sends a message, get a response.

        Serialized via _chat_lock so concurrent calls (e.g. rapid "more"
        actions from the overlay) don't race on _conversation.
        """
        async with self._chat_lock:
            try:
                return await asyncio.wait_for(
                    self._chat_inner(user_message),
                    timeout=self._CHAT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning("Chat timed out after %ds", self._CHAT_TIMEOUT)
                return "Sorry, that took too long — try again?"

    async def _chat_inner(self, user_message: str) -> str:
        context = self._build_context()
        response = sanitize_response(await self._generate_response(user_message, context))

        # Only append to conversation after we have a complete pair
        self._conversation.append(ConversationTurn(role="user", text=user_message))
        self._conversation.append(ConversationTurn(role="assistant", text=response))

        await self._ws_send({
            "type": "append_conversation",
            "role": "assistant",
            "text": response,
        })

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
            "ws_clients": self._ws_manager.client_count if self._ws_manager else 0,
            "tts_enabled": self._voice is not None,
            "voice_enabled": self._voice_enabled,
        }

    # ── Main loop ──────────────────────────────────────────────

    async def run(self) -> None:
        """Background loop: consume queue items, evaluate, act."""
        self._running = True
        logger.info("General agent started (WS broadcast to %d clients)",
                     self._ws_manager.client_count if self._ws_manager else 0)

        while self._running:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Drain excess queue items, keeping only the newest ones
            if self._queue.qsize() >= self._MAX_QUEUE_DEPTH:
                kept: list[PushItem] = [item]
                while not self._queue.empty():
                    try:
                        kept.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                dropped = len(kept) - self._MAX_QUEUE_DEPTH
                if dropped > 0:
                    logger.info("Queue overflow: dropped %d stale items, keeping %d",
                                dropped, self._MAX_QUEUE_DEPTH)
                    kept = kept[-self._MAX_QUEUE_DEPTH:]
                for k in kept:
                    await self._queue.put(k)
                item = await self._queue.get()

            # Drop items that sat in the queue too long
            age = time.time() - item.received_at
            if age > self._STALE_ITEM_S:
                logger.debug("Dropping stale item (%.1fs old): %s",
                             age, self._extract_summary(item)[:80])
                continue

            self._recent_items.append(item)

            try:
                await asyncio.wait_for(
                    self._process_item(item),
                    timeout=self._PROCESS_ITEM_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Processing item timed out after %ds, skipping: %s",
                    self._PROCESS_ITEM_TIMEOUT,
                    self._extract_summary(item)[:100],
                )
            except Exception:
                logger.error("General agent failed processing item", exc_info=True)

        logger.info("General agent stopped")

    async def stop(self) -> None:
        self._running = False
        if self._voice:
            await self._voice.close()

    # ── Item processing ────────────────────────────────────────

    async def _process_item(self, item: PushItem) -> None:
        """Evaluate a pushed event/action and decide whether to surface it."""
        t_start = time.monotonic()
        agent_name = item.data.get("agent_name", "unknown")
        frame_id = item.data.get("frame_id", "?")
        queue_wait = time.time() - item.received_at

        summary = self._extract_summary(item)
        if not summary:
            return

        should_process, importance = self._filter.should_process(item, summary)
        if not should_process:
            logger.debug(
                "GA frame %s | FILTERED (agent=%s, importance=%.2f, waited %.1fs in queue)",
                frame_id, agent_name, importance, queue_wait,
            )
            return

        logger.debug(
            "GA frame %s | START processing agent=%s importance=%.2f queue_wait=%.1fs",
            frame_id, agent_name, importance, queue_wait,
        )

        notification = await self._analyze_screen_context(item, summary)

        t_llm_done = time.monotonic()
        llm_ms = (t_llm_done - t_start) * 1000

        if not notification:
            logger.debug(
                "GA frame %s | LLM returned NOTHING in %.0fms (agent=%s)",
                frame_id, llm_ms, agent_name,
            )
            return

        if self._filter.is_duplicate_notification(notification):
            logger.debug(
                "GA frame %s | duplicate notification suppressed in %.0fms — %s",
                frame_id, llm_ms, notification[:80],
            )
            return

        await self._ws_send({
            "type": "show_proposal",
            "text": notification,
        })

        # Speak high-importance notifications aloud
        if self._voice and self._voice_enabled and importance >= 0.7:
            asyncio.create_task(self._voice.speak(notification))

        self._filter.record_notification(notification)
        total_ms = (time.monotonic() - t_start) * 1000
        logger.info(
            "GA frame %s | SURFACED in %.0fms (llm=%.0fms) agent=%s → %s",
            frame_id, total_ms, llm_ms, agent_name, notification[:100],
        )

    async def _analyze_screen_context(self, item: PushItem, summary: str) -> str:
        """Let the LLM decide if the screen context is worth a proactive notification."""
        agent_name = item.data.get("agent_name", "")
        frame_id = item.data.get("frame_id", "?")

        t_build = time.monotonic()
        if agent_name == "browser_content":
            path_label = "browser"
            messages = self._build_browser_messages(item, summary)
        elif agent_name == "screen_capture":
            path_label = "screen (multimodal)"
            messages = await self._build_screen_messages(item, summary)
        else:
            path_label = "generic"
            messages = self._build_generic_messages(item, summary)
        build_ms = (time.monotonic() - t_build) * 1000

        has_image = any(
            isinstance(m.content, list) and any(p.type == "image" for p in m.content)
            for m in messages
        )
        total_text = sum(
            len(p.text) for m in messages
            for p in (m.content if isinstance(m.content, list) else [])
            if hasattr(p, "text")
        ) + sum(
            len(m.content) for m in messages if isinstance(m.content, str)
        )

        logger.debug(
            "GA frame %s | LLM call: path=%s build=%.0fms image=%s text_chars=%d msgs=%d",
            frame_id, path_label, build_ms, has_image, total_text, len(messages),
        )

        t_llm = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self._llm.complete(messages),
                timeout=self._LLM_CALL_TIMEOUT,
            )
            result = text_content(response).strip()
            llm_ms = (time.monotonic() - t_llm) * 1000
            logger.debug(
                "GA frame %s | LLM responded in %.0fms, %d chars, nothing=%s",
                frame_id, llm_ms, len(result),
                "NOTHING" in result.upper() or not result,
            )
            if "NOTHING" in result.upper() or not result:
                return ""
            return result
        except asyncio.TimeoutError:
            logger.warning("Screen context LLM call timed out after %ds", self._LLM_CALL_TIMEOUT)
            return ""
        except Exception:
            logger.error("Screen context LLM analysis failed", exc_info=True)
            return ""

    def _base_system_prompt(self, frame_type_instruction: str = "") -> str:
        from datetime import date
        today = date.today().isoformat()

        system = (
            f"You are Z, the user's ambient assistant. Today is {today}. "
            "Everything you say is spoken aloud, so talk like a friend — "
            "short, casual, no filler.\n\n"
            "You just got a screen update. Chime in ONLY if you'd actually "
            "tap a friend on the shoulder for it. Never narrate what's on screen.\n\n"
            "Max one sentence. "
            "Don't repeat yourself. If you've already mentioned something, don't mention it again. "
            "If nothing worth saying, respond: NOTHING\n\n"
            "Good: \"Heads up, that book's only 20 pages with bad reviews — "
            "the original paper's free on arXiv.\"\n"
            "Good: \"That header needs more contrast — try #E2E2E2.\"\n"
            "Bad: \"You are viewing a product page on Amazon.\""
        )
        if frame_type_instruction:
            system += f"\n\n{frame_type_instruction}"
        return system

    def _build_browser_messages(self, item: PushItem, summary: str) -> list[Message]:
        metadata = item.data.get("metadata", {})
        url = metadata.get("url", "")
        title = metadata.get("title", "")
        page_text = metadata.get("text", "")
        label = metadata.get("allowlist_label", "")
        app_name = item.data.get("app_name", "")

        instruction = FRAME_PROMPTS.get(label, FRAME_PROMPTS[""])
        system = self._base_system_prompt(instruction)

        user_parts = [f"App: {app_name}"]
        if title:
            user_parts.append(f"Page: {title} ({url})")
        if label:
            user_parts.append(f"Page type: {label}")
        if page_text:
            user_parts.append(f"\nPage content:\n{page_text[:6000]}")

        return [
            Message(role="system", content=system),
            Message(role="user", content="\n".join(user_parts)),
        ]

    async def _build_screen_messages(self, item: PushItem, summary: str) -> list[Message]:
        metadata = item.data.get("metadata", {})
        ocr_text = metadata.get("ocr_text", "")
        snapshot_path = metadata.get("snapshot_path", "")
        app_name = item.data.get("app_name", "")
        window_name = item.data.get("window_name", "")
        frame_id = item.data.get("frame_id", "?")

        system = self._base_system_prompt()

        content_parts: list[ContentPart] = []

        if snapshot_path and Path(snapshot_path).is_file():
            try:
                t_enc = time.monotonic()
                image_data = await asyncio.to_thread(
                    self._encode_screenshot, snapshot_path
                )
                enc_ms = (time.monotonic() - t_enc) * 1000
                content_parts.append(ContentPart(
                    type="image",
                    image_data=image_data,
                    mime_type="image/jpeg",
                ))
                logger.debug(
                    "GA frame %s | screenshot encoded in %.0fms, %d KB",
                    frame_id, enc_ms, len(image_data) // 1024,
                )
            except Exception:
                logger.debug("Failed to load screenshot %s", snapshot_path, exc_info=True)
        else:
            logger.debug("GA frame %s | no screenshot file at %s", frame_id, snapshot_path)

        text_block = f"App: {app_name}"
        if window_name:
            text_block += f"\nWindow: {window_name}"
        if ocr_text:
            text_block += f"\n\nVisible text (OCR):\n{ocr_text[:2000]}"

        content_parts.append(ContentPart(type="text", text=text_block))

        return [
            Message(role="system", content=system),
            Message(role="user", content=content_parts),
        ]

    def _build_generic_messages(self, item: PushItem, summary: str) -> list[Message]:
        """Fallback for any event type not specifically handled."""
        metadata = item.data.get("metadata", {})
        app_name = item.data.get("app_name", "")

        system = self._base_system_prompt()

        context_parts = [f"Screen activity: {summary}"]
        if app_name:
            context_parts.append(f"App: {app_name}")
        if metadata:
            context_parts.append(f"Details: {json.dumps(metadata, default=str)[:500]}")

        return [
            Message(role="system", content=system),
            Message(role="user", content="\n".join(context_parts)),
        ]

    @staticmethod
    def _encode_screenshot(path: str) -> bytes:
        img = Image.open(path).convert("RGB")
        if img.width > MAX_SCREENSHOT_WIDTH:
            ratio = MAX_SCREENSHOT_WIDTH / img.width
            img = img.resize(
                (MAX_SCREENSHOT_WIDTH, int(img.height * ratio)),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=SCREENSHOT_JPEG_QUALITY)
        return buf.getvalue()

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
                response = await asyncio.wait_for(
                    self._llm.complete(messages, tools=tool_specs),
                    timeout=self._LLM_CALL_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning("LLM call timed out after %ds (round %d)",
                               self._LLM_CALL_TIMEOUT, round_num + 1)
                return last_text or "Sorry, I couldn't generate a response right now."
            except Exception:
                logger.error("LLM completion failed", exc_info=True)
                return last_text or "Sorry, I couldn't generate a response right now."

            resp_text = text_content(response)
            if resp_text:
                last_text = resp_text

            if not response.tool_calls:
                return resp_text or last_text or ""

            logger.info("Tool round %d: calling %s", round_num + 1,
                        ", ".join(tc.name for tc in response.tool_calls))

            messages.append(response)

            for tc in response.tool_calls:
                try:
                    result = await asyncio.wait_for(
                        self._tools.call(tc.name, **tc.arguments),
                        timeout=self._TOOL_CALL_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Tool %s timed out after %ds", tc.name, self._TOOL_CALL_TIMEOUT)
                    result = json.dumps({"error": f"Tool {tc.name} timed out"})
                messages.append(Message(
                    role="tool",
                    content=result,
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                ))

        return last_text or "I ran out of steps processing your request."

    # ── Timeouts ──────────────────────────────────────────────

    _PROCESS_ITEM_TIMEOUT = 30  # max seconds for filter + LLM + ws_send per item
    _LLM_CALL_TIMEOUT = 20     # max seconds for a single LLM completion
    _TOOL_CALL_TIMEOUT = 15    # max seconds for a single tool execution
    _STALE_ITEM_S = 30         # drop queued items older than this many seconds
    _MAX_QUEUE_DEPTH = 10      # when queue exceeds this, drain to only the newest items

    # ── WebSocket broadcast to overlay / browser clients ────────

    async def _ws_send(self, message: dict[str, Any]) -> None:
        """Broadcast a JSON message to all connected WS clients."""
        if self._ws_manager is not None:
            await self._ws_manager.broadcast(message)

    # ── Inbound messages from overlay (routed by FastAPI /ws) ──

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
                    asyncio.create_task(self.chat(f"Tell me more about: {text}"))
            elif action == "dismiss":
                logger.info("User dismissed proposal")
        elif msg_type == "pause_toggle":
            paused = msg.get("paused", False)
            logger.info("Overlay pause toggled: %s", paused)
