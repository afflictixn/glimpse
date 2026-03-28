"""Social context agent — detects messaging app activity, scrapes conversation
history and contact info, then pushes enriched context to the GeneralAgent.

Runs as a ProcessAgent in the capture pipeline. Uses a small local LLM
(gemma3:1b) to extract contact names from OCR text, then calls tools
(imessage, contacts, social) to gather context for the big LLM to act on.
"""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from typing import Any

from PIL import Image

from src.process.process_agent import ProcessAgent
from src.storage.models import AppType, Event

logger = logging.getLogger(__name__)

# Apps that indicate a messaging context
_MESSAGING_APPS = {
    "messages", "imessage", "whatsapp", "telegram", "signal",
    "slack", "discord", "messenger", "wechat",
}

_EXTRACT_PROMPT = """\
You are a contact extractor. Given OCR text from a screen capture, determine \
if the user is in a messaging or chat application. If so, extract the name(s), \
email addresses, or phone numbers of people the user is chatting with.

Return ONLY a JSON array of strings. Each string should be a contact name, email, \
or phone number found on screen. Return [] if the screen is NOT a chat app or \
if no contacts are visible. Do NOT invent or hallucinate names."""

_ANALYZE_PROMPT = """\
You are analyzing a chat conversation. Look for ONLY facts explicitly stated in the messages.
Extract any of these if mentioned:
- Birthdays or ages ("my birthday is...", "happy birthday", "turning 30")
- Plans or events ("let's meet Thursday", "dinner at 7", "flying to Paris next week")
- Requests or action items ("can you send me...", "don't forget to...")
- Emotional tone if notable (someone upset, excited, celebrating)

Return a JSON object with:
- "insights": array of short factual strings found in the messages (empty if nothing notable)
- "birthday": string with the date if ANY birthday is mentioned, null otherwise
- "action_items": array of things the user should do/remember, empty if none

ONLY include facts from the actual messages. NEVER invent or assume anything. \
If the conversation is mundane small talk, return {"insights": [], "birthday": null, "action_items": []}."""


class SocialContextAgent(ProcessAgent):
    """Detects messaging activity and enriches it with conversation/contact context."""

    def __init__(
        self,
        tools,  # ToolRegistry — avoid circular import
        ollama_base_url: str = "http://localhost:11434",
        model: str = "gemma3:1b",
        timeout_s: int = 15,
    ) -> None:
        self._tools = tools
        self._base_url = ollama_base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s

    @property
    def name(self) -> str:
        return "social_context"

    async def process(
        self,
        image: Image.Image,
        ocr_text: str,
        app_name: str | None,
        window_name: str | None,
    ) -> Event | None:
        if not ocr_text or len(ocr_text.strip()) < 20:
            return None

        if not self._is_messaging_app(app_name, window_name):
            return None

        names = await self._extract_names(ocr_text)
        if not names:
            return None

        logger.info("Social context agent: detected contacts %s in %s", names, app_name)

        # Gather context for each contact in parallel
        context = await self._gather_context(names)

        if not context:
            return None

        # Analyze the messages with small LLM — extract only real facts
        analysis = await self._analyze_messages(names, context)

        # Build a rich summary for the GeneralAgent
        summary = self._build_summary(names, context, app_name, analysis)

        if not summary:
            return None

        return Event(
            agent_name=self.name,
            app_type=AppType.OTHER,
            summary=summary,
            metadata={
                "contacts": names,
                "app": app_name,
                "context_sources": list(context.keys()),
            },
        )

    def _is_messaging_app(self, app_name: str | None, window_name: str | None) -> bool:
        if not app_name:
            return False
        app_lower = app_name.lower()
        for msg_app in _MESSAGING_APPS:
            if msg_app in app_lower:
                return True
        # Also check window name for web-based messengers
        if window_name:
            win_lower = window_name.lower()
            for msg_app in _MESSAGING_APPS:
                if msg_app in win_lower:
                    return True
        return False

    async def _extract_names(self, ocr_text: str) -> list[str]:
        """Use small Ollama model to extract contact names from OCR text."""
        try:
            raw = await asyncio.to_thread(self._call_ollama, ocr_text[:2000])
        except Exception:
            logger.debug("Name extraction failed", exc_info=True)
            return []

        # Parse JSON response
        try:
            # Strip markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [ln for ln in lines if not ln.strip().startswith("```")]
                cleaned = "\n".join(lines)
            names = json.loads(cleaned)
            if isinstance(names, list):
                return [n for n in names if isinstance(n, str) and len(n) > 1][:5]
        except json.JSONDecodeError:
            logger.debug("Failed to parse name extraction: %s", raw[:200])

        return []

    def _call_ollama(self, ocr_text: str) -> str:
        return self._call_ollama_raw(
            system=_EXTRACT_PROMPT,
            prompt=f"Extract contact names from this messaging app text:\n\n{ocr_text}",
            max_tokens=128,
        )

    def _call_ollama_raw(self, system: str, prompt: str, max_tokens: int = 256) -> str:
        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": max_tokens},
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            data = json.loads(resp.read())
        return data.get("response", "")

    async def _gather_context(self, names: list[str]) -> dict[str, Any]:
        """Call tools to gather context about each contact."""
        context: dict[str, Any] = {}

        for contact_name in names:
            contact_ctx: dict[str, Any] = {}

            # iMessage history — fetch enough to capture context
            try:
                imsg_result = await self._tools.call(
                    "imessage_conversation", contact=contact_name, limit="20",
                )
                parsed = json.loads(imsg_result)
                if parsed and (not isinstance(parsed, dict) or "error" not in parsed):
                    contact_ctx["recent_messages"] = parsed
            except Exception:
                logger.debug("iMessage lookup failed for %s", contact_name)

            # macOS Contacts
            try:
                contacts_result = await self._tools.call(
                    "contacts_search", name=contact_name,
                )
                parsed = json.loads(contacts_result)
                if parsed and isinstance(parsed, list) and len(parsed) > 0:
                    contact_ctx["contact_info"] = parsed[0]
            except Exception:
                logger.debug("Contacts lookup failed for %s", contact_name)

            # Screen history mentions
            try:
                history_result = await self._tools.call(
                    "contact_lookup", name=contact_name,
                )
                parsed = json.loads(history_result)
                if parsed:
                    mentions = parsed.get("event_mentions", [])
                    if mentions:
                        contact_ctx["screen_mentions"] = mentions[:3]
            except Exception:
                logger.debug("History lookup failed for %s", contact_name)

            # Memory
            try:
                memory_result = await self._tools.call(
                    "memory_query", query=contact_name, entity_type="person",
                )
                parsed = json.loads(memory_result)
                if parsed and isinstance(parsed, list) and len(parsed) > 0:
                    contact_ctx["memories"] = parsed
            except Exception:
                logger.debug("Memory lookup failed for %s", contact_name)

            if contact_ctx:
                context[contact_name] = contact_ctx

        return context

    async def _analyze_messages(self, names: list[str], context: dict[str, Any]) -> dict[str, Any]:
        """Use small LLM to extract real facts from message history."""
        # Build the conversation text to analyze
        msg_lines = []
        for name, ctx in context.items():
            msgs = ctx.get("recent_messages", [])
            if not isinstance(msgs, list):
                continue
            for m in msgs[:15]:
                sender = "me" if m.get("is_from_me") or m.get("from") == "me" else name
                text = m.get("text", "")[:200]
                if text:
                    msg_lines.append(f"{sender}: {text}")

        if not msg_lines:
            return {}

        msg_lines.reverse()
        conversation_text = "\n".join(msg_lines)

        try:
            raw = await asyncio.to_thread(
                self._call_ollama_raw,
                system=_ANALYZE_PROMPT,
                prompt=f"Analyze this conversation:\n\n{conversation_text}",
                max_tokens=256,
            )
            # Parse JSON
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [ln for ln in lines if not ln.strip().startswith("```")]
                cleaned = "\n".join(lines)
            return json.loads(cleaned)
        except (json.JSONDecodeError, Exception):
            logger.debug("Message analysis failed: %s", raw[:200] if 'raw' in dir() else "error")
            return {}

    def _build_summary(
        self, names: list[str], context: dict[str, Any], app_name: str | None,
        analysis: dict[str, Any] | None = None,
    ) -> str:
        """Build summary from analyzed facts — only real data, no hallucination."""
        insights = (analysis or {}).get("insights", [])
        birthday = (analysis or {}).get("birthday")
        action_items = (analysis or {}).get("action_items", [])

        # If analysis found nothing interesting, don't push anything
        if not insights and not birthday and not action_items:
            return ""

        parts = [f"User is chatting with {', '.join(names)} in {app_name or 'a messaging app'}."]

        if birthday:
            parts.append(f"BIRTHDAY DETECTED: {birthday}")

        for insight in insights[:5]:
            if isinstance(insight, str) and insight.strip():
                parts.append(f"- {insight}")

        if action_items:
            parts.append("Action items:")
            for item in action_items[:3]:
                if isinstance(item, str) and item.strip():
                    parts.append(f"  - {item}")

        # Include memories if any
        for name, ctx in context.items():
            if "memories" in ctx:
                for mem in ctx["memories"][:2]:
                    if isinstance(mem, dict) and mem.get("fact"):
                        parts.append(f"Remembered about {name}: {mem['fact']}")

        return "\n".join(parts)
