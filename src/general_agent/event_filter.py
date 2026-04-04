"""EventFilter — pre-LLM filtering for the general agent pipeline.

Owns all dedup and same-context suppression state.
The general agent delegates to `should_process()` before making any LLM call.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.general_agent.agent import PushItem

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.75
SUMMARY_WINDOW_S = 60.0
NOTIFICATION_WINDOW_S = 120.0

_CONTEXT_COOLDOWNS: dict[str, float] = {
    "browser_content": 45.0,
    "screen_capture": 60.0,
}
_DEFAULT_CONTEXT_COOLDOWN = 60.0


class EventFilter:
    """Stateful filter that decides which pushed items warrant an LLM call."""

    def __init__(self) -> None:
        self._recent_summaries: deque[tuple[str, float]] = deque(maxlen=200)
        self._recent_notifications: deque[tuple[str, float]] = deque(maxlen=100)
        self._last_context: dict[str, tuple[str, str, float]] = {}

    def should_process(self, item: PushItem, summary: str) -> bool:
        """Run all filters and return whether the item should be processed.

        Filters execute cheapest-first:
        1. Fuzzy summary dedup (skipped for browser_content — URL dedup covers it)
        2. Same-context suppression (URL-based for browser_content)
        """
        agent = item.data.get("agent_name", "")

        if agent != "browser_content" and self._is_similar_to_recent(summary):
            logger.debug("Filter: fuzzy dedup — %s", summary[:80])
            return False

        if self._is_same_context(item):
            _, disc = self._context_key(item.data)
            logger.debug("Filter: same context — %s (key=%s)", summary[:80], disc[:60])
            return False

        self._record_summary(summary)
        self._record_context(item)
        return True

    def is_duplicate_notification(self, notification: str) -> bool:
        """Check if a notification is too similar to one recently surfaced."""
        now = time.time()
        normalized = notification.strip().lower()
        for prev, ts in self._recent_notifications:
            if now - ts > NOTIFICATION_WINDOW_S:
                continue
            ratio = SequenceMatcher(None, normalized, prev).ratio()
            if ratio >= SIMILARITY_THRESHOLD:
                return True
        return False

    def record_notification(self, notification: str) -> None:
        """Track a surfaced notification for fuzzy dedup."""
        self._recent_notifications.append((notification.strip().lower(), time.time()))

    # ── Fuzzy dedup ───────────────────────────────────────────

    def _is_similar_to_recent(self, summary: str) -> bool:
        now = time.time()
        normalized = summary.strip().lower()
        for prev, ts in self._recent_summaries:
            if now - ts > SUMMARY_WINDOW_S:
                continue
            ratio = SequenceMatcher(None, normalized, prev).ratio()
            if ratio >= SIMILARITY_THRESHOLD:
                return True
        return False

    def _record_summary(self, summary: str) -> None:
        self._recent_summaries.append((summary.strip().lower(), time.time()))

    # ── Same-context suppression ──────────────────────────────

    @staticmethod
    def _context_key(data: dict) -> tuple[str, str]:
        """Derive (app, discriminator) for same-context comparison.

        For browser_content the discriminator is the page URL (different
        products = different context even in the same Chrome window).
        For everything else it's the macOS window title.
        """
        agent = data.get("agent_name", "")
        app = data.get("app_name", "") or data.get("metadata", {}).get("app", "")
        if agent == "browser_content":
            url = data.get("metadata", {}).get("url", "")
            return (app, url)
        window = data.get("window_name", "") or ""
        return (app, window)

    def _is_same_context(self, item: PushItem) -> bool:
        data = item.data
        agent = data.get("agent_name", "")
        if not agent:
            return False

        app, discriminator = self._context_key(data)

        prev = self._last_context.get(agent)
        if prev is None:
            return False

        prev_app, prev_disc, prev_ts = prev
        cooldown = _CONTEXT_COOLDOWNS.get(agent, _DEFAULT_CONTEXT_COOLDOWN)
        if app == prev_app and discriminator == prev_disc and (time.time() - prev_ts) < cooldown:
            return True

        return False

    def _record_context(self, item: PushItem) -> None:
        data = item.data
        agent = data.get("agent_name", "")
        if not agent:
            return
        app, discriminator = self._context_key(data)
        self._last_context[agent] = (app, discriminator, time.time())
