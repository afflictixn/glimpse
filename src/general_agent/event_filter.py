"""EventFilter — pre-LLM filtering for the general agent pipeline.

Owns all dedup, same-context suppression, and importance scoring state.
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

SIMILARITY_THRESHOLD = 0.85
SUMMARY_WINDOW_S = 60.0
NOTIFICATION_WINDOW_S = 30.0
IMPORTANCE_THRESHOLD = 0.5

_CONTEXT_COOLDOWNS: dict[str, float] = {
    "gemma3": 20.0,
    "social_context": 20.0,
}
_DEFAULT_CONTEXT_COOLDOWN = 30.0

_HIGH_SIGNAL_ACTION_TYPES = frozenset({
    "flag", "escalate", "warn", "alert",
    "critique", "presentation_critique",
})


class EventFilter:
    """Stateful filter that decides which pushed items warrant an LLM call."""

    def __init__(self) -> None:
        self._recent_summaries: deque[tuple[str, float]] = deque(maxlen=200)
        self._recent_notifications: deque[tuple[str, float]] = deque(maxlen=100)
        self._last_context: dict[str, tuple[str, str, float]] = {}

    def should_process(self, item: PushItem, summary: str) -> tuple[bool, float]:
        """Run all filters and return (should_process, importance_score).

        Filters execute cheapest-first:
        1. Fuzzy summary dedup
        2. Same-context suppression
        3. Importance scoring
        """
        if self._is_similar_to_recent(summary):
            logger.debug("Filter: fuzzy dedup — %s", summary[:80])
            return False, 0.0

        if self._is_same_context(item):
            logger.debug("Filter: same context — %s", summary[:80])
            return False, 0.0

        importance = self._assess_importance(item)
        if importance < IMPORTANCE_THRESHOLD:
            logger.debug("Filter: low importance (%.2f) — %s", importance, summary[:80])
            return False, importance

        self._record_summary(summary)
        self._record_context(item)
        return True, importance

    def record_notification(self, summary: str) -> None:
        """Track a surfaced notification for exact-match dedup."""
        self._recent_notifications.append((summary, time.time()))

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
        for prev, ts in self._recent_notifications:
            if now - ts > NOTIFICATION_WINDOW_S:
                continue
            if prev.strip().lower() == normalized:
                return True
        return False

    def _record_summary(self, summary: str) -> None:
        self._recent_summaries.append((summary.strip().lower(), time.time()))

    # ── Same-context suppression ──────────────────────────────

    def _is_same_context(self, item: PushItem) -> bool:
        data = item.data
        agent = data.get("agent_name", "")
        if not agent:
            return False

        app = data.get("app_name", "") or data.get("metadata", {}).get("app", "")
        window = data.get("window_name", "") or ""

        prev = self._last_context.get(agent)
        if prev is None:
            return False

        prev_app, prev_window, prev_ts = prev
        cooldown = _CONTEXT_COOLDOWNS.get(agent, _DEFAULT_CONTEXT_COOLDOWN)
        if app == prev_app and window == prev_window and (time.time() - prev_ts) < cooldown:
            return True

        return False

    def _record_context(self, item: PushItem) -> None:
        data = item.data
        agent = data.get("agent_name", "")
        if not agent:
            return
        app = data.get("app_name", "") or data.get("metadata", {}).get("app", "")
        window = data.get("window_name", "") or ""
        self._last_context[agent] = (app, window, time.time())

    # ── Importance scoring ────────────────────────────────────

    def _assess_importance(self, item: PushItem) -> float:
        """Score 0.0–1.0 based on agent-aware heuristics."""
        if item.kind == "action":
            return 0.9

        data = item.data
        score = 0.0
        agent = data.get("agent_name", "")

        if data.get("action_type", "").lower() in _HIGH_SIGNAL_ACTION_TYPES:
            score += 0.3

        if agent == "browser_content":
            score += 0.3

        if agent == "social_context":
            contacts = data.get("metadata", {}).get("contacts", [])
            score += 0.4 if contacts else 0.0

        if agent.startswith("gemma"):
            score += 0.1

        summary = data.get("summary", "") or data.get("action_description", "")
        if len(summary) > 150:
            score += 0.1

        return min(score, 1.0)
