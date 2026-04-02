"""Shared prompt, schema, and helpers for vision ProcessAgents."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field

from pydantic import BaseModel

from src.storage.models import AppType, Event

logger = logging.getLogger(__name__)

VISION_SYSTEM_PROMPT = (
    "You are a screen activity analyzer. Given a screenshot (and optionally OCR text), "
    "produce a JSON object describing the user's current activity. "
    "Keep summary to one sentence. Include 2-3 key observations in metadata."
)


class ScreenActivity(BaseModel):
    """Structured output schema for vision process agents."""

    app_type: AppType
    summary: str
    metadata: dict[str, str] = {}


def build_vision_prompt(
    ocr_text: str,
    app_name: str | None,
    window_name: str | None,
    *,
    include_ocr: bool = False,
) -> str:
    parts = ["Analyze this screenshot."]
    if app_name:
        parts.append(f"Active app: {app_name}")
    if window_name:
        parts.append(f"Window: {window_name}")
    if include_ocr and ocr_text:
        parts.append(f"OCR text:\n{ocr_text[:2000]}")
    return "\n".join(parts)


def parse_vision_response(text: str, *, agent_name: str) -> Event:
    """Parse a JSON vision response into an Event, with fallback on bad JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse vision JSON from %s: %.200s", agent_name, text)
        return Event(
            agent_name=agent_name,
            app_type=AppType.OTHER,
            summary=text[:500],
        )

    try:
        app_type = AppType(parsed.get("app_type", "other"))
    except ValueError:
        app_type = AppType.OTHER

    return Event(
        agent_name=agent_name,
        app_type=app_type,
        summary=parsed.get("summary", ""),
        metadata=parsed.get("metadata", {}),
    )


def screen_activity_to_event(activity: ScreenActivity, *, agent_name: str) -> Event:
    """Convert a parsed ScreenActivity Pydantic model into an Event."""
    return Event(
        agent_name=agent_name,
        app_type=activity.app_type,
        summary=activity.summary,
        metadata=dict(activity.metadata),
    )


# ── Token counting ────────────────────────────────────────────


@dataclass
class _ModelTokens:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


class LlmTokenCounter:
    """Thread-safe accumulator of input/output token counts keyed by model name."""

    def __init__(self) -> None:
        self._counts: dict[str, _ModelTokens] = {}
        self._lock = threading.Lock()

    def record(self, model: str, *, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            if model not in self._counts:
                self._counts[model] = _ModelTokens()
            entry = self._counts[model]
            entry.input_tokens += input_tokens
            entry.output_tokens += output_tokens
            entry.calls += 1

    def get(self, model: str) -> dict[str, int]:
        with self._lock:
            entry = self._counts.get(model)
            if entry is None:
                return {"input_tokens": 0, "output_tokens": 0, "calls": 0}
            return {
                "input_tokens": entry.input_tokens,
                "output_tokens": entry.output_tokens,
                "calls": entry.calls,
            }

    @property
    def totals(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {model: self.get(model) for model in self._counts}

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()
