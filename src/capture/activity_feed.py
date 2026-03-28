from __future__ import annotations

import threading
import time

from src.capture.event_tap import ActivityKind, CaptureTrigger


class ActivityFeed:
    def __init__(
        self,
        typing_pause_delay_ms: int = 500,
        idle_capture_interval_ms: int = 30_000,
    ) -> None:
        self._typing_pause_ms = typing_pause_delay_ms
        self._idle_interval_ms = idle_capture_interval_ms
        self._lock = threading.Lock()

        now = time.monotonic()
        self._last_activity_ms = now
        self._last_keyboard_ms = 0.0
        self._keyboard_count = 0
        self._was_typing = False
        self._last_capture = now

    def record(self, kind: ActivityKind) -> None:
        with self._lock:
            now = time.monotonic()
            self._last_activity_ms = now
            if kind == ActivityKind.KEYBOARD:
                self._last_keyboard_ms = now
                self._keyboard_count += 1

    def poll(self) -> CaptureTrigger | None:
        with self._lock:
            now = time.monotonic()
            keyboard_idle_ms = (now - self._last_keyboard_ms) * 1000 if self._last_keyboard_ms else float("inf")
            is_typing = keyboard_idle_ms < 300

            if self._was_typing and keyboard_idle_ms >= self._typing_pause_ms and not is_typing:
                self._was_typing = False
                self._keyboard_count = 0
                self._last_capture = now
                return CaptureTrigger.TYPING_PAUSE

            if is_typing:
                self._was_typing = True

            time_since_capture_ms = (now - self._last_capture) * 1000
            if time_since_capture_ms >= self._idle_interval_ms:
                self._last_capture = now
                return CaptureTrigger.IDLE

            return None

    def mark_captured(self) -> None:
        with self._lock:
            self._last_capture = time.monotonic()
