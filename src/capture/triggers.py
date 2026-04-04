from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timezone

from PIL import Image

from src.capture.activity_feed import ActivityFeed
from src.capture.event_tap import CaptureTrigger
from src.capture.frame_compare import FrameComparer
from src.capture.screenshot import capture_screen, get_focused_app
from src.config import Settings
from src.context.context_provider import ContextProvider
from src.ocr.apple_vision import perform_ocr
from src.storage.database import DatabaseManager
from src.storage.models import Frame
from src.storage.snapshot_writer import SnapshotWriter

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.capture.browser_content import BrowserContentAgent
    from src.general_agent.agent import GeneralAgent

logger = logging.getLogger(__name__)


async def _save_snapshot(writer: SnapshotWriter, image: Image.Image) -> str:
    return await asyncio.to_thread(writer.save, image)


async def _run_ocr(image: Image.Image) -> tuple[str, float]:
    """Run OCR and return (text, confidence). Returns ("", 0.0) on failure."""
    try:
        result = await asyncio.to_thread(perform_ocr, image)
        return result.text, result.confidence
    except Exception:
        logger.error("OCR failed", exc_info=True)
        return "", 0.0


async def process_frame(
    image: Image.Image,
    app_name: str | None,
    window_name: str | None,
    trigger: str,
    db: DatabaseManager,
    writer: SnapshotWriter,
    providers: list[ContextProvider],
    general_agent: GeneralAgent | None = None,
    browser_agent: BrowserContentAgent | None = None,
) -> int:
    """Process a captured frame through the full pipeline.

    Runs screenshot save, OCR, and browser content extraction concurrently,
    then does sequential DB writes and pushes to GeneralAgent.
    Returns the frame_id.
    """
    t_start = time.monotonic()

    # Fire all three I/O-heavy operations concurrently
    coros: list = [
        _save_snapshot(writer, image),
        _run_ocr(image),
    ]
    if browser_agent is not None:
        coros.append(browser_agent.extract(app_name, window_name))

    results = await asyncio.gather(*coros, return_exceptions=True)

    snapshot_result = results[0]
    ocr_result = results[1]
    browser_result = results[2] if len(results) > 2 else None

    t_parallel = time.monotonic()

    # Unpack snapshot
    if isinstance(snapshot_result, BaseException):
        logger.error("Snapshot save failed: %s", snapshot_result, exc_info=snapshot_result)
        snapshot_path = ""
    else:
        snapshot_path = snapshot_result

    # Unpack OCR
    if isinstance(ocr_result, BaseException):
        logger.error("OCR failed: %s", ocr_result, exc_info=ocr_result)
        ocr_text, ocr_confidence = "", 0.0
    else:
        ocr_text, ocr_confidence = ocr_result

    # Unpack browser extraction
    if isinstance(browser_result, BaseException):
        logger.error("Browser extraction failed: %s", browser_result, exc_info=browser_result)
        browser_event = None
    else:
        browser_event = browser_result

    logger.debug(
        "parallel phase done in %.0fms (snap=%s, ocr=%d chars/%.2f, browser=%s)",
        (t_parallel - t_start) * 1000,
        "ok" if snapshot_path else "FAIL",
        len(ocr_text), ocr_confidence,
        browser_event.agent_name if browser_event else "none",
    )

    # ── Sequential DB writes ──
    content_hash = hashlib.md5(
        image.resize((image.width // 8, image.height // 8), Image.Resampling.NEAREST).tobytes()
    ).hexdigest()

    frame = Frame(
        timestamp=datetime.now(timezone.utc).isoformat(),
        snapshot_path=snapshot_path,
        app_name=app_name,
        window_name=window_name,
        focused=True,
        capture_trigger=trigger,
        content_hash=content_hash,
    )
    frame_id = await db.insert_frame(frame)

    if ocr_text:
        from src.storage.models import OCRResult
        await db.insert_ocr(frame_id, OCRResult(text=ocr_text, text_json="[]", confidence=ocr_confidence))

    # ── Push to GeneralAgent ──
    if browser_event is not None:
        browser_event.frame_id = frame_id
        await db.insert_event(frame_id, browser_event)

        logger.debug(
            "frame %d | browser_content event: %s, text=%d chars",
            frame_id, browser_event.summary[:80],
            len(browser_event.metadata.get("text", "")),
        )

        if general_agent is not None:
            await general_agent.push("event", {
                "frame_id": frame_id,
                "agent_name": browser_event.agent_name,
                "app_type": browser_event.app_type,
                "app_name": app_name or "",
                "window_name": window_name or "",
                "summary": browser_event.summary,
                "metadata": browser_event.metadata,
            })
    elif general_agent is not None:
        logger.debug(
            "frame %d | no browser event, pushing screen_capture to GA",
            frame_id,
        )
        await general_agent.push("event", {
            "frame_id": frame_id,
            "agent_name": "screen_capture",
            "app_type": "other",
            "app_name": app_name or "",
            "window_name": window_name or "",
            "summary": f"Screen: {app_name or 'Unknown'} — {window_name or ''}",
            "metadata": {
                "ocr_text": ocr_text,
                "snapshot_path": snapshot_path,
            },
        })

    # ── Context providers ──
    if providers:
        provider_results = await asyncio.gather(
            *[p.collect(app_name, window_name) for p in providers],
            return_exceptions=True,
        )
        for i, result in enumerate(provider_results):
            if isinstance(result, BaseException):
                logger.error(
                    "ContextProvider %s failed: %s",
                    providers[i].name, result, exc_info=result,
                )
                continue
            if isinstance(result, list):
                for ctx in result:
                    ctx.frame_id = frame_id
                    await db.insert_context(frame_id, ctx)

    logger.debug(
        "frame %d | pipeline total %.0fms (parallel=%.0f)",
        frame_id,
        (time.monotonic() - t_start) * 1000,
        (t_parallel - t_start) * 1000,
    )

    return frame_id


class CaptureLoop:
    def __init__(
        self,
        settings: Settings,
        db: DatabaseManager,
        snapshot_writer: SnapshotWriter,
        trigger_queue: asyncio.Queue[CaptureTrigger],
        activity_feed: ActivityFeed,
        context_providers: list[ContextProvider],
        general_agent: GeneralAgent | None = None,
        browser_agent: BrowserContentAgent | None = None,
    ) -> None:
        self._settings = settings
        self._db = db
        self._writer = snapshot_writer
        self._queue = trigger_queue
        self._activity = activity_feed
        self._providers = context_providers
        self._general_agent = general_agent
        self._browser_agent = browser_agent
        self._comparer = FrameComparer()
        self._running = False
        self._last_capture_time = 0.0
        self._last_visual_check = 0.0
        self._capture_count = 0
        self._processing_task: asyncio.Task | None = None

    async def run(self) -> None:
        self._running = True
        poll_s = self._settings.poll_interval_ms / 1000.0
        logger.info("Capture loop started (poll every %dms)", self._settings.poll_interval_ms)

        while self._running:
            try:
                if self._processing_task is not None and not self._processing_task.done():
                    try:
                        self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    await asyncio.sleep(poll_s)
                    continue

                trigger = await self._poll_trigger()
                if trigger is not None:
                    now = time.monotonic()
                    elapsed_ms = (now - self._last_capture_time) * 1000
                    if elapsed_ms >= self._settings.min_capture_interval_ms:
                        self._processing_task = asyncio.create_task(
                            self._do_capture(trigger)
                        )

                await asyncio.sleep(poll_s)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error("Capture loop error", exc_info=True)
                await asyncio.sleep(1.0)

        logger.info("Capture loop stopped after %d captures", self._capture_count)

    async def _poll_trigger(self) -> CaptureTrigger | None:
        try:
            trigger = self._queue.get_nowait()
            return trigger
        except asyncio.QueueEmpty:
            pass

        trigger = self._activity.poll()
        if trigger is not None:
            return trigger

        now = time.monotonic()
        elapsed_ms = (now - self._last_visual_check) * 1000
        if elapsed_ms >= self._settings.visual_check_interval_ms:
            self._last_visual_check = now
            try:
                image = await asyncio.to_thread(capture_screen)
                distance = await asyncio.to_thread(self._comparer.compare, image)
                if distance > self._settings.visual_change_threshold:
                    return CaptureTrigger.VISUAL_CHANGE
            except Exception:
                logger.debug("Visual change check failed", exc_info=True)

        return None

    async def _do_capture(self, trigger: CaptureTrigger) -> None:
        t_cap_start = time.monotonic()

        try:
            image = await asyncio.to_thread(capture_screen)
        except Exception:
            logger.error("Screenshot capture failed", exc_info=True)
            return
        t_screen = time.monotonic()

        app_name, window_name = "", ""
        try:
            app_name, window_name = await asyncio.to_thread(get_focused_app)
        except Exception:
            logger.debug("get_focused_app failed", exc_info=True)

        try:
            frame_id = await asyncio.wait_for(
                process_frame(
                    image=image,
                    app_name=app_name or None,
                    window_name=window_name or None,
                    trigger=trigger.value,
                    db=self._db,
                    writer=self._writer,
                    providers=self._providers,
                    general_agent=self._general_agent,
                    browser_agent=self._browser_agent,
                ),
                timeout=self._settings.capture_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("Frame processing timed out after %ds (trigger=%s)", self._settings.capture_timeout_s, trigger.value)
            return
        except Exception:
            logger.error("Frame processing failed", exc_info=True)
            return

        self._capture_count += 1
        self._last_capture_time = time.monotonic()
        self._activity.mark_captured()
        total_ms = (time.monotonic() - t_cap_start) * 1000
        screen_ms = (t_screen - t_cap_start) * 1000
        logger.debug(
            "Capture #%d: frame=%d trigger=%s app=%s | total=%.0fms (screen_grab=%.0fms)",
            self._capture_count,
            frame_id,
            trigger.value,
            app_name,
            total_ms,
            screen_ms,
        )

    async def stop(self) -> None:
        self._running = False
        if self._processing_task is not None and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except (asyncio.CancelledError, Exception):
                pass

    @property
    def capture_count(self) -> int:
        return self._capture_count
