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
from src.process.process_agent import ProcessAgent
from src.storage.database import DatabaseManager
from src.storage.models import Frame
from src.storage.snapshot_writer import SnapshotWriter

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.general_agent.agent import GeneralAgent

logger = logging.getLogger(__name__)


async def process_frame(
    image: Image.Image,
    app_name: str | None,
    window_name: str | None,
    trigger: str,
    db: DatabaseManager,
    writer: SnapshotWriter,
    agents: list[ProcessAgent],
    providers: list[ContextProvider],
    general_agent: GeneralAgent | None = None,
) -> int:
    """Process a captured frame through the full pipeline.

    Used by both the local CaptureLoop and the external ingest endpoint.
    Returns the frame_id.
    """
    t_start = time.monotonic()

    snapshot_path = await asyncio.to_thread(writer.save, image)
    t_snap = time.monotonic()

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

    logger.debug(
        "frame %d | screenshot saved in %.0fms (%s) app=%s",
        frame_id, (t_snap - t_start) * 1000, snapshot_path, app_name or "?",
    )

    t_ocr_start = time.monotonic()
    try:
        ocr_result = await asyncio.to_thread(perform_ocr, image)
        await db.insert_ocr(frame_id, ocr_result)
        ocr_text = ocr_result.text
        logger.debug(
            "frame %d | OCR done in %.0fms, %d chars, confidence=%.2f",
            frame_id, (time.monotonic() - t_ocr_start) * 1000,
            len(ocr_text), ocr_result.confidence,
        )
    except Exception:
        logger.error("OCR failed for frame %d", frame_id, exc_info=True)
        ocr_text = ""

    t_agents_start = time.monotonic()
    agent_coros = [
        agent.process(image, ocr_text, app_name, window_name)
        for agent in agents
    ]
    provider_coros = [
        provider.collect(app_name, window_name)
        for provider in providers
    ]

    all_results = await asyncio.gather(
        *agent_coros, *provider_coros, return_exceptions=True
    )
    t_agents_done = time.monotonic()

    agent_results = all_results[: len(agents)]
    provider_results = all_results[len(agents) :]

    any_event = False
    for i, result in enumerate(agent_results):
        if isinstance(result, BaseException):
            logger.error(
                "ProcessAgent %s failed: %s",
                agents[i].name,
                result,
                exc_info=result,
            )
            continue
        if result is not None:
            any_event = True
            result.frame_id = frame_id
            await db.insert_event(frame_id, result)

            text_len = len(result.metadata.get("text", ""))
            logger.debug(
                "frame %d | agent=%s produced event in %.0fms, summary=%s, text=%d chars",
                frame_id, result.agent_name,
                (t_agents_done - t_agents_start) * 1000,
                result.summary[:80], text_len,
            )

            if general_agent is not None:
                await general_agent.push("event", {
                    "frame_id": frame_id,
                    "agent_name": result.agent_name,
                    "app_type": result.app_type,
                    "app_name": app_name or "",
                    "window_name": window_name or "",
                    "summary": result.summary,
                    "metadata": result.metadata,
                })

    # Non-browser frames: no agent produces an event, push raw capture to GA
    if not any_event and general_agent is not None:
        logger.debug(
            "frame %d | no agent event (agents ran in %.0fms), pushing screen_capture to GA",
            frame_id, (t_agents_done - t_agents_start) * 1000,
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

    for i, result in enumerate(provider_results):
        if isinstance(result, BaseException):
            logger.error(
                "ContextProvider %s failed: %s",
                providers[i].name,
                result,
                exc_info=result,
            )
            continue
        if isinstance(result, list):
            for ctx in result:
                ctx.frame_id = frame_id
                await db.insert_context(frame_id, ctx)

    logger.debug(
        "frame %d | pipeline total %.0fms (snap=%.0f ocr=%.0f agents=%.0f)",
        frame_id,
        (time.monotonic() - t_start) * 1000,
        (t_snap - t_start) * 1000,
        (t_agents_start - t_ocr_start) * 1000,
        (t_agents_done - t_agents_start) * 1000,
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
        process_agents: list[ProcessAgent],
        context_providers: list[ContextProvider],
        general_agent: GeneralAgent | None = None,
    ) -> None:
        self._settings = settings
        self._db = db
        self._writer = snapshot_writer
        self._queue = trigger_queue
        self._activity = activity_feed
        self._agents = process_agents
        self._providers = context_providers
        self._general_agent = general_agent
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
                # If busy processing, drain the queue but don't start new captures
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
                    agents=self._agents,
                    providers=self._providers,
                    general_agent=self._general_agent,
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
