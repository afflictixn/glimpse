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
from src.intelligence.intelligence_layer import IntelligenceLayer
from src.ocr.apple_vision import perform_ocr
from src.process.process_agent import ProcessAgent
from src.storage.database import DatabaseManager
from src.storage.models import Frame
from src.storage.snapshot_writer import SnapshotWriter

logger = logging.getLogger(__name__)


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
        intelligence_layer: IntelligenceLayer | None = None,
    ) -> None:
        self._settings = settings
        self._db = db
        self._writer = snapshot_writer
        self._queue = trigger_queue
        self._activity = activity_feed
        self._agents = process_agents
        self._providers = context_providers
        self._intelligence = intelligence_layer
        self._comparer = FrameComparer()
        self._running = False
        self._last_capture_time = 0.0
        self._last_visual_check = 0.0
        self._capture_count = 0

    async def run(self) -> None:
        self._running = True
        poll_s = self._settings.poll_interval_ms / 1000.0
        logger.info("Capture loop started (poll every %dms)", self._settings.poll_interval_ms)

        while self._running:
            try:
                trigger = await self._poll_trigger()
                if trigger is not None:
                    now = time.monotonic()
                    elapsed_ms = (now - self._last_capture_time) * 1000
                    if elapsed_ms >= self._settings.min_capture_interval_ms:
                        await self._do_capture(trigger)
                        self._last_capture_time = now
                        self._activity.mark_captured()

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
        try:
            image = await asyncio.to_thread(capture_screen)
        except Exception:
            logger.error("Screenshot capture failed", exc_info=True)
            return

        app_name, window_name = "", ""
        try:
            app_name, window_name = await asyncio.to_thread(get_focused_app)
        except Exception:
            logger.debug("get_focused_app failed", exc_info=True)

        snapshot_path = await asyncio.to_thread(self._writer.save, image)

        content_hash = hashlib.md5(
            image.resize((image.width // 8, image.height // 8), Image.Resampling.NEAREST).tobytes()
        ).hexdigest()

        frame = Frame(
            timestamp=datetime.now(timezone.utc).isoformat(),
            snapshot_path=snapshot_path,
            app_name=app_name or None,
            window_name=window_name or None,
            focused=True,
            capture_trigger=trigger.value,
            content_hash=content_hash,
        )
        frame_id = await self._db.insert_frame(frame)

        try:
            ocr_result = await asyncio.to_thread(perform_ocr, image)
            await self._db.insert_ocr(frame_id, ocr_result)
            ocr_text = ocr_result.text
        except Exception:
            logger.error("OCR failed for frame %d", frame_id, exc_info=True)
            ocr_text = ""

        agent_coros = [
            agent.process(image, ocr_text, app_name or None, window_name or None)
            for agent in self._agents
        ]
        provider_coros = [
            provider.collect(app_name or None, window_name or None)
            for provider in self._providers
        ]

        all_results = await asyncio.gather(
            *agent_coros, *provider_coros, return_exceptions=True
        )

        agent_results = all_results[: len(self._agents)]
        provider_results = all_results[len(self._agents) :]

        for i, result in enumerate(agent_results):
            if isinstance(result, BaseException):
                logger.error(
                    "ProcessAgent %s failed: %s",
                    self._agents[i].name,
                    result,
                    exc_info=result,
                )
                continue
            if result is not None:
                result.frame_id = frame_id
                event_id = await self._db.insert_event(frame_id, result)
                if self._intelligence and result.id is not None:
                    await self._intelligence.submit(result)

        for i, result in enumerate(provider_results):
            if isinstance(result, BaseException):
                logger.error(
                    "ContextProvider %s failed: %s",
                    self._providers[i].name,
                    result,
                    exc_info=result,
                )
                continue
            if isinstance(result, list):
                for ctx in result:
                    ctx.frame_id = frame_id
                    await self._db.insert_context(frame_id, ctx)

        self._capture_count += 1
        logger.debug(
            "Capture #%d: frame=%d trigger=%s app=%s",
            self._capture_count,
            frame_id,
            trigger.value,
            app_name,
        )

    async def stop(self) -> None:
        self._running = False

    @property
    def capture_count(self) -> int:
        return self._capture_count
