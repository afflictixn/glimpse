"""Integration tests for the capture pipeline — CaptureLoop, process_frame, timeouts.

These test the actual fixes for the hanging bug:
1. Capture loop must keep polling for new triggers while a slow agent is processing
2. Frame processing must time out instead of hanging forever
3. After a timeout the loop must recover and process the next trigger promptly
4. Wall-clock timing assertions prove things complete fast, not after 30s
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.capture.activity_feed import ActivityFeed
from src.capture.event_tap import CaptureTrigger
from src.capture.triggers import CaptureLoop, process_frame
from src.config import Settings
from src.process.process_agent import ProcessAgent
from src.storage.database import DatabaseManager
from src.storage.models import AppType, Event, OCRResult
from src.storage.snapshot_writer import SnapshotWriter


# ── Helpers ───────────────────────────────────────────────────


def _make_image(width: int = 100, height: int = 100) -> Image.Image:
    return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))


def _mock_ocr(image):
    return OCRResult(text="mocked ocr text", text_json="[]", confidence=0.99)


class FastAgent(ProcessAgent):
    """Completes instantly, returns an event."""

    def __init__(self, agent_name: str = "fast"):
        self._name = agent_name
        self.call_count = 0
        self.call_timestamps: list[float] = []

    @property
    def name(self) -> str:
        return self._name

    async def process(self, image, ocr_text, app_name, window_name) -> Event | None:
        self.call_count += 1
        self.call_timestamps.append(time.monotonic())
        return Event(
            agent_name=self._name,
            app_type=AppType.BROWSER,
            summary=f"Fast result #{self.call_count} for {app_name}",
            metadata={"window": window_name},
        )


class SlowAsyncAgent(ProcessAgent):
    """Simulates a slow cooperative async call (like an awaited API request)."""

    def __init__(self, delay_s: float = 10.0, agent_name: str = "slow_async"):
        self._delay = delay_s
        self._name = agent_name
        self.call_count = 0
        self.started = asyncio.Event()

    @property
    def name(self) -> str:
        return self._name

    async def process(self, image, ocr_text, app_name, window_name) -> Event | None:
        self.call_count += 1
        self.started.set()
        await asyncio.sleep(self._delay)
        return Event(agent_name=self._name, app_type=AppType.OTHER, summary="Slow")


class BlockingThreadAgent(ProcessAgent):
    """Simulates a blocking I/O call offloaded to a thread (like urllib in GemmaAgent).

    This is the realistic pattern — a sync HTTP call run via to_thread.
    """

    def __init__(self, delay_s: float = 10.0, agent_name: str = "blocking"):
        self._delay = delay_s
        self._name = agent_name
        self.call_count = 0
        self.started = asyncio.Event()

    @property
    def name(self) -> str:
        return self._name

    async def process(self, image, ocr_text, app_name, window_name) -> Event | None:
        self.call_count += 1
        self.started.set()
        # This is how real blocking I/O works in this codebase (see GemmaAgent)
        await asyncio.to_thread(time.sleep, self._delay)
        return Event(agent_name=self._name, app_type=AppType.OTHER, summary="Blocking")


class FailingAgent(ProcessAgent):
    """Raises an exception every time."""

    def __init__(self, agent_name: str = "failing"):
        self._name = agent_name
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def process(self, image, ocr_text, app_name, window_name) -> Event | None:
        self.call_count += 1
        raise RuntimeError(f"Agent {self._name} intentional failure")


def _make_capture_loop(
    settings: Settings,
    db: DatabaseManager,
    agents: list[ProcessAgent],
    trigger_queue: asyncio.Queue | None = None,
) -> CaptureLoop:
    writer = SnapshotWriter(settings)
    queue = trigger_queue or asyncio.Queue()
    activity = ActivityFeed(
        typing_pause_delay_ms=999_999,
        idle_capture_interval_ms=999_999,
    )
    return CaptureLoop(
        settings=settings,
        db=db,
        snapshot_writer=writer,
        trigger_queue=queue,
        activity_feed=activity,
        process_agents=agents,
        context_providers=[],
    )


def _patch_capture():
    """Context manager that patches out macOS-native capture_screen, get_focused_app, and OCR."""
    return _MultiPatch(
        patch("src.capture.triggers.capture_screen", return_value=_make_image()),
        patch("src.capture.triggers.get_focused_app", return_value=("TestApp", "TestWindow")),
        patch("src.capture.triggers.perform_ocr", side_effect=_mock_ocr),
    )


class _MultiPatch:
    """Helper to use multiple patches in a single `with` statement."""

    def __init__(self, *patches):
        self._patches = patches
        self._mocks = []

    def __enter__(self):
        self._mocks = [p.__enter__() for p in self._patches]
        return self._mocks

    def __exit__(self, *args):
        for p in reversed(self._patches):
            p.__exit__(*args)


async def _stop_loop(loop: CaptureLoop, task: asyncio.Task) -> None:
    await loop.stop()
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=3.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass


# ── process_frame tests ──────────────────────────────────────


@pytest.mark.asyncio
class TestProcessFrame:

    async def test_happy_path_writes_to_db(self, db, tmp_settings):
        """Fast agent runs, event lands in the database."""
        agent = FastAgent()
        writer = SnapshotWriter(tmp_settings)

        with patch("src.capture.triggers.perform_ocr", side_effect=_mock_ocr):
            frame_id = await process_frame(
                image=_make_image(),
                app_name="Safari",
                window_name="Google",
                trigger="click",
                db=db,
                writer=writer,
                agents=[agent],
                providers=[],
            )

        assert frame_id >= 1
        events = await db.get_events_for_frame(frame_id)
        assert len(events) == 1
        assert events[0]["agent_name"] == "fast"
        assert "Safari" in events[0]["summary"]

    async def test_failing_agent_doesnt_kill_sibling(self, db, tmp_settings):
        """One agent crashes, the other still saves its event."""
        fast = FastAgent()
        failing = FailingAgent()
        writer = SnapshotWriter(tmp_settings)

        with patch("src.capture.triggers.perform_ocr", side_effect=_mock_ocr):
            frame_id = await process_frame(
                image=_make_image(),
                app_name="Terminal",
                window_name="zsh",
                trigger="click",
                db=db,
                writer=writer,
                agents=[fast, failing],
                providers=[],
            )

        events = await db.get_events_for_frame(frame_id)
        assert len(events) == 1
        assert events[0]["agent_name"] == "fast"

    async def test_wait_for_kills_slow_process_frame(self, db, tmp_settings):
        """asyncio.wait_for can cancel a hanging process_frame — this is
        what _do_capture relies on for its timeout."""
        slow = SlowAsyncAgent(delay_s=30)
        writer = SnapshotWriter(tmp_settings)

        t0 = time.monotonic()
        with patch("src.capture.triggers.perform_ocr", side_effect=_mock_ocr):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    process_frame(
                        image=_make_image(),
                        app_name="Keynote",
                        window_name="Slides",
                        trigger="app_switch",
                        db=db,
                        writer=writer,
                        agents=[slow],
                        providers=[],
                    ),
                    timeout=0.5,
                )
        elapsed = time.monotonic() - t0

        assert elapsed < 2.0, f"Timeout took {elapsed:.1f}s — should be ~0.5s"
        assert slow.call_count == 1

    async def test_wait_for_kills_blocking_thread_agent(self, db, tmp_settings):
        """Even a thread-blocking agent (like real HTTP calls) gets cancelled."""
        blocking = BlockingThreadAgent(delay_s=30)
        writer = SnapshotWriter(tmp_settings)

        t0 = time.monotonic()
        with patch("src.capture.triggers.perform_ocr", side_effect=_mock_ocr):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    process_frame(
                        image=_make_image(),
                        app_name="Chrome",
                        window_name="Tab",
                        trigger="click",
                        db=db,
                        writer=writer,
                        agents=[blocking],
                        providers=[],
                    ),
                    timeout=0.5,
                )
        elapsed = time.monotonic() - t0

        assert elapsed < 2.0, f"Timeout took {elapsed:.1f}s — should be ~0.5s"


# ── CaptureLoop tests ────────────────────────────────────────


@pytest.mark.asyncio
class TestCaptureLoopNonBlocking:
    """Tests that prove the capture loop doesn't block on slow agents.

    The core of the hanging fix: before, `await _do_capture()` was inline,
    so the `while self._running` loop couldn't advance to poll for new
    triggers. Now it's `create_task`, so the loop keeps polling.
    """

    async def test_loop_keeps_polling_queue_during_slow_processing(self, db, tmp_settings):
        """THE critical regression test.

        Push a trigger that starts slow processing. Then push more triggers.
        Verify the queue is being drained (loop is polling) even while the
        first frame is still processing.

        Before the fix: the loop was stuck at `await _do_capture()` and the
        queue would pile up. After: create_task returns immediately and the
        loop keeps polling.
        """
        tmp_settings.min_capture_interval_ms = 0
        tmp_settings.poll_interval_ms = 50
        tmp_settings.capture_timeout_s = 10
        tmp_settings.visual_check_interval_ms = 999_999

        slow = SlowAsyncAgent(delay_s=10)
        queue: asyncio.Queue[CaptureTrigger] = asyncio.Queue()
        loop = _make_capture_loop(tmp_settings, db, [slow], queue)

        with _patch_capture():
            task = asyncio.create_task(loop.run())

            # Push first trigger — starts slow processing
            await queue.put(CaptureTrigger.APP_SWITCH)
            # Wait for the slow agent to actually start
            await asyncio.wait_for(slow.started.wait(), timeout=2.0)

            # Now push more triggers while the first is still processing
            for _ in range(3):
                await queue.put(CaptureTrigger.CLICK)

            # Give the loop time to poll and drain the queue
            await asyncio.sleep(0.3)

            # The queue should be drained — the loop is polling and consuming
            # triggers even though the first frame is still being processed.
            # Before the fix, these would sit in the queue forever.
            assert queue.qsize() == 0, (
                f"Queue has {queue.qsize()} items stuck — loop is not polling "
                f"(this is the hanging bug)"
            )

            await _stop_loop(loop, task)

    async def test_wall_clock_second_trigger_completes_promptly(self, db, tmp_settings):
        """Slow first trigger + fast second trigger should complete in
        ~timeout seconds, not ~slow_agent seconds.

        Uses wall-clock timing to prove the loop recovered from the timeout
        and processed the second trigger promptly.
        """
        tmp_settings.min_capture_interval_ms = 0
        tmp_settings.poll_interval_ms = 50
        tmp_settings.capture_timeout_s = 0.5
        tmp_settings.visual_check_interval_ms = 999_999

        call_number = 0

        class HangThenFastAgent(ProcessAgent):
            @property
            def name(self):
                return "hang_then_fast"

            async def process(self, image, ocr_text, app_name, window_name):
                nonlocal call_number
                call_number += 1
                if call_number == 1:
                    await asyncio.sleep(30)  # simulate a Gemini hang
                return Event(
                    agent_name="hang_then_fast",
                    app_type=AppType.BROWSER,
                    summary=f"call {call_number}",
                )

        queue: asyncio.Queue[CaptureTrigger] = asyncio.Queue()
        loop = _make_capture_loop(tmp_settings, db, [HangThenFastAgent()], queue)

        t0 = time.monotonic()

        with _patch_capture():
            task = asyncio.create_task(loop.run())

            # First trigger — will hang and timeout at 0.5s
            await queue.put(CaptureTrigger.APP_SWITCH)
            await asyncio.sleep(0.8)  # wait for timeout

            # Second trigger — should be processed fast
            await queue.put(CaptureTrigger.CLICK)
            # Wait for it to be processed
            for _ in range(30):
                await asyncio.sleep(0.05)
                if call_number >= 2:
                    break

            elapsed = time.monotonic() - t0
            await _stop_loop(loop, task)

        assert call_number >= 2, f"Only {call_number} calls — second trigger wasn't processed"
        assert elapsed < 3.0, (
            f"Took {elapsed:.1f}s — should be ~1.3s (0.5s timeout + 0.8s wait + fast). "
            f"If it took 30s+, the old blocking bug is back."
        )

        # The second call should have actually written an event to DB
        counts = await db.get_counts()
        assert counts["events"] >= 1

    async def test_timeout_completes_within_budget(self, db, tmp_settings):
        """A 30s-hanging agent with a 0.5s timeout must complete in ~0.5s, not 30s."""
        tmp_settings.min_capture_interval_ms = 0
        tmp_settings.poll_interval_ms = 50
        tmp_settings.capture_timeout_s = 0.5
        tmp_settings.visual_check_interval_ms = 999_999

        slow = SlowAsyncAgent(delay_s=30)
        queue: asyncio.Queue[CaptureTrigger] = asyncio.Queue()
        loop = _make_capture_loop(tmp_settings, db, [slow], queue)

        with _patch_capture():
            task = asyncio.create_task(loop.run())
            await queue.put(CaptureTrigger.APP_SWITCH)

            t0 = time.monotonic()
            # Wait for the processing task to finish (via timeout)
            for _ in range(40):
                await asyncio.sleep(0.05)
                if loop._processing_task and loop._processing_task.done():
                    break
            elapsed = time.monotonic() - t0

            await _stop_loop(loop, task)

        assert loop._processing_task is not None
        assert loop._processing_task.done()
        assert elapsed < 2.0, (
            f"Processing task took {elapsed:.1f}s to finish — timeout is 0.5s. "
            f"If close to 30s, the timeout isn't firing."
        )

    async def test_rapid_triggers_dont_pile_up(self, db, tmp_settings):
        """Rapid-fire triggers during slow processing: the loop should consume
        them from the queue (polling works) but skip capture (agent busy)."""
        tmp_settings.min_capture_interval_ms = 0
        tmp_settings.poll_interval_ms = 50
        tmp_settings.capture_timeout_s = 5
        tmp_settings.visual_check_interval_ms = 999_999

        slow = SlowAsyncAgent(delay_s=0.8, agent_name="medium")
        queue: asyncio.Queue[CaptureTrigger] = asyncio.Queue()
        loop = _make_capture_loop(tmp_settings, db, [slow], queue)

        with _patch_capture():
            task = asyncio.create_task(loop.run())

            # Fire 10 triggers in rapid succession
            for _ in range(10):
                await queue.put(CaptureTrigger.CLICK)

            # Wait for queue to be fully drained
            await asyncio.sleep(0.5)

            queue_drained = queue.qsize() == 0

            # Wait for processing to finish
            await asyncio.sleep(1.5)
            await _stop_loop(loop, task)

        assert queue_drained, (
            f"Queue still has {queue.qsize()} items after 0.5s — "
            f"loop is stuck and not polling"
        )
        # Only 1 or 2 calls, not 10 — most were skipped because agent was busy
        assert slow.call_count < 10, (
            f"Agent was called {slow.call_count} times — triggers should be "
            f"skipped while processing, not queued up"
        )


@pytest.mark.asyncio
class TestCaptureLoopResilience:
    """Tests for error recovery in the capture loop."""

    async def test_screenshot_crash_doesnt_kill_loop(self, db, tmp_settings):
        """First screenshot fails, second succeeds. Loop must survive."""
        tmp_settings.min_capture_interval_ms = 0
        tmp_settings.poll_interval_ms = 50
        tmp_settings.visual_check_interval_ms = 999_999

        agent = FastAgent()
        queue: asyncio.Queue[CaptureTrigger] = asyncio.Queue()
        loop = _make_capture_loop(tmp_settings, db, [agent], queue)

        call_seq = 0

        def flaky_capture():
            nonlocal call_seq
            call_seq += 1
            if call_seq == 1:
                raise RuntimeError("Screen capture failed")
            return _make_image()

        with (
            patch("src.capture.triggers.capture_screen", side_effect=flaky_capture),
            patch("src.capture.triggers.get_focused_app", return_value=("Terminal", "bash")),
            patch("src.capture.triggers.perform_ocr", side_effect=_mock_ocr),
        ):
            task = asyncio.create_task(loop.run())

            await queue.put(CaptureTrigger.CLICK)
            await asyncio.sleep(0.3)

            # The first capture failed (screenshot crash), but the processing
            # task completed (with error), so we can push another trigger
            await queue.put(CaptureTrigger.CLICK)

            for _ in range(30):
                await asyncio.sleep(0.05)
                if agent.call_count >= 1:
                    break

            await _stop_loop(loop, task)

        assert call_seq >= 2, "Loop should have attempted capture after failure"
        assert agent.call_count >= 1, "Second capture should have succeeded"

    async def test_agent_exception_doesnt_kill_loop(self, db, tmp_settings):
        """An agent that always throws doesn't prevent subsequent captures."""
        tmp_settings.min_capture_interval_ms = 0
        tmp_settings.poll_interval_ms = 50
        tmp_settings.capture_timeout_s = 5
        tmp_settings.visual_check_interval_ms = 999_999

        failing = FailingAgent()
        queue: asyncio.Queue[CaptureTrigger] = asyncio.Queue()
        loop = _make_capture_loop(tmp_settings, db, [failing], queue)

        with _patch_capture():
            task = asyncio.create_task(loop.run())

            await queue.put(CaptureTrigger.APP_SWITCH)
            await asyncio.sleep(0.3)
            await queue.put(CaptureTrigger.CLICK)
            await asyncio.sleep(0.3)

            await _stop_loop(loop, task)

        # Both triggers should have been processed (agent called twice)
        # even though it throws every time
        assert failing.call_count >= 2, (
            f"Agent was only called {failing.call_count} times — "
            f"exception should not prevent next capture"
        )

    async def test_ocr_crash_still_runs_agents(self, db, tmp_settings):
        """OCR failure shouldn't prevent agents from processing the frame."""
        agent = FastAgent()
        writer = SnapshotWriter(tmp_settings)

        def exploding_ocr(image):
            raise RuntimeError("OCR crashed")

        with patch("src.capture.triggers.perform_ocr", side_effect=exploding_ocr):
            frame_id = await process_frame(
                image=_make_image(),
                app_name="Finder",
                window_name="Desktop",
                trigger="app_switch",
                db=db,
                writer=writer,
                agents=[agent],
                providers=[],
            )

        assert agent.call_count == 1
        events = await db.get_events_for_frame(frame_id)
        assert len(events) == 1


@pytest.mark.asyncio
class TestCaptureLoopWithBlockingIO:
    """Tests using thread-blocking I/O to simulate realistic agent behavior.

    The Gemma agent does synchronous HTTP via urllib in a thread. The Gemini
    agent uses an async SDK. Both patterns need to be timeout-safe.
    """

    async def test_blocking_thread_agent_gets_timed_out(self, db, tmp_settings):
        """An agent that blocks a thread (like GemmaAgent's urllib call)
        must be cancelled by the capture_timeout_s."""
        tmp_settings.min_capture_interval_ms = 0
        tmp_settings.poll_interval_ms = 50
        tmp_settings.capture_timeout_s = 0.5
        tmp_settings.visual_check_interval_ms = 999_999

        blocking = BlockingThreadAgent(delay_s=30)
        queue: asyncio.Queue[CaptureTrigger] = asyncio.Queue()
        loop = _make_capture_loop(tmp_settings, db, [blocking], queue)

        with _patch_capture():
            task = asyncio.create_task(loop.run())

            t0 = time.monotonic()
            await queue.put(CaptureTrigger.APP_SWITCH)

            # Wait for timeout + buffer
            for _ in range(40):
                await asyncio.sleep(0.05)
                if loop._processing_task and loop._processing_task.done():
                    break

            elapsed = time.monotonic() - t0
            await _stop_loop(loop, task)

        assert elapsed < 3.0, (
            f"Blocking agent took {elapsed:.1f}s — timeout is 0.5s. "
            f"Thread-blocking calls must be killed by the timeout."
        )

    async def test_loop_accepts_new_trigger_after_blocking_timeout(self, db, tmp_settings):
        """After a thread-blocking agent times out, the loop must accept
        and process a new trigger with a different (fast) agent path."""
        tmp_settings.min_capture_interval_ms = 0
        tmp_settings.poll_interval_ms = 50
        tmp_settings.capture_timeout_s = 0.5
        tmp_settings.visual_check_interval_ms = 999_999

        attempt = 0

        class BlockThenFastAgent(ProcessAgent):
            @property
            def name(self):
                return "block_then_fast"

            async def process(self, image, ocr_text, app_name, window_name):
                nonlocal attempt
                attempt += 1
                if attempt == 1:
                    # Simulate a real blocking HTTP call that hangs
                    await asyncio.to_thread(time.sleep, 30)
                return Event(
                    agent_name="block_then_fast",
                    app_type=AppType.OTHER,
                    summary=f"attempt {attempt}",
                )

        queue: asyncio.Queue[CaptureTrigger] = asyncio.Queue()
        loop = _make_capture_loop(tmp_settings, db, [BlockThenFastAgent()], queue)

        t0 = time.monotonic()

        with _patch_capture():
            task = asyncio.create_task(loop.run())

            # First trigger — blocks and times out
            await queue.put(CaptureTrigger.APP_SWITCH)
            await asyncio.sleep(1.0)

            # Second trigger — should work
            await queue.put(CaptureTrigger.CLICK)
            for _ in range(30):
                await asyncio.sleep(0.05)
                if attempt >= 2:
                    break

            elapsed = time.monotonic() - t0
            await _stop_loop(loop, task)

        assert attempt >= 2, f"Only {attempt} attempts — loop didn't recover after timeout"
        assert elapsed < 4.0, (
            f"Took {elapsed:.1f}s — should be ~1.5s. "
            f"If 30s+, the blocking call wasn't cancelled."
        )
        counts = await db.get_counts()
        assert counts["events"] >= 1, "Second (successful) attempt should have written an event"


@pytest.mark.asyncio
class TestCaptureLoopShutdown:
    """Tests for clean shutdown — the processing task must not leak."""

    async def test_stop_cancels_inflight_processing_task(self, db, tmp_settings):
        """Calling stop() while a slow agent is processing must cancel the
        processing task, not leave it running as an orphan.

        Before the fix, stop() only set _running=False but didn't cancel
        _processing_task. The task would keep running until the event loop
        exited.
        """
        tmp_settings.min_capture_interval_ms = 0
        tmp_settings.poll_interval_ms = 50
        tmp_settings.capture_timeout_s = 30  # high — we want stop() to cancel, not timeout
        tmp_settings.visual_check_interval_ms = 999_999

        slow = SlowAsyncAgent(delay_s=30)
        queue: asyncio.Queue[CaptureTrigger] = asyncio.Queue()
        loop = _make_capture_loop(tmp_settings, db, [slow], queue)

        with _patch_capture():
            task = asyncio.create_task(loop.run())
            await queue.put(CaptureTrigger.APP_SWITCH)

            # Wait for the slow agent to start
            await asyncio.wait_for(slow.started.wait(), timeout=2.0)
            assert loop._processing_task is not None
            assert not loop._processing_task.done()

            # Now stop — this should cancel the processing task
            t0 = time.monotonic()
            await _stop_loop(loop, task)
            elapsed = time.monotonic() - t0

        assert loop._processing_task.done(), (
            "Processing task still running after stop() — leaked task"
        )
        assert elapsed < 3.0, (
            f"stop() took {elapsed:.1f}s — should be instant. "
            f"If ~30s, stop() didn't cancel the processing task."
        )

    async def test_stop_with_no_active_task_is_clean(self, db, tmp_settings):
        """stop() when nothing is processing shouldn't crash."""
        tmp_settings.poll_interval_ms = 50
        tmp_settings.visual_check_interval_ms = 999_999

        loop = _make_capture_loop(tmp_settings, db, [FastAgent()])

        with _patch_capture():
            task = asyncio.create_task(loop.run())
            await asyncio.sleep(0.1)
            await _stop_loop(loop, task)

        # No crash = success


@pytest.mark.asyncio
class TestLastCaptureTimestamp:
    """Tests that _last_capture_time is only updated on success."""

    async def test_failed_capture_doesnt_block_next_attempt(self, db, tmp_settings):
        """If a capture fails, _last_capture_time should NOT be updated,
        so the next trigger isn't delayed by min_capture_interval_ms.

        Before the fix, _last_capture_time was set when create_task fired,
        regardless of whether the capture succeeded. A failed capture would
        then block the next attempt for min_capture_interval_ms.
        """
        tmp_settings.min_capture_interval_ms = 5000  # 5s — would be noticeable if wrong
        tmp_settings.poll_interval_ms = 50
        tmp_settings.capture_timeout_s = 0.3
        tmp_settings.visual_check_interval_ms = 999_999

        call_number = 0

        class TimeoutThenFastAgent(ProcessAgent):
            @property
            def name(self):
                return "timeout_then_fast"

            async def process(self, image, ocr_text, app_name, window_name):
                nonlocal call_number
                call_number += 1
                if call_number == 1:
                    await asyncio.sleep(30)  # will be timed out
                return Event(
                    agent_name="timeout_then_fast",
                    app_type=AppType.OTHER,
                    summary=f"attempt {call_number}",
                )

        queue: asyncio.Queue[CaptureTrigger] = asyncio.Queue()
        loop = _make_capture_loop(tmp_settings, db, [TimeoutThenFastAgent()], queue)

        t0 = time.monotonic()

        with _patch_capture():
            task = asyncio.create_task(loop.run())

            # First trigger — times out (0.3s)
            await queue.put(CaptureTrigger.APP_SWITCH)
            await asyncio.sleep(0.6)

            # Second trigger — should NOT wait 5s for min_capture_interval
            await queue.put(CaptureTrigger.CLICK)
            for _ in range(40):
                await asyncio.sleep(0.05)
                if call_number >= 2:
                    break

            elapsed = time.monotonic() - t0
            await _stop_loop(loop, task)

        assert call_number >= 2, f"Only {call_number} calls"
        assert elapsed < 3.0, (
            f"Took {elapsed:.1f}s — if ~5s+, _last_capture_time was wrongly "
            f"updated on the failed first attempt, delaying the second."
        )
