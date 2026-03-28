"""Benchmark the full capture-to-event pipeline (mirrors CaptureLoop._do_capture).

Usage:
    ./venv/bin/python -m benchmarks.bench_capture_pipeline
    ./venv/bin/python -m benchmarks.bench_capture_pipeline --runs 5 --skip-gemma
    ./venv/bin/python -m benchmarks.bench_capture_pipeline --keep-snapshots
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from benchmarks.utils import RunResult, StepTiming, Timer, fmt, print_run, print_summary
from src.capture.screenshot import capture_screen, get_focused_app
from src.config import Settings
from src.ocr.apple_vision import perform_ocr
from src.process.browser_content_agent import BrowserContentAgent
from src.process.gemma_agent import GemmaAgent
from src.process.process_agent import NoOpAgent
from src.storage.database import DatabaseManager
from src.storage.models import Frame
from src.storage.snapshot_writer import SnapshotWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark the capture-to-event pipeline")
    p.add_argument("--runs", type=int, default=3, help="Number of iterations (default: 3)")
    p.add_argument("--skip-gemma", action="store_true", help="Skip the Ollama/Gemma call")
    p.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    p.add_argument("--ollama-model", type=str, default="gemma3:12b")
    p.add_argument("--keep-snapshots", action="store_true", help="Keep snapshot files after run")
    return p.parse_args()


async def run_once(
    run_num: int,
    db: DatabaseManager,
    writer: SnapshotWriter,
    gemma: GemmaAgent | None,
    browser: BrowserContentAgent,
) -> RunResult:
    result = RunResult(run_number=run_num)

    # 1. Screenshot
    with Timer("capture_screen") as t:
        image = capture_screen()
    result.steps.append(t.to_step(f"{image.width}x{image.height}"))

    # 2. Focused app
    with Timer("get_focused_app") as t:
        app_name, window_name = get_focused_app()
    result.steps.append(t.to_step(f"app={app_name}"))

    # 3. Save snapshot
    with Timer("snapshot_save") as t:
        snapshot_path = writer.save(image)
    file_kb = Path(snapshot_path).stat().st_size / 1024
    result.steps.append(t.to_step(f"{file_kb:.0f} KB"))

    # 4. Content hash
    with Timer("content_hash") as t:
        content_hash = hashlib.md5(
            image.resize(
                (image.width // 8, image.height // 8), Image.Resampling.NEAREST
            ).tobytes()
        ).hexdigest()
    result.steps.append(t.to_step())

    # 5. DB insert frame
    frame = Frame(
        timestamp=datetime.now(timezone.utc).isoformat(),
        snapshot_path=snapshot_path,
        app_name=app_name or None,
        window_name=window_name or None,
        focused=True,
        capture_trigger="benchmark",
        content_hash=content_hash,
    )
    with Timer("db_insert_frame") as t:
        frame_id = await db.insert_frame(frame)
    result.steps.append(t.to_step())

    # 6. OCR
    with Timer("ocr") as t:
        ocr_result = perform_ocr(image)
    ocr_chars = len(ocr_result.text)
    result.steps.append(t.to_step(f"{ocr_chars} chars"))

    # 7. DB insert OCR
    with Timer("db_insert_ocr") as t:
        await db.insert_ocr(frame_id, ocr_result)
    result.steps.append(t.to_step())

    # 8. Process agents in parallel (mirrors asyncio.gather in _do_capture)
    agents_to_run = []
    agent_names = []
    if gemma is not None:
        agents_to_run.append(gemma)
        agent_names.append("gemma_agent")
    agents_to_run.append(browser)
    agent_names.append("browser_content_agent")

    with Timer("agents_parallel") as t_parallel:
        agent_timings: list[StepTiming] = []
        async def timed_agent(agent, name):
            with Timer(name) as ta:
                ev = await agent.process(image, ocr_result.text, app_name or None, window_name or None)
            detail = ev.summary[:60] if ev else "no event"
            agent_timings.append(ta.to_step(detail))
            return ev

        agent_results = await asyncio.gather(
            *(timed_agent(a, n) for a, n in zip(agents_to_run, agent_names)),
            return_exceptions=True,
        )
    # Add individual agent timings, then the parallel wall clock
    for at in agent_timings:
        result.steps.append(at)
    result.steps.append(t_parallel.to_step("wall clock"))

    # 9. DB insert events
    events_inserted = 0
    with Timer("db_insert_events") as t:
        for i, ev_result in enumerate(agent_results):
            if isinstance(ev_result, BaseException):
                continue
            if ev_result is not None:
                ev_result.frame_id = frame_id
                await db.insert_event(frame_id, ev_result)
                events_inserted += 1
    result.steps.append(t.to_step(f"{events_inserted} events"))

    return result


async def main() -> None:
    args = parse_args()

    # Temp directory for DB and snapshots
    tmp_dir = Path(tempfile.mkdtemp(prefix="glimpse_bench_"))
    snapshots_dir = tmp_dir / "snapshots"
    snapshots_dir.mkdir()

    settings = Settings(
        data_dir=tmp_dir,
        jpeg_quality=80,
    )

    db = DatabaseManager(settings)
    await db.initialize()

    writer = SnapshotWriter(settings)

    gemma: GemmaAgent | None = None
    if not args.skip_gemma:
        gemma = GemmaAgent(
            ollama_base_url=args.ollama_url,
            model=args.ollama_model,
        )

    browser = BrowserContentAgent()

    print(f"Capture Pipeline Benchmark ({args.runs} runs)")
    print(f"  Gemma: {'SKIP' if args.skip_gemma else f'{args.ollama_model} @ {args.ollama_url}'}")
    print(f"  Temp dir: {tmp_dir}")

    results: list[RunResult] = []
    for i in range(1, args.runs + 1):
        r = await run_once(i, db, writer, gemma, browser)
        print_run(r)
        results.append(r)

    print_summary(results)

    await db.close()

    if args.keep_snapshots:
        print(f"\nSnapshots kept at: {snapshots_dir}")
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\nTemp files cleaned up.")


if __name__ == "__main__":
    asyncio.run(main())
