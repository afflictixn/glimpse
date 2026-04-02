"""Benchmark the full capture-to-event pipeline (mirrors CaptureLoop._do_capture).

Usage:
    .venv/bin/python -m benchmarks.bench_capture_pipeline
    .venv/bin/python -m benchmarks.bench_capture_pipeline --runs 5 --skip-vision
    .venv/bin/python -m benchmarks.bench_capture_pipeline --vision-provider openai
    .venv/bin/python -m benchmarks.bench_capture_pipeline --vision-provider ollama --ollama-model gemma3:12b
    .venv/bin/python -m benchmarks.bench_capture_pipeline --keep-snapshots
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from benchmarks.utils import RunResult, StepTiming, Timer, print_run, print_summary
from src.capture.screenshot import capture_screen, get_focused_app
from src.config import Settings
from src.ocr.apple_vision import perform_ocr
from src.process.browser_content_agent import BrowserContentAgent
from src.process.gemini_vision_agent import GeminiVisionAgent
from src.process.gemma_agent import GemmaAgent
from src.process.openai_vision_agent import OpenAIVisionAgent
from src.process.process_agent import ProcessAgent
from src.storage.database import DatabaseManager
from src.storage.models import Frame
from src.storage.snapshot_writer import SnapshotWriter

_DEFAULTS = Settings()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark the capture-to-event pipeline")
    p.add_argument("--runs", type=int, default=3, help="Number of iterations (default: 3)")
    p.add_argument("--skip-vision", action="store_true", help="Skip the vision agent call")
    p.add_argument("--vision-provider", type=str, default=_DEFAULTS.vision_provider,
                   help="Vision provider: gemini, openai, or ollama (default: %(default)s)")
    p.add_argument("--ollama-url", type=str, default=_DEFAULTS.ollama_base_url)
    p.add_argument("--ollama-model", type=str, default=_DEFAULTS.ollama_model)
    p.add_argument("--gemini-vision-model", type=str, default=_DEFAULTS.gemini_vision_model)
    p.add_argument("--openai-vision-model", type=str, default=_DEFAULTS.openai_vision_model)
    p.add_argument("--openai-image-detail", type=str, default=_DEFAULTS.openai_image_detail)
    p.add_argument("--max-image-width", type=int, default=_DEFAULTS.ollama_max_image_width,
                   help="Max image width (0=no resize)")
    p.add_argument("--keep-snapshots", action="store_true", help="Keep snapshot files after run")
    return p.parse_args()


def create_vision_agent(args: argparse.Namespace) -> ProcessAgent:
    if args.vision_provider == "openai":
        return OpenAIVisionAgent(
            model=args.openai_vision_model,
            max_image_width=args.max_image_width,
            image_detail=args.openai_image_detail,
        )
    elif args.vision_provider == "gemini":
        return GeminiVisionAgent(
            model=args.gemini_vision_model,
            max_image_width=args.max_image_width,
        )
    else:
        return GemmaAgent(
            ollama_base_url=args.ollama_url,
            model=args.ollama_model,
            max_image_width=args.max_image_width,
        )


def describe_vision_agent(args: argparse.Namespace) -> str:
    if args.skip_vision:
        return "SKIP"
    if args.vision_provider == "openai":
        return f"{args.openai_vision_model} (detail={args.openai_image_detail})"
    if args.vision_provider == "gemini":
        return args.gemini_vision_model
    return f"{args.ollama_model} @ {args.ollama_url}"


async def run_once(
    run_num: int,
    db: DatabaseManager,
    writer: SnapshotWriter,
    vision_agent: ProcessAgent | None,
    browser: BrowserContentAgent,
) -> RunResult:
    result = RunResult(run_number=run_num)

    with Timer("capture_screen") as t:
        image = capture_screen()
    result.steps.append(t.to_step(f"{image.width}x{image.height}"))

    with Timer("get_focused_app") as t:
        app_name, window_name = get_focused_app()
    result.steps.append(t.to_step(f"app={app_name}"))

    with Timer("snapshot_save") as t:
        snapshot_path = writer.save(image)
    file_kb = Path(snapshot_path).stat().st_size / 1024
    result.steps.append(t.to_step(f"{file_kb:.0f} KB"))

    with Timer("content_hash") as t:
        content_hash = hashlib.md5(
            image.resize(
                (image.width // 8, image.height // 8), Image.Resampling.NEAREST
            ).tobytes()
        ).hexdigest()
    result.steps.append(t.to_step())

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

    with Timer("ocr") as t:
        ocr_result = perform_ocr(image)
    ocr_chars = len(ocr_result.text)
    result.steps.append(t.to_step(f"{ocr_chars} chars"))

    with Timer("db_insert_ocr") as t:
        await db.insert_ocr(frame_id, ocr_result)
    result.steps.append(t.to_step())

    agents_to_run: list[ProcessAgent] = []
    agent_names: list[str] = []
    if vision_agent is not None:
        agents_to_run.append(vision_agent)
        agent_names.append(vision_agent.name)
    agents_to_run.append(browser)
    agent_names.append(browser.name)

    with Timer("agents_parallel") as t_parallel:
        agent_timings: list[StepTiming] = []

        async def timed_agent(agent: ProcessAgent, name: str):
            with Timer(name) as ta:
                ev = await agent.process(image, ocr_result.text, app_name or None, window_name or None)
            detail = ev.summary[:60] if ev else "no event"
            agent_timings.append(ta.to_step(detail))
            return ev

        agent_results = await asyncio.gather(
            *(timed_agent(a, n) for a, n in zip(agents_to_run, agent_names)),
            return_exceptions=True,
        )

    for at in agent_timings:
        result.steps.append(at)
    result.steps.append(t_parallel.to_step("wall clock"))

    events_inserted = 0
    with Timer("db_insert_events") as t:
        for ev_result in agent_results:
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

    vision_agent: ProcessAgent | None = None
    if not args.skip_vision:
        vision_agent = create_vision_agent(args)

    browser = BrowserContentAgent()

    print(f"Capture Pipeline Benchmark ({args.runs} runs)")
    print(f"  Vision: {describe_vision_agent(args)}")
    print(f"  Max image width: {args.max_image_width or 'no resize'}")
    print(f"  Temp dir: {tmp_dir}")

    results: list[RunResult] = []
    for i in range(1, args.runs + 1):
        r = await run_once(i, db, writer, vision_agent, browser)
        print_run(r)
        results.append(r)

    print_summary(results)

    await db.close()

    if args.keep_snapshots:
        print(f"\nSnapshots kept at: {snapshots_dir}")
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("\nTemp files cleaned up.")


if __name__ == "__main__":
    asyncio.run(main())
