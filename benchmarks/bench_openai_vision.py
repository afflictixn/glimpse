"""Benchmark OpenAI vision agent latency using the actual ProcessAgent.

Captures a real screenshot (or loads from file), runs OpenAIVisionAgent.process()
at multiple resolutions and detail levels, and reports latency + token usage.

Usage:
    .venv/bin/python -m benchmarks.bench_openai_vision
    .venv/bin/python -m benchmarks.bench_openai_vision --runs 5 --model gpt-5.4-nano
    .venv/bin/python -m benchmarks.bench_openai_vision --from-file /path/to/screenshot.jpg
    .venv/bin/python -m benchmarks.bench_openai_vision --resolutions 960 640 --detail low
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass

from PIL import Image

from benchmarks.utils import Timer, fmt
from src.config import Settings
from src.process.openai_vision_agent import OpenAIVisionAgent
from src.process.vision_shared import LlmTokenCounter

_DEFAULTS = Settings()


@dataclass
class CallResult:
    label: str
    detail: str
    max_width: int
    wall_clock_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    summary: str = ""


async def run_variant(
    image: Image.Image,
    app_name: str,
    model: str,
    max_width: int,
    detail: str,
    timeout: float,
) -> CallResult:
    counter = LlmTokenCounter()
    agent = OpenAIVisionAgent(
        model=model,
        max_image_width=max_width,
        image_detail=detail,
        timeout_s=timeout,
        token_counter=counter,
    )

    t0 = time.perf_counter()
    event = await agent.process(image, "", app_name, None)
    wall = time.perf_counter() - t0

    usage = counter.get(model)
    res_label = "original" if max_width == 0 else str(max_width)
    label = f"{res_label} detail={detail}"

    return CallResult(
        label=label,
        detail=detail,
        max_width=max_width,
        wall_clock_s=wall,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        summary=event.summary[:80] if event else "no event",
    )


def print_variant(v: CallResult, target_s: float) -> None:
    status = "PASS" if v.wall_clock_s < target_s else "SLOW"
    print(f"\n  --- {v.label} [{status}] ---")
    print(f"    wall clock:      {fmt(v.wall_clock_s)}")
    print(f"    tokens:          {v.input_tokens} in + {v.output_tokens} out = {v.input_tokens + v.output_tokens}")
    print(f"    summary:         {v.summary}")


def print_comparison(results: list[CallResult], target_s: float) -> None:
    print(f"\n{'=' * 95}")
    print(f"COMPARISON TABLE  (target: <{target_s:.1f}s)")
    print(f"{'=' * 95}")

    header = f"  {'Variant':>22s}  {'WallClk':>8s}  {'In':>6s}  {'Out':>5s}  {'Total':>6s}  {'Status':>6s}"
    print(header)
    print(f"  {'─' * 62}")

    for v in results:
        total = v.input_tokens + v.output_tokens
        status = "OK" if v.wall_clock_s < target_s else "SLOW"
        print(
            f"  {v.label:>22s}"
            f"  {fmt(v.wall_clock_s):>8s}"
            f"  {v.input_tokens:>6d}"
            f"  {v.output_tokens:>5d}"
            f"  {total:>6d}"
            f"  {status:>6s}"
        )


def print_multi_run_summary(
    all_runs: list[list[CallResult]], target_s: float,
) -> None:
    if not all_runs or not all_runs[0]:
        return

    labels = [v.label for v in all_runs[0]]

    print(f"\n{'=' * 95}")
    print(f"MULTI-RUN SUMMARY  ({len(all_runs)} runs, target: <{target_s:.1f}s)")
    print(f"{'=' * 95}")

    header = f"  {'Variant':>22s}  {'Avg':>8s}  {'Min':>8s}  {'Max':>8s}  {'P50':>8s}  {'AvgTok':>7s}  {'<Target':>7s}"
    print(header)
    print(f"  {'─' * 80}")

    for i, label in enumerate(labels):
        times = sorted(run[i].wall_clock_s for run in all_runs if i < len(run))
        tokens = [run[i].input_tokens + run[i].output_tokens for run in all_runs if i < len(run)]
        if not times:
            continue
        avg = sum(times) / len(times)
        mn, mx = min(times), max(times)
        p50 = times[len(times) // 2]
        avg_tok = sum(tokens) // len(tokens)
        passes = sum(1 for t in times if t < target_s)
        pct = f"{passes}/{len(times)}"
        print(
            f"  {label:>22s}"
            f"  {fmt(avg):>8s}"
            f"  {fmt(mn):>8s}"
            f"  {fmt(mx):>8s}"
            f"  {fmt(p50):>8s}"
            f"  {avg_tok:>7d}"
            f"  {pct:>7s}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark OpenAI vision agent latency")
    p.add_argument("--model", type=str, default=_DEFAULTS.openai_vision_model)
    p.add_argument("--runs", type=int, default=3, help="Number of iterations (default: 3)")
    p.add_argument(
        "--resolutions", type=int, nargs="+", default=[960, 640],
        help="Max widths to test (0 = original). Default: 960 640",
    )
    p.add_argument(
        "--detail", type=str, nargs="+", default=["low", "auto"],
        help="OpenAI image detail levels to test. Default: low auto",
    )
    p.add_argument("--from-file", type=str, default=None, help="Load image from file")
    p.add_argument("--target", type=float, default=2.0, help="Target latency in seconds (default: 2.0)")
    p.add_argument("--timeout", type=float, default=15.0, help="Per-call timeout in seconds (default: 15.0)")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    if args.from_file:
        print(f"Loading image from: {args.from_file}")
        image = Image.open(args.from_file).convert("RGB")
        app_name = "loaded_from_file"
    else:
        from src.capture.screenshot import capture_screen, get_focused_app
        print("Capturing screenshot...")
        with Timer("capture") as tc:
            image = capture_screen()
        app_name, _ = get_focused_app()
        print(f"  Screenshot: {image.width}x{image.height} in {fmt(tc.elapsed)}, app={app_name}")

    total_variants = len(args.resolutions) * len(args.detail)

    print("\nOpenAI Vision Agent Benchmark")
    print(f"  Model:       {args.model}")
    print(f"  Image:       {image.width}x{image.height}")
    print(f"  Resolutions: {args.resolutions}")
    print(f"  Detail:      {args.detail}")
    print(f"  Runs:        {args.runs}")
    print(f"  Target:      <{args.target}s")

    all_runs: list[list[CallResult]] = []

    for run_num in range(1, args.runs + 1):
        print(f"\n{'=' * 65}")
        print(f"Run {run_num}/{args.runs}")
        print(f"{'=' * 65}")

        run_results: list[CallResult] = []
        variant_num = 0

        for max_w in args.resolutions:
            for detail in args.detail:
                variant_num += 1
                res_label = "original" if max_w == 0 else str(max_w)
                print(f"\n  [{variant_num}/{total_variants}] {res_label} detail={detail}...", end="", flush=True)

                try:
                    v = await run_variant(image, app_name, args.model, max_w, detail, args.timeout)
                    print(f" {fmt(v.wall_clock_s)}")
                    print_variant(v, args.target)
                    run_results.append(v)
                except Exception as e:
                    print(f" ERROR: {e}")

        if run_results:
            print_comparison(run_results, args.target)
        all_runs.append(run_results)

    print_multi_run_summary(all_runs, args.target)


if __name__ == "__main__":
    asyncio.run(main())
