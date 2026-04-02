"""Benchmark GPT-5.4-nano vision latency for the ProcessAgent task.

Captures a real screenshot (or loads from file), sends it to OpenAI's
vision API at multiple resolutions, and reports latency breakdowns.

Usage:
    ./venv/bin/python -m benchmarks.bench_openai_vision
    ./venv/bin/python -m benchmarks.bench_openai_vision --runs 5 --model gpt-5.4-nano
    ./venv/bin/python -m benchmarks.bench_openai_vision --from-file /path/to/screenshot.jpg
    ./venv/bin/python -m benchmarks.bench_openai_vision --resolutions 960 640 --detail low
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI
from PIL import Image

from benchmarks.utils import Timer, fmt

DEFAULT_RESOLUTIONS = [0, 1280, 960, 640]
DEFAULT_MODEL = "gpt-5.4-nano"

_SYSTEM_PROMPT = """\
You are a screen activity analyzer. Given a screenshot (and optionally OCR text), \
produce a JSON object describing the user's current activity. \
Keep summary to one sentence. Include 2-3 key observations in metadata.

Required JSON fields:
- "app_type": one of "browser", "terminal", "ide", "other"
- "summary": one-sentence description
- "metadata": object with additional observations

Respond ONLY with valid JSON."""

_SCHEMA = {"type": "json_object"}


@dataclass
class CallResult:
    label: str
    width: int
    height: int
    jpeg_quality: int
    detail: str
    payload_kb: float
    encode_ms: float
    wall_clock_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    response_text: str = ""


def encode_image(image: Image.Image, max_width: int, quality: int) -> tuple[Image.Image, str, float]:
    """Resize, JPEG-encode, and base64-encode. Returns (resized_image, b64, elapsed_s)."""
    t0 = time.perf_counter()
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    if max_width > 0 and rgb.width > max_width:
        ratio = max_width / rgb.width
        rgb = rgb.resize((max_width, int(rgb.height * ratio)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    elapsed = time.perf_counter() - t0
    return rgb, b64, elapsed


async def call_openai_vision(
    client: AsyncOpenAI,
    model: str,
    image_b64: str,
    prompt: str,
    detail: str,
    timeout: float = 15.0,
) -> tuple[dict, float]:
    """Send image to OpenAI vision API. Returns (response_dict, wall_clock_s)."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": detail,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    t0 = time.perf_counter()
    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=_SCHEMA,
            temperature=0.2,
        ),
        timeout=timeout,
    )
    wall = time.perf_counter() - t0

    choice = response.choices[0]
    usage = response.usage

    return {
        "text": choice.message.content or "",
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }, wall


async def run_variant(
    client: AsyncOpenAI,
    model: str,
    image: Image.Image,
    max_width: int,
    quality: int,
    detail: str,
    app_name: str,
) -> CallResult:
    resized, b64, encode_time = encode_image(image, max_width, quality)
    payload_kb = len(b64) * 3 / 4 / 1024

    prompt = f"Analyze this screenshot.\nActive app: {app_name}"

    resp, wall = await call_openai_vision(client, model, b64, prompt, detail)

    res_label = "original" if max_width == 0 else str(max_width)
    label = f"{resized.width}x{resized.height} q{quality} {detail}"

    return CallResult(
        label=label,
        width=resized.width,
        height=resized.height,
        jpeg_quality=quality,
        detail=detail,
        payload_kb=payload_kb,
        encode_ms=encode_time * 1000,
        wall_clock_s=wall,
        prompt_tokens=resp["prompt_tokens"],
        completion_tokens=resp["completion_tokens"],
        total_tokens=resp["total_tokens"],
        response_text=resp["text"],
    )


def print_variant(v: CallResult) -> None:
    status = "PASS" if v.wall_clock_s < 2.0 else "SLOW"
    print(f"\n  --- {v.label} (payload {v.payload_kb:.0f} KB) [{status}] ---")
    print(f"    encode:          {v.encode_ms:.0f}ms")
    print(f"    wall clock:      {fmt(v.wall_clock_s)}")
    print(f"    tokens:          {v.prompt_tokens} prompt + {v.completion_tokens} completion = {v.total_tokens}")
    print(f"    response:        {v.response_text[:120]}")


def print_comparison(results: list[CallResult], target_s: float = 2.0) -> None:
    print(f"\n{'=' * 105}")
    print(f"COMPARISON TABLE  (target: <{target_s:.1f}s)")
    print(f"{'=' * 105}")

    header = (
        f"  {'Variant':>28s}  {'Payload':>8s}  {'Encode':>8s}  "
        f"{'WallClk':>8s}  {'Tokens':>12s}  {'Status':>6s}"
    )
    print(header)
    print(f"  {'─' * 98}")

    for v in results:
        tok = f"{v.prompt_tokens}+{v.completion_tokens}"
        status = "OK" if v.wall_clock_s < target_s else "SLOW"
        print(
            f"  {v.label:>28s}"
            f"  {v.payload_kb:>6.0f}KB"
            f"  {v.encode_ms:>6.0f}ms"
            f"  {fmt(v.wall_clock_s):>8s}"
            f"  {tok:>12s}"
            f"  {status:>6s}"
        )


def print_multi_run_summary(
    all_runs: list[list[CallResult]], target_s: float = 2.0,
) -> None:
    if not all_runs or not all_runs[0]:
        return

    labels = [v.label for v in all_runs[0]]

    print(f"\n{'=' * 105}")
    print(f"MULTI-RUN SUMMARY  ({len(all_runs)} runs, target: <{target_s:.1f}s)")
    print(f"{'=' * 105}")

    header = f"  {'Variant':>28s}  {'Avg':>8s}  {'Min':>8s}  {'Max':>8s}  {'P50':>8s}  {'<Target':>7s}"
    print(header)
    print(f"  {'─' * 80}")

    for i, label in enumerate(labels):
        times = sorted(run[i].wall_clock_s for run in all_runs if i < len(run))
        if not times:
            continue
        avg = sum(times) / len(times)
        mn, mx = min(times), max(times)
        p50 = times[len(times) // 2]
        passes = sum(1 for t in times if t < target_s)
        pct = f"{passes}/{len(times)}"
        print(
            f"  {label:>28s}"
            f"  {fmt(avg):>8s}"
            f"  {fmt(mn):>8s}"
            f"  {fmt(mx):>8s}"
            f"  {fmt(p50):>8s}"
            f"  {pct:>7s}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark OpenAI vision model latency")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--runs", type=int, default=3, help="Number of iterations (default: 3)")
    p.add_argument(
        "--resolutions", type=int, nargs="+", default=DEFAULT_RESOLUTIONS,
        help="Max widths to test (0 = original). Default: 0 1280 960 640",
    )
    p.add_argument("--quality", type=int, default=70, help="JPEG quality (default: 70)")
    p.add_argument(
        "--detail", type=str, nargs="+", default=["low", "auto"],
        help="OpenAI image detail levels to test. Default: low auto",
    )
    p.add_argument("--from-file", type=str, default=None, help="Load image from file")
    p.add_argument("--target", type=float, default=2.0, help="Target latency in seconds (default: 2.0)")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)

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

    print(f"\nOpenAI Vision Benchmark")
    print(f"  Model:       {args.model}")
    print(f"  Image:       {image.width}x{image.height}")
    print(f"  Resolutions: {args.resolutions}")
    print(f"  Detail:      {args.detail}")
    print(f"  Quality:     {args.quality}")
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
                    v = await run_variant(
                        client, args.model, image, max_w, args.quality, detail, app_name,
                    )
                    print(f" {fmt(v.wall_clock_s)}")
                    print_variant(v)
                    run_results.append(v)
                except Exception as e:
                    print(f" ERROR: {e}")

        if run_results:
            print_comparison(run_results, args.target)
        all_runs.append(run_results)

    print_multi_run_summary(all_runs, args.target)


if __name__ == "__main__":
    asyncio.run(main())
