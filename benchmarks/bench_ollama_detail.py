"""Deep-dive benchmark into the Ollama/Gemma call.

Captures a real screenshot, tests multiple resolutions and JPEG qualities,
and saves all inputs/outputs for inspection.

Usage:
    ./venv/bin/python -m benchmarks.bench_ollama_detail
    ./venv/bin/python -m benchmarks.bench_ollama_detail --resolutions 1920 1280 --qualities 70 40
    ./venv/bin/python -m benchmarks.bench_ollama_detail --from-file /path/to/screenshot.jpg
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from benchmarks.utils import Timer, fmt

DEFAULT_RESOLUTIONS = [0, 1920, 1280, 960]  # 0 = original
DEFAULT_QUALITIES = [70, 40]
OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma3:12b"

SYSTEM_PROMPT = """\
You are a screen activity analyzer. Given a screenshot (and optionally OCR text), \
produce a JSON object with these fields:
- "app_type": one of "browser", "terminal", "ide", "other"
- "summary": one-sentence description of what the user is doing
- "metadata": object with any additional observations

Respond ONLY with valid JSON, no markdown fences or extra text."""


@dataclass
class OllamaTimings:
    """All timing fields returned by the Ollama API (in seconds)."""
    total: float = 0.0
    load: float = 0.0
    prompt_eval: float = 0.0
    generation: float = 0.0
    preprocessing: float = 0.0  # total - load - prompt_eval - generation

    prompt_tokens: int = 0
    output_tokens: int = 0

    prompt_tok_per_s: float = 0.0
    output_tok_per_s: float = 0.0

    wall_clock: float = 0.0  # measured externally


@dataclass
class VariantResult:
    label: str
    width: int
    height: int
    jpeg_quality: int
    payload_kb: float
    encode_ms: float
    timings: OllamaTimings
    response_text: str = ""


def encode_image(image: Image.Image, quality: int) -> tuple[str, float]:
    """Encode image to JPEG base64, return (b64_string, elapsed_seconds)."""
    t0 = time.perf_counter()
    buf = io.BytesIO()
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    rgb.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    elapsed = time.perf_counter() - t0
    return b64, elapsed


def resize_image(image: Image.Image, max_width: int) -> Image.Image:
    if max_width <= 0 or image.width <= max_width:
        return image
    ratio = max_width / image.width
    new_h = int(image.height * ratio)
    return image.resize((max_width, new_h), Image.Resampling.LANCZOS)


def call_ollama(
    prompt: str,
    image_b64: str,
    model: str,
    base_url: str,
    timeout: int = 60,
) -> tuple[dict, float]:
    """Call Ollama and return (full_response_dict, wall_clock_seconds)."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.1},
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    wall = time.perf_counter() - t0

    return data, wall


def extract_timings(data: dict, wall_clock: float) -> OllamaTimings:
    total_ns = data.get("total_duration", 0)
    load_ns = data.get("load_duration", 0)
    prompt_eval_ns = data.get("prompt_eval_duration", 0)
    eval_ns = data.get("eval_duration", 0)
    prompt_count = data.get("prompt_eval_count", 0)
    eval_count = data.get("eval_count", 0)

    total_s = total_ns / 1e9
    load_s = load_ns / 1e9
    prompt_eval_s = prompt_eval_ns / 1e9
    gen_s = eval_ns / 1e9
    preprocess_s = max(0.0, total_s - load_s - prompt_eval_s - gen_s)

    return OllamaTimings(
        total=total_s,
        load=load_s,
        prompt_eval=prompt_eval_s,
        generation=gen_s,
        preprocessing=preprocess_s,
        prompt_tokens=prompt_count,
        output_tokens=eval_count,
        prompt_tok_per_s=prompt_count / prompt_eval_s if prompt_eval_s > 0 else 0,
        output_tok_per_s=eval_count / gen_s if gen_s > 0 else 0,
        wall_clock=wall_clock,
    )


def run_variant(
    image: Image.Image,
    max_width: int,
    quality: int,
    app_name: str,
    model: str,
    base_url: str,
) -> VariantResult:
    resized = resize_image(image, max_width)
    b64, encode_time = encode_image(resized, quality)
    payload_kb = len(b64) * 3 / 4 / 1024

    prompt = f"Analyze this screenshot.\nActive app: {app_name}"

    data, wall = call_ollama(prompt, b64, model, base_url)
    timings = extract_timings(data, wall)

    label = f"{resized.width}x{resized.height} q{quality}"

    return VariantResult(
        label=label,
        width=resized.width,
        height=resized.height,
        jpeg_quality=quality,
        payload_kb=payload_kb,
        encode_ms=encode_time * 1000,
        timings=timings,
        response_text=data.get("response", ""),
    )


def print_variant(v: VariantResult) -> None:
    t = v.timings
    print(f"\n  --- {v.label} (payload {v.payload_kb:.0f} KB) ---")
    print(f"    encode:          {v.encode_ms:.0f}ms")
    print(f"    model load:      {fmt(t.load)}")
    print(f"    preprocessing:   {fmt(t.preprocessing)}  (image decode + tokenize)")
    print(f"    prompt eval:     {fmt(t.prompt_eval)}  ({t.prompt_tokens} tokens, {t.prompt_tok_per_s:.0f} tok/s)")
    print(f"    generation:      {fmt(t.generation)}  ({t.output_tokens} tokens, {t.output_tok_per_s:.0f} tok/s)")
    print(f"    server total:    {fmt(t.total)}")
    print(f"    wall clock:      {fmt(t.wall_clock)}")
    print(f"    response:        {v.response_text[:100]}")


def print_comparison_table(results: list[VariantResult]) -> None:
    print(f"\n{'=' * 100}")
    print("COMPARISON TABLE")
    print(f"{'=' * 100}")

    header = f"  {'Variant':>18s}  {'Payload':>8s}  {'Preproc':>8s}  {'PromptEv':>8s}  {'Generate':>8s}  {'Total':>8s}  {'Tokens':>8s}"
    print(header)
    print(f"  {'─' * 94}")

    for v in results:
        t = v.timings
        tok = f"{t.prompt_tokens}+{t.output_tokens}"
        print(
            f"  {v.label:>18s}"
            f"  {v.payload_kb:>6.0f}KB"
            f"  {fmt(t.preprocessing):>8s}"
            f"  {fmt(t.prompt_eval):>8s}"
            f"  {fmt(t.generation):>8s}"
            f"  {fmt(t.total):>8s}"
            f"  {tok:>8s}"
        )


def save_artifacts(
    output_dir: Path,
    original: Image.Image,
    results: list[VariantResult],
    app_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    original.save(str(output_dir / "original.png"))

    for v in results:
        safe_label = v.label.replace(" ", "_")
        resized = resize_image(original, v.width)
        resized.save(
            str(output_dir / f"input_{safe_label}.jpg"),
            "JPEG",
            quality=v.jpeg_quality,
        )

        resp_data = {
            "label": v.label,
            "dimensions": f"{v.width}x{v.height}",
            "jpeg_quality": v.jpeg_quality,
            "payload_kb": round(v.payload_kb, 1),
            "app_name": app_name,
            "timings": {
                "wall_clock": round(v.timings.wall_clock, 3),
                "server_total": round(v.timings.total, 3),
                "model_load": round(v.timings.load, 3),
                "preprocessing": round(v.timings.preprocessing, 3),
                "prompt_eval": round(v.timings.prompt_eval, 3),
                "generation": round(v.timings.generation, 3),
                "prompt_tokens": v.timings.prompt_tokens,
                "output_tokens": v.timings.output_tokens,
                "prompt_tok_per_s": round(v.timings.prompt_tok_per_s, 1),
                "output_tok_per_s": round(v.timings.output_tok_per_s, 1),
            },
            "response": v.response_text,
        }
        with open(output_dir / f"result_{safe_label}.json", "w") as f:
            json.dump(resp_data, f, indent=2)

    print(f"\nArtifacts saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep-dive Ollama/Gemma call benchmark")
    p.add_argument(
        "--resolutions", type=int, nargs="+", default=DEFAULT_RESOLUTIONS,
        help="Max widths to test (0 = original). Default: 0 1920 1280 960",
    )
    p.add_argument(
        "--qualities", type=int, nargs="+", default=DEFAULT_QUALITIES,
        help="JPEG qualities to test. Default: 70 40",
    )
    p.add_argument("--from-file", type=str, default=None, help="Load image from file instead of capturing screen")
    p.add_argument("--ollama-url", type=str, default=OLLAMA_URL)
    p.add_argument("--ollama-model", type=str, default=MODEL)
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save artifacts. Default: benchmarks/output/<timestamp>",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

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

    resolutions = args.resolutions
    qualities = args.qualities
    total_variants = len(resolutions) * len(qualities)

    print(f"\nOllama Detail Benchmark")
    print(f"  Model: {args.ollama_model} @ {args.ollama_url}")
    print(f"  Image: {image.width}x{image.height}")
    print(f"  Resolutions: {resolutions}")
    print(f"  Qualities: {qualities}")
    print(f"  Total variants: {total_variants}")

    results: list[VariantResult] = []
    for i, max_w in enumerate(resolutions):
        for j, q in enumerate(qualities):
            variant_num = i * len(qualities) + j + 1
            res_label = "original" if max_w == 0 else str(max_w)
            print(f"\n[{variant_num}/{total_variants}] {res_label} q{q}...", end="", flush=True)

            v = run_variant(image, max_w, q, app_name, args.ollama_model, args.ollama_url)
            print(f" {fmt(v.timings.total)}")
            print_variant(v)
            results.append(v)

    print_comparison_table(results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("benchmarks/output") / ts
    save_artifacts(output_dir, image, results, app_name)


if __name__ == "__main__":
    main()
