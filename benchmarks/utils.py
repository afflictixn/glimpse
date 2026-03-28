"""Shared timing and reporting utilities for Glimpse benchmarks."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


def fmt(sec: float) -> str:
    """Format seconds as human-readable duration."""
    if sec < 0.001:
        return f"{sec * 1_000_000:.0f}us"
    if sec < 1.0:
        return f"{sec * 1000:.0f}ms"
    return f"{sec:.2f}s"


@dataclass
class StepTiming:
    name: str
    elapsed: float = 0.0
    detail: str = ""


class Timer:
    """Sync context manager that records perf_counter elapsed time."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._start

    def to_step(self, detail: str = "") -> StepTiming:
        return StepTiming(name=self.name, elapsed=self.elapsed, detail=detail)


@dataclass
class RunResult:
    run_number: int
    steps: list[StepTiming] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(s.elapsed for s in self.steps)

    def step_dict(self) -> dict[str, float]:
        return {s.name: s.elapsed for s in self.steps}


def print_run(result: RunResult) -> None:
    """Print detailed breakdown for a single run."""
    print(f"\n{'=' * 65}")
    print(f"Run {result.run_number}")
    print(f"{'=' * 65}")
    for step in result.steps:
        detail = f"  ({step.detail})" if step.detail else ""
        print(f"  {step.name:30s} {fmt(step.elapsed):>8s}{detail}")
    print(f"  {'─' * 45}")
    print(f"  {'TOTAL':30s} {fmt(result.total):>8s}")


def print_summary(results: list[RunResult]) -> None:
    """Print avg/min/max summary table across all runs."""
    if not results:
        return

    all_step_names: list[str] = []
    seen: set[str] = set()
    for r in results:
        for s in r.steps:
            if s.name not in seen:
                all_step_names.append(s.name)
                seen.add(s.name)

    print(f"\n{'=' * 65}")
    print("SUMMARY")
    print(f"{'=' * 65}")

    for name in all_step_names:
        vals = [r.step_dict().get(name, 0.0) for r in results]
        avg = sum(vals) / len(vals)
        mn, mx = min(vals), max(vals)
        print(f"  {name:30s}  avg={fmt(avg):>8s}  min={fmt(mn):>8s}  max={fmt(mx):>8s}")

    totals = [r.total for r in results]
    avg_t = sum(totals) / len(totals)
    print(f"  {'─' * 58}")
    print(f"  {'TOTAL':30s}  avg={fmt(avg_t):>8s}  min={fmt(min(totals)):>8s}  max={fmt(max(totals)):>8s}")
