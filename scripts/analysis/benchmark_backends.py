"""Time the parallel backends (sequential / processes / ray) on real workloads.

Runs the actual scripts as subprocesses, so what's measured is what a user
experiences, including Ray's runtime startup:

  stats    scripts/data/compute_stats.py     (7 train datasets, ~7 GB read)
  convert  scripts/data/convert_to_zarr.py   (9 raw files -> zarr, ~10 GB)
  suite    scripts/analysis/run_all_checks.py (6 checks x 8 datasets)

Backends are interleaved round-robin across repeats (not all runs of one
backend back to back), after `--warmup` untimed rounds, so the OS page cache
doesn't systematically favor whichever backend happens to run later. Outputs
go to a scratch directory, never the DVC-tracked zarr/stats files.

Usage:
    uv run scripts/analysis/benchmark_backends.py --workload stats --repeats 3
    uv run scripts/analysis/benchmark_backends.py --workload suite --repeats 2 \
        --backend processes --backend ray
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from mhd_surrogate.utils.logging_config import add_log_level_arg, setup_logging
from mhd_surrogate.utils.parallel import BACKENDS

log = logging.getLogger(__name__)

WORKLOADS = ("stats", "convert", "suite")


def workload_command(workload: str, backend: str, workers: int | None, scratch: Path) -> list[str]:
    """The command for one timed run; outputs are redirected into `scratch`."""
    w = ["--workers", str(workers)] if workers else []
    if workload == "stats":
        script = ["scripts/data/compute_stats.py", "--out", str(scratch / "stats.json")]
    elif workload == "convert":
        script = [
            "scripts/data/convert_to_zarr.py",
            "--overwrite",
            "--out",
            str(scratch / "bench.zarr"),
        ]
    elif workload == "suite":
        script = ["scripts/analysis/run_all_checks.py", "--summary-dir", str(scratch / "summaries")]
    else:
        raise ValueError(f"unknown workload {workload!r}")
    return [sys.executable, *script, "--backend", backend, *w]


def time_run(cmd: list[str]) -> float:
    """Wall-clock seconds for one run of `cmd` (which must succeed).

    CPU time is deliberately not reported: pool and Ray workers aren't
    children this process waits on, so `getrusage` would undercount them.
    """
    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.monotonic() - start
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{result.stderr[-2000:]}")
    return wall


def summarize(runs: dict[str, list[float]]) -> list[dict[str, float | str]]:
    return [
        {
            "backend": backend,
            "n": len(walls),
            "wall_median": statistics.median(walls),
            "wall_min": min(walls),
            "wall_max": max(walls),
        }
        for backend, walls in runs.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument(
        "--backend", action="append", choices=BACKENDS, help="Repeatable; default: all three"
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timed runs per backend")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed rounds first")
    parser.add_argument("--workers", type=int, default=None, help="Default: all CPUs")
    parser.add_argument("--out", type=Path, default=None, help="Also write results as JSON")
    add_log_level_arg(parser)
    args = parser.parse_args()
    setup_logging(args.log_level)

    backends = args.backend or list(BACKENDS)
    runs: dict[str, list[float]] = {b: [] for b in backends}
    scratch = Path(tempfile.mkdtemp(prefix="bench_", dir="data/processed"))
    try:
        for round_ in range(-args.warmup, args.repeats):
            for backend in backends:
                cmd = workload_command(args.workload, backend, args.workers, scratch)
                label = "warmup" if round_ < 0 else f"run {round_ + 1}/{args.repeats}"
                log.info("%s %s: %s", args.workload, backend, label)
                wall = time_run(cmd)
                log.info("  %.1fs", wall)
                if round_ >= 0:
                    runs[backend].append(wall)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    rows = summarize(runs)
    print(f"\n=== {args.workload}: wall seconds (median / min / max) ===")
    for r in rows:
        print(
            f"{r['backend']:<11} {r['wall_median']:>8.1f} {r['wall_min']:>8.1f} "
            f"{r['wall_max']:>8.1f}   (n={r['n']})"
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "workload": args.workload,
                    "workers": args.workers,
                    "rows": rows,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
