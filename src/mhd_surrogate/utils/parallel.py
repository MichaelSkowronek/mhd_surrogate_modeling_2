"""Generic parallel dispatch of independent subprocess jobs.

Used for embarrassingly parallel work across scripts/datasets (running the
check_*.py suite, rendering videos for every dataset, ...): each job is a
separate `python <script> ...` subprocess, run via a thread pool -- the
threads just block on subprocess.run while the real work happens in the
child processes, on separate cores, with full process isolation (no shared
matplotlib/zarr state to worry about). That's enough for workloads that run
in minutes on one machine; see the README for why a distributed framework
like Ray isn't warranted yet.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# src/mhd_surrogate/utils/parallel.py -> parents[3] is the repo root.
SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"

log = logging.getLogger(__name__)


def run_subprocess(script: str, args: list[str], label: str) -> dict[str, Any]:
    """Run `python <scripts_dir>/<script> *args`, returning a result record.

    `script` is a path relative to the scripts directory, e.g.
    "analysis/check_split.py".

    `label` identifies the job in output (e.g. the dataset name it's for).
    """
    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script), *args],
        capture_output=True,
        text=True,
    )
    return {
        "script": script,
        "label": label,
        "returncode": result.returncode,
        "elapsed": time.monotonic() - start,
        "stderr": result.stderr,
    }


def run_parallel(
    jobs: list[tuple[str, list[str], str]], workers: int | None
) -> list[dict[str, Any]]:
    """Run (script, args, label) jobs concurrently.

    Logs a one-line status per job as it finishes and returns the result
    records in completion order.
    """
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_subprocess, script, args, label): label for script, args, label in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "ok" if result["returncode"] == 0 else "FAILED"
            log.info(
                "[%s] %s %s (%.1fs)", status, result["script"], result["label"], result["elapsed"]
            )
    return results


def log_failures(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Log stderr for any failed job; returns the list of failures."""
    failures = [r for r in results if r["returncode"] != 0]
    for failure in failures:
        log.error(
            "%s %s failed, stderr:\n%s", failure["script"], failure["label"], failure["stderr"]
        )
    return failures
