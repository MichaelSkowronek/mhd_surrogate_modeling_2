"""Parallel dispatch of independent work, with a choice of backend.

Two shapes of work, each with the same three backends (`BACKENDS`):

- `run_parallel`: independent `python <script> ...` subprocess jobs (the
  check_*.py suite, video rendering, ...). The real work always happens in
  child processes with full isolation (no shared matplotlib/zarr state);
  the backend only decides what dispatches them: a plain loop
  (`sequential`), a thread pool whose threads block on `subprocess.run`
  (`processes`), or Ray tasks (`ray`).
- `map_tasks`: a picklable Python function over a list of argument tuples
  (preprocessing: one task per dataset), run in a loop, a process pool or
  Ray tasks. Results come back in input order whatever the backend, so a
  caller that merges them in that order gets the same answer on every
  backend.

Ray is an optional dependency (the `ray` extra), imported only when asked
for. See the README's "Parallel backends" section for measured timings and
when each backend is the right choice.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

BACKENDS = ("sequential", "processes", "ray")

# Relative to the working directory, like every other path in this project
# (scripts are always run from the repo root / the container's /app). Locating
# scripts/ from this file's own location only works for an editable install:
# in a regular install the package lives in site-packages, far from scripts/.
SCRIPTS_DIR = Path("scripts").resolve()

log = logging.getLogger(__name__)


def run_subprocess(
    script: str, args: list[str], label: str, scripts_dir: str | Path | None = None
) -> dict[str, Any]:
    """Run `python <scripts_dir>/<script> *args`, returning a result record.

    `script` is a path relative to the scripts directory, e.g.
    "analysis/check_split.py".

    `label` identifies the job in output (e.g. the dataset name it's for).
    `scripts_dir` defaults to `SCRIPTS_DIR`; the Ray backend passes the
    driver's value explicitly, since a worker's own working directory isn't
    guaranteed to be the repo root.
    """
    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, str(Path(scripts_dir or SCRIPTS_DIR) / script), *args],
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


def _log_result(result: dict[str, Any]) -> None:
    status = "ok" if result["returncode"] == 0 else "FAILED"
    log.info("[%s] %s %s (%.1fs)", status, result["script"], result["label"], result["elapsed"])


def _check_backend(backend: str) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}, expected one of {BACKENDS}")


def run_parallel(
    jobs: list[tuple[str, list[str], str]],
    workers: int | None,
    backend: str = "processes",
    memory_gb: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Run (script, args, label) jobs, at most `workers` at a time.

    `memory_gb` maps a script (as it appears in `jobs`) to the memory a job
    of that script needs, in GB; scripts not listed request none. The Ray
    backend schedules only as many jobs at once as fit in the machine's
    memory, whatever `workers` allows, so a heavy script can't pile up and
    exhaust RAM. The other backends have no such notion and run `workers`
    jobs regardless: for them, size `workers` and order `jobs` by hand (put
    heavy jobs apart, not back to back).

    Logs a one-line status per job as it finishes and returns the result
    records in completion order (input order for `sequential`).
    """
    _check_backend(backend)
    if backend == "sequential":
        results = []
        for script, args, label in jobs:
            results.append(run_subprocess(script, args, label))
            _log_result(results[-1])
        return results
    if backend == "ray":
        return _run_parallel_ray(jobs, workers, memory_gb or {})

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_subprocess, script, args, label): label for script, args, label in jobs
        }
        for future in as_completed(futures):
            results.append(future.result())
            _log_result(results[-1])
    return results


def _init_ray(workers: int | None):
    """Start a local Ray runtime limited to `workers` CPUs; caller shuts it down."""
    # When the driver was started with `uv run`, Ray's uv integration would
    # package the whole working directory (here: the repo, including ~20 GB
    # of data) as the workers' runtime environment. Workers already run in
    # this same virtualenv, so turn it off. Must be set before ray is imported.
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    import ray

    # No dashboard (a server we never look at), quiet driver logging; a small
    # object store, since tasks here return only small result records.
    ray.init(
        num_cpus=workers,
        include_dashboard=False,
        logging_level=logging.WARNING,
        object_store_memory=200 * 1024**2,
    )
    return ray


def _run_parallel_ray(
    jobs: list[tuple[str, list[str], str]], workers: int | None, memory_gb: Mapping[str, float]
) -> list[dict[str, Any]]:
    ray = _init_ray(workers)
    try:
        # `memory` is a logical resource Ray budgets against the node's
        # memory (a scheduling hint, not a hard limit on the process).
        remote = ray.remote(num_cpus=1)(run_subprocess)
        scripts_dir = str(SCRIPTS_DIR)
        pending = []
        for script, args, label in jobs:
            options = {"memory": int(memory_gb[script] * 1024**3)} if script in memory_gb else {}
            pending.append(remote.options(**options).remote(script, args, label, scripts_dir))
        results = []
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            results.append(ray.get(done[0]))
            _log_result(results[-1])
        return results
    finally:
        ray.shutdown()


def map_tasks(
    fn: Callable[..., Any],
    arg_tuples: Sequence[tuple[Any, ...]],
    backend: str = "processes",
    workers: int | None = None,
) -> list[Any]:
    """`[fn(*args) for args in arg_tuples]`, run on the chosen backend.

    Results are returned in input order regardless of backend or completion
    order. `fn` must be a picklable top-level function (importable by
    workers), and its arguments and results must be picklable too.
    """
    _check_backend(backend)
    if backend == "sequential":
        return [fn(*args) for args in arg_tuples]
    if backend == "processes":
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fn, *args) for args in arg_tuples]
            return [f.result() for f in futures]

    ray = _init_ray(workers)
    try:
        remote = ray.remote(num_cpus=1)(fn)
        return ray.get([remote.remote(*args) for args in arg_tuples])
    finally:
        ray.shutdown()


def log_failures(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Log stderr for any failed job; returns the list of failures."""
    failures = [r for r in results if r["returncode"] != 0]
    for failure in failures:
        log.error(
            "%s %s failed, stderr:\n%s", failure["script"], failure["label"], failure["stderr"]
        )
    return failures


def add_backend_args(parser: argparse.ArgumentParser, default: str = "processes") -> None:
    """Add the shared `--backend` / `--workers` flags to an argparse parser."""
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=default,
        help="How to run the parallel work (default: %(default)s)",
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Max parallel workers (default: all CPUs)"
    )
