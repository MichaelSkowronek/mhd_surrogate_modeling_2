"""Run the full analysis suite across datasets, in parallel, and build a
cross-dataset comparison table from the JSON summaries each check writes.

Each (script, dataset) pair is completely independent -- embarrassingly
parallel -- so this dispatches them as separate `python check_*.py
--dataset X` subprocesses via a thread pool (the threads just block on
`subprocess.run`; the actual numpy/FFT work runs in the child processes, on
separate cores, with full isolation -- no shared matplotlib/zarr state to
worry about). That is enough for this workload (single machine, a few
minutes total); see the README for why a distributed framework like Ray
isn't warranted yet.

Usage:
    uv run scripts/run_all_checks.py
    uv run scripts/run_all_checks.py --dataset re16k_t400_0 --dataset re16k_t400_1
    uv run scripts/run_all_checks.py --script check_split.py --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from mhd_surrogate.summary import DEFAULT_SUMMARY_DIR, filter_datasets

SCRIPTS = [
    "check_split.py",
    "check_divergence.py",
    "check_vorticity.py",
    "check_spectrum.py",
    "check_autocorrelation.py",
]
DEFAULT_MANIFEST = Path("data/processed/splits/split_manifest.json")
SCRIPTS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Limit to this dataset (repeatable); default: all datasets in the manifest",
    )
    parser.add_argument(
        "--script",
        action="append",
        default=None,
        choices=SCRIPTS,
        help="Limit to this check script (repeatable); default: all of them",
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--workers", type=int, default=None, help="Default: os.cpu_count()")
    return parser.parse_args()


def run_one(script: str, dataset: str, manifest: Path, summary_dir: Path) -> dict[str, Any]:
    start = time.monotonic()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / script),
            "--manifest",
            str(manifest),
            "--dataset",
            dataset,
            "--summary-dir",
            str(summary_dir),
        ],
        capture_output=True,
        text=True,
    )
    return {
        "script": script,
        "dataset": dataset,
        "returncode": result.returncode,
        "elapsed": time.monotonic() - start,
        "stderr": result.stderr,
    }


def count_flags(data: Any) -> int:
    """Recursively count `"flagged": true` occurrences in a summary dict."""
    if isinstance(data, dict):
        count = 1 if data.get("flagged") is True else 0
        return count + sum(count_flags(v) for v in data.values())
    if isinstance(data, list):
        return sum(count_flags(v) for v in data)
    return 0


def build_comparison_table(dataset_names: list[str], summary_dir: Path) -> list[dict[str, Any]]:
    """One row per dataset, pulling headline scalars out of each check's
    summary; the full nested detail stays in the JSON files themselves.
    """
    rows = []
    for name in dataset_names:
        row: dict[str, Any] = {"dataset": name}
        summaries = {}
        for script in SCRIPTS:
            path = summary_dir / f"{name}__{Path(script).stem}.json"
            if path.exists():
                summaries[Path(script).stem] = json.loads(path.read_text())

        split = summaries.get("check_split")
        if split:
            row["n_steps"] = split["n_steps"]
            row["train_steps"] = split["train_range"][1] - split["train_range"][0]
            row["test_steps"] = split["test_range"][1] - split["test_range"][0]
            row["split_flags"] = count_flags(split)

        vorticity = summaries.get("check_vorticity")
        if vorticity:
            row["vorticity_flags"] = count_flags(vorticity)

        divergence = summaries.get("check_divergence")
        if divergence:
            row["div_normalized_train"] = round(divergence["train"]["normalized_mean"], 4)
            row["div_normalized_test"] = round(divergence["test"]["normalized_mean"], 4)

        autocorr = summaries.get("check_autocorrelation")
        if autocorr:
            row["ux_train_tau_int"] = round(autocorr["field"]["u_x"]["train"]["tau_int"], 2)
            n_eff = autocorr["field"]["u_x"]["test"]["n_eff"]
            row["ux_test_n_eff"] = round(n_eff, 1) if n_eff is not None else None

        rows.append(row)
    return rows


def _sorted_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns = {key for row in rows for key in row}
    return sorted(columns, key=lambda c: (c != "dataset", c))


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("no summaries found")
        return
    columns = _sorted_columns(rows)
    widths = {c: max(len(c), max(len(str(row.get(c, ""))) for row in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_sorted_columns(rows))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    splits = filter_datasets(manifest["splits"], args.dataset)
    scripts = args.script or SCRIPTS
    dataset_names = list(splits)
    workers = args.workers or os.cpu_count()

    jobs = [(script, name) for script in scripts for name in dataset_names]
    print(
        f"running {len(jobs)} jobs ({len(scripts)} scripts x {len(dataset_names)} datasets) "
        f"with {workers} workers..."
    )

    results = []
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, script, name, args.manifest, args.summary_dir): (script, name)
            for script, name in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "ok" if result["returncode"] == 0 else "FAILED"
            print(f"  [{status}] {result['script']} {result['dataset']} ({result['elapsed']:.1f}s)")

    elapsed = time.monotonic() - start
    failures = [r for r in results if r["returncode"] != 0]
    print(f"\n{len(results) - len(failures)}/{len(results)} jobs ok in {elapsed:.1f}s")
    for failure in failures:
        print(f"\n--- {failure['script']} {failure['dataset']} stderr ---")
        print(failure["stderr"])

    print("\n=== comparison table ===")
    rows = build_comparison_table(dataset_names, args.summary_dir)
    print_table(rows)
    csv_path = args.summary_dir / "comparison.csv"
    write_csv(rows, csv_path)
    print(f"\nwrote {csv_path}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
