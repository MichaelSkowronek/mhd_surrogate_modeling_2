"""Render a video for every dataset, in parallel.

Each dataset's video is completely independent, so this dispatches them in
parallel via mhd_surrogate.parallel (see its docstring for how, and the
README for why a distributed framework like Ray isn't warranted yet).

Dataset names come from the split manifest (same source as
scripts/run_all_checks.py), not from --store directly, so the set of
datasets stays consistent with the rest of the suite.

Usage:
    uv run scripts/make_all_videos.py
    uv run scripts/make_all_videos.py --field speed --stride 2 --fps 30
    uv run scripts/make_all_videos.py --dataset re16k_t400_0 --workers 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from mhd_surrogate.logging_config import add_log_level_arg, setup_logging
from mhd_surrogate.parallel import log_failures, run_parallel
from mhd_surrogate.summary import filter_datasets

log = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path("data/processed/splits/split_manifest.json")
DEFAULT_STORE = Path("data/processed/re16k_t400.zarr")
DEFAULT_OUT_DIR = Path("reports/videos")
FIELDS = ["vorticity", "u_x", "u_y", "speed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Limit to this dataset (repeatable); default: all datasets in the manifest",
    )
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--field", choices=FIELDS, default="vorticity")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth time step")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--workers", type=int, default=None, help="Default: os.cpu_count()")
    add_log_level_arg(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    manifest = json.loads(args.manifest.read_text())
    splits = filter_datasets(manifest["splits"], args.dataset)
    dataset_names = list(splits)
    workers = args.workers or os.cpu_count()

    jobs = [
        (
            "make_video.py",
            [
                "--dataset",
                name,
                "--store",
                str(args.store),
                "--out-dir",
                str(args.out_dir),
                "--field",
                args.field,
                "--stride",
                str(args.stride),
                "--fps",
                str(args.fps),
            ],
            name,
        )
        for name in dataset_names
    ]
    log.info("rendering %d videos (field=%s) with %d workers...", len(jobs), args.field, workers)

    start = time.monotonic()
    results = run_parallel(jobs, workers)
    elapsed = time.monotonic() - start

    failures = log_failures(results)
    log.info("%d/%d videos ok in %.1fs", len(results) - len(failures), len(results), elapsed)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
