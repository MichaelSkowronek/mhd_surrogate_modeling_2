"""Checks and provenance for DVC-versioned data.

`dvc.yaml` pins the raw data, the zarr store and the normalization stats
(`dvc.lock`), but a training run reads whatever is on disk: raw data edited
without re-running `dvc repro` would silently train on a stale store. This
module lets an entry point refuse to run in that state, and reads the pinned
hashes back so the run can record exactly which data it used.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class DataVersionError(RuntimeError):
    """The data on disk doesn't match what `dvc.lock` records (or DVC can't tell)."""


def dvc_status(repo_root: Path | str = ".") -> dict[str, Any]:
    """`dvc status --json` for the repo: `{}` when every stage is up to date,
    else a mapping of stage name to what changed. Takes under a second, since
    DVC caches file hashes between calls.
    """
    try:
        result = subprocess.run(
            ["dvc", "status", "--json"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise DataVersionError(
            "dvc not found, so the data version can't be verified; install it (`uv sync`) or, "
            "where DVC isn't available (the container), set verify_data_version=false"
        ) from e
    if result.returncode != 0:
        raise DataVersionError(f"`dvc status` failed:\n{result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise DataVersionError(f"unparseable `dvc status` output: {result.stdout!r}") from e


def ensure_up_to_date(status: Mapping[str, Any]) -> None:
    """Raise DataVersionError unless `status` (from `dvc_status`) is empty."""
    if not status:
        return
    lines = [f"  {stage}: {json.dumps(changes)}" for stage, changes in status.items()]
    raise DataVersionError(
        "data on disk doesn't match dvc.lock:\n"
        + "\n".join(lines)
        + "\nrun `uv run dvc repro` (and `dvc pull` for raw data) to bring it up to date"
    )


def data_provenance(
    lock_path: Path | str = "dvc.lock", raw_dvc_path: Path | str = "data/raw.dvc"
) -> dict[str, str]:
    """The content hashes `dvc.lock`/`data/raw.dvc` pin for the raw data, the
    zarr store and the normalization stats, keyed `raw_md5`, `zarr_md5` and
    `stats_md5`.
    """
    lock = yaml.safe_load(Path(lock_path).read_text())
    raw = yaml.safe_load(Path(raw_dvc_path).read_text())
    stages = lock["stages"]
    zarr_out = _by_path(stages["convert_to_zarr"]["outs"], "data/processed/re16k_t400.zarr")
    stats_out = _by_path(stages["compute_stats"]["outs"], "data/processed/normalization_stats.json")
    return {
        "raw_md5": raw["outs"][0]["md5"],
        "zarr_md5": zarr_out["md5"],
        "stats_md5": stats_out["md5"],
    }


def _by_path(entries: list[dict[str, Any]], path: str) -> dict[str, Any]:
    for entry in entries:
        if entry["path"] == path:
            return entry
    raise DataVersionError(f"{path} not found in dvc.lock")
