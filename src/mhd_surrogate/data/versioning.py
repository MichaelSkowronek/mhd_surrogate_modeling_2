"""Provenance for DVC-versioned data.

`dvc.yaml` pins the raw data, the zarr store and the normalization stats in
`dvc.lock`; this reads those pinned hashes back so a training run can record
which data version it was run against. It reports what `dvc.lock` pins, not
what is on disk -- keeping the two in sync is the job of running the pipeline
through `dvc repro`, not of this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


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
    raise ValueError(f"{path} not found in dvc.lock")
