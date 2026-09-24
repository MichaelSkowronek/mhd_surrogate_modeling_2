"""Shared helpers for the check_*.py scripts' JSON summaries and --dataset filter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SUMMARY_DIR = Path("reports/summaries")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """--dataset (repeatable, filters the manifest to run on) and --summary-dir."""
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Limit to this dataset (repeatable); default: all datasets in the manifest",
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)


def filter_datasets(splits: dict[str, Any], only: list[str] | None) -> dict[str, Any]:
    """Filter a manifest's `splits` dict to `only` dataset names, in manifest order."""
    if not only:
        return splits
    missing = set(only) - set(splits)
    if missing:
        raise ValueError(f"unknown dataset(s): {sorted(missing)}; available: {sorted(splits)}")
    return {name: splits[name] for name in splits if name in only}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def write_summary(
    dataset: str, script: str, data: dict[str, Any], out_dir: Path = DEFAULT_SUMMARY_DIR
) -> Path:
    """Write {"dataset", "script", **data} as JSON to <out_dir>/<dataset>__<script>.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset}__{script}.json"
    payload = {"dataset": dataset, "script": script, **data}
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    return path
