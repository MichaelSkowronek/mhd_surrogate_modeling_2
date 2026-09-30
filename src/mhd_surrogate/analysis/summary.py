"""Shared helpers for the check_*.py scripts' JSON summaries and --dataset filter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from mhd_surrogate.utils.logging_config import add_log_level_arg

DEFAULT_SUMMARY_DIR = Path("reports/summaries")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """--dataset (repeatable, filters the configured datasets to run on),
    --summary-dir and --log-level, shared by all check_*.py scripts.
    """
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Limit to this dataset (repeatable); default: all configured datasets",
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    add_log_level_arg(parser)


def filter_datasets(datasets: list[str], only: list[str] | None) -> list[str]:
    """Filter `datasets` to `only`, preserving `datasets`'s original order."""
    if not only:
        return datasets
    missing = set(only) - set(datasets)
    if missing:
        raise ValueError(f"unknown dataset(s): {sorted(missing)}; available: {sorted(datasets)}")
    return [name for name in datasets if name in only]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
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
