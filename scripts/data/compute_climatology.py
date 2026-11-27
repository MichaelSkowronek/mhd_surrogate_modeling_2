"""Compute the real flow's climatology of its domain-mean velocity from the
training datasets: the climatological drift band's reference
(`mhd_surrogate.data.climatology`).

Reads only `train_datasets` from the data config (never the val or test
dataset) and the window, stride and block length from the evaluation
config's `drift_band`, prints per-dataset window offsets (how far each
realization's windows stray from the pooled level, in the band's unit) and
writes the pooled `Climatology` as a JSON artifact for training to load.

Usage:
    uv run scripts/data/compute_climatology.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
import zarr

from mhd_surrogate.data.climatology import (
    DEFAULT_CHUNK_T,
    climatology_from_series,
    store_domain_means,
    window_statistics,
)
from mhd_surrogate.utils.logging_config import add_log_level_arg, setup_logging
from mhd_surrogate.utils.parallel import add_backend_args

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/data/re16k.yaml")
DEFAULT_EVALUATION = Path("configs/evaluation/default.yaml")
DEFAULT_OUT_PATH = Path("data/processed/climatology.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--chunk-t", type=int, default=DEFAULT_CHUNK_T)
    add_backend_args(parser)
    add_log_level_arg(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    cfg = yaml.safe_load(args.config.read_text())
    band = yaml.safe_load(args.evaluation.read_text())["drift_band"]
    train = cfg["train_datasets"]
    if cfg["test_dataset"] in train or cfg["val_dataset"] in train:
        raise ValueError("train_datasets overlaps the val/test dataset")

    root = zarr.open_group(store=cfg["zarr_store"], mode="r")
    channels = list(root.attrs["channel_names"])
    # One parallel task per dataset, merged in `train` order, so the
    # DVC-tracked file is identical on every backend.
    series = store_domain_means(
        cfg["zarr_store"], train, args.chunk_t, backend=args.backend, workers=args.workers
    )
    window, stride, block = band["window_steps"], band["stride"], band["block_steps"]
    climatology = climatology_from_series(series, channels, window, stride, block)

    print(f"{'dataset':<16}{'windows':>8}  " + "  ".join(f"{c} offset (signed)" for c in channels))
    level, unit = np.array(climatology.level), np.array(climatology.unit)
    for name, values in series.items():
        means, _, _ = window_statistics(values, window, stride, block)
        signed = (means - level) / unit
        cols = "  ".join(f"{lo:+.2f} to {hi:+.2f}" for lo, hi in zip(signed.min(0), signed.max(0)))
        print(f"{name:<16}{len(means):>8}  {cols}")
    print(f"pooled: {climatology.n_windows} windows; level {climatology.level}")
    print(f"unit {climatology.unit}, slow std {climatology.slow_std}")
    print(
        f"largest window offset {climatology.max_window_offset}, "
        f"std ratios {climatology.std_ratio_range}"
    )

    climatology.save(args.out)
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
