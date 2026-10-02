"""Compute per-channel normalization statistics from the training datasets.

Reads only `train_datasets` from the data config (never the val or test
dataset), streams each one in time chunks, prints per-dataset and pooled
mean/std, and writes the pooled statistics as a validated JSON artifact
(`NormalizationStats`) for training to load.

The per-dataset printout is the check that the statistics are a stable
property of the flow: if the means or stds differ much between
realizations, pooling them hides that.

Usage:
    uv run scripts/data/compute_stats.py
    uv run scripts/data/compute_stats.py --out data/processed/normalization_stats.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
import zarr

from mhd_surrogate.data.normalization import (
    DEFAULT_CHUNK_T,
    pool_moments,
    stats_from_moments,
    store_moments,
)
from mhd_surrogate.utils.logging_config import add_log_level_arg, setup_logging
from mhd_surrogate.utils.parallel import add_backend_args

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/data/re16k.yaml")
DEFAULT_OUT_PATH = Path("data/processed/normalization_stats.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--chunk-t", type=int, default=DEFAULT_CHUNK_T)
    add_backend_args(parser)
    add_log_level_arg(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    cfg = yaml.safe_load(args.config.read_text())
    train = cfg["train_datasets"]
    if cfg["test_dataset"] in train or cfg["val_dataset"] in train:
        raise ValueError("train_datasets overlaps the val/test dataset")

    root = zarr.open_group(store=cfg["zarr_store"], mode="r")
    channels = list(root.attrs["channel_names"])

    # One parallel task per dataset; merged below in `train` order, so the
    # result (and the DVC-tracked stats file) is identical on every backend.
    per_dataset = store_moments(
        cfg["zarr_store"], train, args.chunk_t, backend=args.backend, workers=args.workers
    )
    pooled = pool_moments(per_dataset, len(channels))

    print(f"{'dataset':<16}{'steps':>7}  " + "  ".join(f"mean_{c:<5} std_{c:<5}" for c in channels))
    for name, (n, mean, m2) in per_dataset.items():
        std = np.sqrt(m2 / n)
        cols = "  ".join(f"{m:>10.4f} {s:>9.4f}" for m, s in zip(mean, std))
        print(f"{name:<16}{root[name].shape[0]:>7}  {cols}")

    stats = stats_from_moments(pooled, channels, train)
    cols = "  ".join(f"{m:>10.4f} {s:>9.4f}" for m, s in zip(stats.mean, stats.std))
    print(f"{'pooled':<16}{'':>7}  {cols}")

    stats.save(args.out)
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
