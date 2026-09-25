"""Compute train/test splits from a config file and write a split manifest.

Usage:
    uv run scripts/data/split_data.py
    uv run scripts/data/split_data.py --config configs/analysis/split.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml
import zarr

from mhd_surrogate.data.splitting import trailing_split
from mhd_surrogate.utils.logging_config import add_log_level_arg, setup_logging

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/analysis/split.yaml")
DEFAULT_MANIFEST = Path("data/processed/splits/split_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    add_log_level_arg(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = yaml.safe_load(args.config.read_text())

    root = zarr.open_group(store=config["zarr_store"], mode="r")
    manifest = {"config": config, "splits": {}}

    for name in config["datasets"]:
        n_steps = root[name].shape[0]
        split = trailing_split(n_steps, config["test_fraction"], config["buffer_steps"])
        manifest["splits"][name] = {
            "n_steps": split.n_steps,
            "train": [split.train_start, split.train_end],
            "buffer": [split.train_end, split.test_start],
            "test": [split.test_start, split.test_end],
        }
        log.info(
            "%s: n_steps=%d train=[0,%d) buffer=[%d,%d) test=[%d,%d)",
            name,
            n_steps,
            split.train_end,
            split.train_end,
            split.test_start,
            split.test_start,
            n_steps,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("wrote manifest to %s", args.out)


if __name__ == "__main__":
    main()
