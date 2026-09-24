"""Compute train/test splits from a config file and write a split manifest.

Usage:
    uv run scripts/split_data.py
    uv run scripts/split_data.py --config configs/split.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
import zarr

from mhd_surrogate.splitting import trailing_split

DEFAULT_CONFIG = Path("configs/split.yaml")
DEFAULT_MANIFEST = Path("data/processed/splits/split_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        print(
            f"{name}: n_steps={n_steps} "
            f"train=[0,{split.train_end}) buffer=[{split.train_end},{split.test_start}) "
            f"test=[{split.test_start},{n_steps})"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote manifest to {args.out}")


if __name__ == "__main__":
    main()
