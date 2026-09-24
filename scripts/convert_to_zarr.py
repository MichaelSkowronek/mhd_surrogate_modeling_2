"""Convert the raw re16k_t400_*.npy simulations into a single zarr store.

Each source file has shape (T, 2, H, W) with a different T per simulation
(same spatial grid), so they are written as separate arrays within one
zarr group rather than stacked into a single array.

Usage:
    uv run scripts/convert_to_zarr.py
    uv run scripts/convert_to_zarr.py --chunk-t 64 --out data/processed/re16k_t400.zarr
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import zarr

from mhd_surrogate.logging_config import add_log_level_arg, setup_logging

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_OUT_PATH = Path("data/processed/re16k_t400.zarr")
DEFAULT_CHUNK_T = 32

CHANNEL_NAMES = ["u_x", "u_y"]
EXCLUDED_FILES = ["re16k_t400_7.npy", "re16k_t400_8.npy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument(
        "--chunk-t",
        type=int,
        default=DEFAULT_CHUNK_T,
        help="Chunk size along the time axis (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing store at --out",
    )
    add_log_level_arg(parser)
    return parser.parse_args()


def convert(data_dir: Path, out_path: Path, chunk_t: int, overwrite: bool) -> None:
    paths = sorted(data_dir.glob("*.npy"))
    if not paths:
        log.warning("No .npy files found in %s. Copy your data there first.", data_dir)
        return

    root = zarr.open_group(store=str(out_path), mode="w" if overwrite else "a")
    root.attrs["description"] = "2D slices of a 3D MHD DNS flow, one array per simulation"
    root.attrs["channel_names"] = CHANNEL_NAMES
    root.attrs["excluded_files"] = EXCLUDED_FILES

    for path in paths:
        src = np.load(path, mmap_mode="r")
        name = path.stem
        n_steps = src.shape[0]

        log.info("%s: %s %s -> %s/%s", name, src.shape, src.dtype, out_path, name)
        arr = root.create_array(
            name=name,
            shape=src.shape,
            dtype=src.dtype,
            chunks=(min(chunk_t, n_steps),) + src.shape[1:],
            overwrite=overwrite,
        )
        for start in range(0, n_steps, chunk_t):
            end = min(start + chunk_t, n_steps)
            arr[start:end] = src[start:end]

        arr.attrs["source_file"] = path.name
        arr.attrs["n_steps"] = n_steps

    log.info("wrote %d simulations to %s", len(paths), out_path)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    convert(args.data_dir, args.out, args.chunk_t, args.overwrite)


if __name__ == "__main__":
    main()
