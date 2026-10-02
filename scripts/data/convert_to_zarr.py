"""Convert the raw re16k_t400_*.npy simulations into a single zarr store.

Each source file has shape (T, 2, H, W) with a different T per simulation
(same spatial grid), so they are written as separate arrays within one
zarr group rather than stacked into a single array.

Usage:
    uv run scripts/data/convert_to_zarr.py
    uv run scripts/data/convert_to_zarr.py --chunk-t 64 --out data/processed/re16k_t400.zarr
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mhd_surrogate.data.conversion import convert_to_zarr
from mhd_surrogate.utils.logging_config import add_log_level_arg, setup_logging
from mhd_surrogate.utils.parallel import add_backend_args

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
    add_backend_args(parser)
    add_log_level_arg(parser)
    return parser.parse_args()


def convert(
    data_dir: Path, out_path: Path, chunk_t: int, overwrite: bool, backend: str, workers: int | None
) -> None:
    paths = sorted(data_dir.glob("*.npy"))
    if not paths:
        log.warning("No .npy files found in %s. Copy your data there first.", data_dir)
        return

    convert_to_zarr(
        paths,
        out_path,
        chunk_t,
        overwrite,
        description="2D slices of a 3D MHD DNS flow, one array per simulation",
        channel_names=CHANNEL_NAMES,
        excluded_files=EXCLUDED_FILES,
        backend=backend,
        workers=workers,
    )
    log.info("wrote %d simulations to %s", len(paths), out_path)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    convert(args.data_dir, args.out, args.chunk_t, args.overwrite, args.backend, args.workers)


if __name__ == "__main__":
    main()
