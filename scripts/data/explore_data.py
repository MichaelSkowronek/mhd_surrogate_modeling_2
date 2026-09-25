"""Inspect raw .npy simulation files: shape, dtype, value ranges, and a preview plot.

Usage:
    uv run scripts/data/explore_data.py                       # explore all files in data/raw
    uv run scripts/data/explore_data.py --file re16k_t400_0.npy
    uv run scripts/data/explore_data.py --no-plot
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mhd_surrogate.utils.logging_config import add_log_level_arg, setup_logging

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_FIGURE_DIR = Path("reports/figures")

# Known channel semantics for the re16k_t400_*.npy dataset: (T, 2, H, W)
# with velocity components u_x, u_y along axis 1.
CHANNEL_NAMES = ["u_x", "u_y"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing raw .npy files (default: %(default)s)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Explore a single file by name instead of every .npy file in --data-dir",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory to write preview plots to (default: %(default)s)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating preview plots, only print statistics",
    )
    add_log_level_arg(parser)
    return parser.parse_args()


def summarize(array: np.ndarray, name: str) -> None:
    nan_count = int(np.isnan(array).sum()) if np.issubdtype(array.dtype, np.floating) else 0
    inf_count = int(np.isinf(array).sum()) if np.issubdtype(array.dtype, np.floating) else 0
    size_mb = array.nbytes / (1024**2)

    print(f"\n=== {name} ===")
    print(f"shape:      {array.shape}")
    print(f"dtype:      {array.dtype}")
    print(f"size:       {size_mb:.1f} MB")
    print(f"min / max:  {array.min():.6g} / {array.max():.6g}")
    print(f"mean / std: {array.mean():.6g} / {array.std():.6g}")
    print(f"nan / inf:  {nan_count} / {inf_count}")


def plot_preview(array: np.ndarray, name: str, out_dir: Path) -> None:
    """Best-effort preview: plot the first, middle, and last frame along axis 0.

    Assumes axis 0 is time and, for a 4D array, axis 1 is a small channel
    axis (channel-first, i.e. `(T, C, H, W)`) with axes 2 and 3 the spatial
    grid. A 3D array is treated as a single-channel `(T, H, W)` series.
    """
    if array.ndim not in (3, 4):
        log.warning(
            "skipping plot for %s: ndim=%d, expected 3 (T,H,W) or 4 (T,C,H,W)", name, array.ndim
        )
        return

    n_steps = array.shape[0]
    step_indices = sorted({0, n_steps // 2, n_steps - 1})
    n_channels = array.shape[1] if array.ndim == 4 else 1
    channel_names = (
        CHANNEL_NAMES
        if n_channels == len(CHANNEL_NAMES)
        else [f"channel={i}" for i in range(n_channels)]
    )

    panels = [(t, row) for row in range(n_channels) for t in step_indices]

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(14, 4 * len(panels)),
        squeeze=False,
    )

    for i, (t, row) in enumerate(panels):
        frame = array[t, row] if array.ndim == 4 else array[t]
        ax = axes[i][0]
        im = ax.imshow(np.rot90(frame), cmap="viridis")
        title = f"t={t}"
        if n_channels > 1:
            title += f", {channel_names[row]}"
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(name)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(name).stem}_preview.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"preview plot: {out_path}")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            file_path = args.data_dir / file_path.name
        paths = [file_path]
    else:
        paths = sorted(args.data_dir.glob("*.npy"))

    if not paths:
        log.warning("No .npy files found in %s. Copy your data there first.", args.data_dir)
        return

    for path in paths:
        if not path.exists():
            log.warning("skipping %s: file not found", path)
            continue

        array = np.load(path)
        summarize(array, path.name)

        if not args.no_plot:
            plot_preview(array, path.name, args.out_dir)


if __name__ == "__main__":
    main()
