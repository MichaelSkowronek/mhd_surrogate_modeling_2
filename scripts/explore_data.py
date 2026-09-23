"""Inspect raw .npy simulation files: shape, dtype, value ranges, and a preview plot.

Usage:
    uv run scripts/explore_data.py                       # explore all files in data/raw
    uv run scripts/explore_data.py --file re16k_t400_0.npy
    uv run scripts/explore_data.py --no-plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_FIGURE_DIR = Path("reports/figures")


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
        print(f"skipping plot for {name}: ndim={array.ndim}, expected 3 (T,H,W) or 4 (T,C,H,W)")
        return

    n_steps = array.shape[0]
    step_indices = sorted({0, n_steps // 2, n_steps - 1})
    n_channels = array.shape[1] if array.ndim == 4 else 1

    fig, axes = plt.subplots(
        n_channels,
        len(step_indices),
        figsize=(4 * len(step_indices), 4 * n_channels),
        squeeze=False,
    )

    for row in range(n_channels):
        for col, t in enumerate(step_indices):
            frame = array[t, row] if array.ndim == 4 else array[t]
            ax = axes[row][col]
            im = ax.imshow(frame, origin="lower", cmap="viridis")
            ax.set_title(f"t={t}" + (f", channel={row}" if n_channels > 1 else ""))
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

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            file_path = args.data_dir / file_path.name
        paths = [file_path]
    else:
        paths = sorted(args.data_dir.glob("*.npy"))

    if not paths:
        print(f"No .npy files found in {args.data_dir}. Copy your data there first.")
        return

    for path in paths:
        if not path.exists():
            print(f"skipping {path}: file not found")
            continue

        array = np.load(path)
        summarize(array, path.name)

        if not args.no_plot:
            plot_preview(array, path.name, args.out_dir)


if __name__ == "__main__":
    main()
