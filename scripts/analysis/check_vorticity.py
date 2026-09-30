"""Compute vorticity and enstrophy of the 2D velocity field.

The out-of-plane vorticity is w = du_y/dx - du_x/dy, and the enstrophy
proxy is 0.5*w^2 (spatial mean per time step), matching the 0.5*u^2 energy
convention in check_split.py. Only the z-component is available: the other
vorticity components need z-derivatives, which a 2D slice does not contain,
so this is a 2D enstrophy, not the full 3D one.

Assumes the array layout (T, 2, Nx, Ny): axis 2 is x (streamwise), axis 3 is
y, channel 0 is u_x and channel 1 is u_y. Derivatives use second-order
central differences (one-sided at the boundaries) with uniform spacing
taken from the domain lengths in configs/analysis/grid.yaml; --dx/--dy override it.
In y the data was linearly interpolated onto a uniform grid from a non-uniform
DNS grid, so y-derivatives are piecewise-constant approximations. The absolute
values depend on those lengths.

It prints statistics for the mean vorticity and the enstrophy, plots both
over time, and maps the vorticity at the first, middle and last time step --
all over each dataset's full recorded length. Like every
scripts/analysis/*.py script, this never reads re16k_t400_5, the model's
held-out test set (excluded entirely from configs/analysis/split.yaml).

Usage:
    uv run scripts/analysis/check_vorticity.py
    uv run scripts/analysis/check_vorticity.py --dx 1 --dy 1   # grid units
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
import zarr

from mhd_surrogate.analysis.fields import vorticity as compute_vorticity
from mhd_surrogate.analysis.summary import add_common_args, filter_datasets, write_summary
from mhd_surrogate.data.grid import grid_spacing
from mhd_surrogate.utils.logging_config import setup_logging

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/analysis/split.yaml")
DEFAULT_OUT_DIR = Path("reports/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dx", type=float, default=None, help="Override spacing along axis 2")
    parser.add_argument("--dy", type=float, default=None, help="Override spacing along axis 3")
    parser.add_argument(
        "--chunk-t",
        type=int,
        default=32,
        help="Time steps per read (default: %(default)s)",
    )
    add_common_args(parser)
    return parser.parse_args()


def vorticity_stats(
    arr,
    dx: float,
    dy: float,
    chunk_t: int,
    snapshot_steps: list[int],
    n_steps: int | None = None,
):
    """Per-timestep mean vorticity and enstrophy, plus vorticity snapshots.

    Returns (mean_vorticity, enstrophy, snapshots) where the first two have
    shape (T,) and snapshots maps a time step to its (Nx, Ny) vorticity field.

    `n_steps` bounds how much of `arr` is read (default: all of it) -- for
    reading only a leading region of a larger array without loading the
    rest.
    """
    n_steps = arr.shape[0] if n_steps is None else n_steps
    mean_vorticity = np.empty(n_steps, dtype=np.float64)
    enstrophy = np.empty(n_steps, dtype=np.float64)
    snapshots: dict[int, np.ndarray] = {}

    for start in range(0, n_steps, chunk_t):
        end = min(start + chunk_t, n_steps)
        block = arr[start:end].astype(np.float64)
        w = compute_vorticity(block, dx, dy)

        mean_vorticity[start:end] = w.mean(axis=(1, 2))
        enstrophy[start:end] = 0.5 * (w**2).mean(axis=(1, 2))

        for t in snapshot_steps:
            if start <= t < end:
                snapshots[t] = w[t - start]

    return mean_vorticity, enstrophy, snapshots


def print_scalar_stats(label: str, series: np.ndarray) -> dict:
    mean, std = series.mean(), series.std()
    print(f"  {label}: mean={mean:.4g} std={std:.4g}")
    return {"mean": mean, "std": std}


def plot_over_time(name: str, series: dict[str, np.ndarray], out_dir: Path) -> Path:
    fig, axes = plt.subplots(len(series), 1, figsize=(12, 4 * len(series)), squeeze=False)
    for (label, values), ax in zip(series.items(), axes[:, 0]):
        ax.plot(values)
        ax.set_title(f"{name}: {label} over time")
        ax.set_xlabel("time step")

    fig.tight_layout()
    out_path = out_dir / f"{name}_vorticity.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_vorticity_maps(name: str, snapshots: dict[int, np.ndarray], out_dir: Path) -> Path:
    steps = sorted(snapshots)
    limit = max(np.percentile(np.abs(snapshots[t]), 99) for t in steps)

    fig, axes = plt.subplots(len(steps), 1, figsize=(14, 4 * len(steps)), squeeze=False)
    for ax, t in zip(axes[:, 0], steps):
        im = ax.imshow(np.rot90(snapshots[t]), cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(f"{name}: vorticity at t={t}")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    out_path = out_dir / f"{name}_vorticity_maps.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = yaml.safe_load(args.config.read_text())
    root = zarr.open_group(store=config["zarr_store"], mode="r")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = filter_datasets(config["datasets"], args.dataset)
    for name in dataset_names:
        arr = root[name]
        if arr.ndim != 4 or arr.shape[1] != 2:
            log.warning("skipping %s: expected shape (T, 2, Nx, Ny), got %s", name, arr.shape)
            continue

        n_steps = arr.shape[0]
        snapshot_steps = sorted({0, n_steps // 2, n_steps - 1})
        dx, dy = grid_spacing(arr.shape[2], arr.shape[3])
        dx = args.dx if args.dx is not None else dx
        dy = args.dy if args.dy is not None else dy
        mean_vorticity, enstrophy, snapshots = vorticity_stats(
            arr, dx, dy, args.chunk_t, snapshot_steps, n_steps=n_steps
        )

        print(f"\n=== {name} (dx={dx:.5g}, dy={dy:.5g}, {n_steps} steps) ===")
        vorticity_summary = print_scalar_stats("mean vorticity", mean_vorticity)
        enstrophy_summary = print_scalar_stats("enstrophy 0.5*w^2, spatial mean", enstrophy)

        series = {
            "mean vorticity (spatial)": mean_vorticity,
            "enstrophy 0.5*w^2 (spatial mean)": enstrophy,
        }
        print(f"  plot: {plot_over_time(name, series, args.out_dir)}")
        print(f"  maps: {plot_vorticity_maps(name, snapshots, args.out_dir)}")

        summary_path = write_summary(
            name,
            "check_vorticity",
            {
                "dx": dx,
                "dy": dy,
                "mean_vorticity": vorticity_summary,
                "enstrophy": enstrophy_summary,
            },
            args.summary_dir,
        )
        print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
