"""Compute vorticity and enstrophy of the 2D velocity field.

The out-of-plane vorticity is w = du_y/dx - du_x/dy, and the enstrophy
proxy is 0.5*w^2 (spatial mean per time step), matching the 0.5*u^2 energy
convention in check_split.py. Only the z-component is available: the other
vorticity components need z-derivatives, which a 2D slice does not contain,
so this is a 2D enstrophy, not the full 3D one.

Assumes the array layout (T, 2, Nx, Ny): axis 2 is x (streamwise), axis 3 is
y, channel 0 is u_x and channel 1 is u_y. Derivatives use second-order
central differences (one-sided at the boundaries) with uniform spacing
taken from the domain lengths in configs/grid.yaml; --dx/--dy override it.
In y the data was linearly interpolated onto a uniform grid from a non-uniform
DNS grid, so y-derivatives are piecewise-constant approximations. The absolute
values depend on those lengths.

It prints train vs. test statistics for the mean vorticity and the
enstrophy, plots both over time with the train/test boundary marked, and
maps the vorticity at the first, middle and last time step.

Usage:
    uv run scripts/check_vorticity.py
    uv run scripts/check_vorticity.py --dx 1 --dy 1   # grid units
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

from mhd_surrogate.fields import vorticity as compute_vorticity
from mhd_surrogate.grid import grid_spacing

DEFAULT_MANIFEST = Path("data/processed/splits/split_manifest.json")
DEFAULT_OUT_DIR = Path("reports/figures")
STD_DIFF_WARN_THRESHOLD = 0.2  # flag mean shifts larger than this many std devs
REL_DIFF_WARN_THRESHOLD = 0.1  # flag std ratio changes larger than this fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dx", type=float, default=None, help="Override spacing along axis 2")
    parser.add_argument("--dy", type=float, default=None, help="Override spacing along axis 3")
    parser.add_argument(
        "--chunk-t",
        type=int,
        default=32,
        help="Time steps per read (default: %(default)s)",
    )
    return parser.parse_args()


def vorticity_stats(arr, dx: float, dy: float, chunk_t: int, snapshot_steps: list[int]):
    """Per-timestep mean vorticity and enstrophy, plus vorticity snapshots.

    Returns (mean_vorticity, enstrophy, snapshots) where the first two have
    shape (T,) and snapshots maps a time step to its (Nx, Ny) vorticity field.
    """
    n_steps = arr.shape[0]
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


def print_scalar_comparison(
    label: str, series: np.ndarray, train_end: int, test_start: int
) -> None:
    train_vals, test_vals = series[:train_end], series[test_start:]
    train_mean, test_mean = train_vals.mean(), test_vals.mean()
    train_std, test_std = train_vals.std(), test_vals.std()
    mean_diff_in_std = abs(test_mean - train_mean) / (train_std + 1e-8)
    std_rel_diff = abs(test_std - train_std) / (abs(train_std) + 1e-8)
    flag = (
        " <-- check this"
        if mean_diff_in_std > STD_DIFF_WARN_THRESHOLD or std_rel_diff > REL_DIFF_WARN_THRESHOLD
        else ""
    )
    print(
        f"  {label}: train mean={train_mean:.4g} std={train_std:.4g} | "
        f"test mean={test_mean:.4g} std={test_std:.4g} | "
        f"mean shift={mean_diff_in_std:.2f} std devs, std rel diff={std_rel_diff:.1%}{flag}"
    )


def plot_over_time(
    name: str,
    series: dict[str, np.ndarray],
    train_end: int,
    test_start: int,
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(len(series), 1, figsize=(12, 4 * len(series)), squeeze=False)
    for (label, values), ax in zip(series.items(), axes[:, 0]):
        ax.plot(values)
        if test_start > train_end:
            ax.axvspan(train_end, test_start, color="grey", alpha=0.4, label="buffer")
        ax.axvline(test_start, color="red", linestyle="--", label="train/test boundary")
        ax.set_title(f"{name}: {label} over time")
        ax.set_xlabel("time step")
        ax.legend()

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
    manifest = json.loads(args.manifest.read_text())
    root = zarr.open_group(store=manifest["config"]["zarr_store"], mode="r")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, split in manifest["splits"].items():
        arr = root[name]
        if arr.ndim != 4 or arr.shape[1] != 2:
            print(f"skipping {name}: expected shape (T, 2, Nx, Ny), got {arr.shape}")
            continue

        n_steps, train_end, test_start = (
            split["n_steps"],
            split["train"][1],
            split["test"][0],
        )
        snapshot_steps = sorted({0, n_steps // 2, n_steps - 1})
        dx, dy = grid_spacing(arr.shape[2], arr.shape[3])
        dx = args.dx if args.dx is not None else dx
        dy = args.dy if args.dy is not None else dy
        mean_vorticity, enstrophy, snapshots = vorticity_stats(
            arr, dx, dy, args.chunk_t, snapshot_steps
        )

        print(f"\n=== {name} (dx={dx:.5g}, dy={dy:.5g}) ===")
        print_scalar_comparison("mean vorticity", mean_vorticity, train_end, test_start)
        print_scalar_comparison("enstrophy 0.5*w^2, spatial mean", enstrophy, train_end, test_start)

        series = {
            "mean vorticity (spatial)": mean_vorticity,
            "enstrophy 0.5*w^2 (spatial mean)": enstrophy,
        }
        print(f"  plot: {plot_over_time(name, series, train_end, test_start, args.out_dir)}")
        print(f"  maps: {plot_vorticity_maps(name, snapshots, args.out_dir)}")


if __name__ == "__main__":
    main()
