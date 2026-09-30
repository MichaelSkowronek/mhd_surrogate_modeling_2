"""Check incompressibility of the 2D velocity field via its divergence.

For an incompressible flow, div(u) = du_x/dx + du_y/dy should vanish. For a
2D slice of a 3D flow it only vanishes if du_z/dz is negligible, which is
the quasi-2D hypothesis behind this project, so a non-zero divergence here is
informative rather than necessarily a data error.

Assumes the array layout (T, 2, Nx, Ny): axis 2 is x (streamwise), axis 3 is
y, channel 0 is u_x and channel 1 is u_y. Derivatives use second-order
central differences (one-sided at the boundaries) with uniform spacing
taken from the domain lengths in configs/analysis/grid.yaml; --dx/--dy override it.
In y the data was linearly interpolated onto a uniform grid from a non-uniform
DNS grid, so y-derivatives are piecewise-constant approximations.

Per time step it reports the RMS divergence and that RMS normalized by the
RMS of the two derivative terms (a scale-free measure: ~0 for a
divergence-free field, order 1 when the terms do not cancel), over the
train+val region only -- like every scripts/analysis/*.py script, this never
reads the held-out test region. It plots both over time, plus divergence
maps at the first, middle and last time step.

Usage:
    uv run scripts/analysis/check_divergence.py
    uv run scripts/analysis/check_divergence.py --dx 1 --dy 1   # grid units
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

from mhd_surrogate.analysis.summary import add_common_args, filter_datasets, write_summary
from mhd_surrogate.data.grid import grid_spacing
from mhd_surrogate.utils.logging_config import setup_logging

log = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path("data/processed/splits/split_manifest.json")
DEFAULT_OUT_DIR = Path("reports/figures")


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
    add_common_args(parser)
    return parser.parse_args()


def divergence_stats(
    arr,
    dx: float,
    dy: float,
    chunk_t: int,
    snapshot_steps: list[int],
    n_steps: int | None = None,
):
    """Per-timestep RMS divergence and normalized RMS, plus divergence snapshots.

    Returns (rms_div, rel_div, snapshots) where the first two have shape (T,)
    and snapshots maps a time step to its (Nx, Ny) divergence field.

    `n_steps` bounds how much of `arr` is read (default: all of it) -- for
    reading only a leading region of a larger array without loading the
    rest.
    """
    n_steps = arr.shape[0] if n_steps is None else n_steps
    rms_div = np.empty(n_steps, dtype=np.float64)
    rel_div = np.empty(n_steps, dtype=np.float64)
    snapshots: dict[int, np.ndarray] = {}

    for start in range(0, n_steps, chunk_t):
        end = min(start + chunk_t, n_steps)
        block = arr[start:end].astype(np.float64)
        dux_dx = np.gradient(block[:, 0], dx, axis=1)
        duy_dy = np.gradient(block[:, 1], dy, axis=2)
        div = dux_dx + duy_dy

        rms_div[start:end] = np.sqrt((div**2).mean(axis=(1, 2)))
        term_scale = np.sqrt((dux_dx**2 + duy_dy**2).mean(axis=(1, 2)))
        rel_div[start:end] = rms_div[start:end] / (term_scale + 1e-12)

        for t in snapshot_steps:
            if start <= t < end:
                snapshots[t] = div[t - start]

    return rms_div, rel_div, snapshots


def plot_divergence_over_time(
    name: str,
    rms_div: np.ndarray,
    rel_div: np.ndarray,
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), squeeze=False)
    panels = [
        ("RMS divergence", rms_div),
        ("RMS divergence / RMS of derivative terms", rel_div),
    ]
    for (label, values), ax in zip(panels, axes[:, 0]):
        ax.plot(values)
        ax.set_title(f"{name}: {label} over time (train+val)")
        ax.set_xlabel("time step")

    fig.tight_layout()
    out_path = out_dir / f"{name}_divergence.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_divergence_maps(name: str, snapshots: dict[int, np.ndarray], out_dir: Path) -> Path:
    steps = sorted(snapshots)
    limit = max(np.percentile(np.abs(snapshots[t]), 99) for t in steps)

    fig, axes = plt.subplots(len(steps), 1, figsize=(14, 4 * len(steps)), squeeze=False)
    for ax, t in zip(axes[:, 0], steps):
        im = ax.imshow(np.rot90(snapshots[t]), cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_title(f"{name}: divergence at t={t}")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    out_path = out_dir / f"{name}_divergence_maps.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    manifest = json.loads(args.manifest.read_text())
    root = zarr.open_group(store=manifest["config"]["zarr_store"], mode="r")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = filter_datasets(manifest["splits"], args.dataset)
    for name, split in splits.items():
        arr = root[name]
        if arr.ndim != 4 or arr.shape[1] != 2:
            log.warning("skipping %s: expected shape (T, 2, Nx, Ny), got %s", name, arr.shape)
            continue

        train_end = split["trainval"][1]
        snapshot_steps = sorted({0, train_end // 2, train_end - 1})
        dx, dy = grid_spacing(arr.shape[2], arr.shape[3])
        dx = args.dx if args.dx is not None else dx
        dy = args.dy if args.dy is not None else dy
        rms_div, rel_div, snapshots = divergence_stats(
            arr, dx, dy, args.chunk_t, snapshot_steps, n_steps=train_end
        )

        print(f"\n=== {name} (dx={dx:.5g}, dy={dy:.5g}, train+val: [0,{train_end})) ===")
        print(
            f"  RMS divergence mean={rms_div.mean():.4g} max={rms_div.max():.4g} | "
            f"normalized mean={rel_div.mean():.4g} max={rel_div.max():.4g}"
        )
        region_summary = {
            "rms_mean": rms_div.mean(),
            "rms_max": rms_div.max(),
            "normalized_mean": rel_div.mean(),
            "normalized_max": rel_div.max(),
        }
        plot_path = plot_divergence_over_time(name, rms_div, rel_div, args.out_dir)
        print(f"  plot: {plot_path}")
        print(f"  maps: {plot_divergence_maps(name, snapshots, args.out_dir)}")

        summary_path = write_summary(
            name,
            "check_divergence",
            {"dx": dx, "dy": dy, **region_summary},
            args.summary_dir,
        )
        print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
