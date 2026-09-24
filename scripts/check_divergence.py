"""Check incompressibility of the 2D velocity field via its divergence.

For an incompressible flow, div(u) = du_x/dx + du_y/dy should vanish. For a
2D slice of a 3D flow it only vanishes if du_z/dz is negligible, which is
the quasi-2D hypothesis behind this project, so a non-zero divergence here is
informative rather than necessarily a data error.

Assumes the array layout (T, 2, Nx, Ny): axis 2 is x (streamwise), axis 3 is
y, channel 0 is u_x and channel 1 is u_y. Derivatives use second-order
central differences (one-sided at the boundaries) with uniform spacing
taken from the domain lengths in configs/grid.yaml; --dx/--dy override it.
In y the data was linearly interpolated onto a uniform grid from a non-uniform
DNS grid, so y-derivatives are piecewise-constant approximations.

Per time step it reports the RMS divergence and that RMS normalized by the
RMS of the two derivative terms (a scale-free measure: ~0 for a
divergence-free field, order 1 when the terms do not cancel). It plots both
over time with the train/test boundary marked, plus divergence maps at the
first, middle and last time step.

Usage:
    uv run scripts/check_divergence.py
    uv run scripts/check_divergence.py --dx 1 --dy 1   # grid units
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

from mhd_surrogate.grid import grid_spacing
from mhd_surrogate.logging_config import setup_logging
from mhd_surrogate.summary import add_common_args, filter_datasets, write_summary

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


def divergence_stats(arr, dx: float, dy: float, chunk_t: int, snapshot_steps: list[int]):
    """Per-timestep RMS divergence and normalized RMS, plus divergence snapshots.

    Returns (rms_div, rel_div, snapshots) where the first two have shape (T,)
    and snapshots maps a time step to its (Nx, Ny) divergence field.
    """
    n_steps = arr.shape[0]
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
    train_end: int,
    test_start: int,
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), squeeze=False)
    panels = [
        ("RMS divergence", rms_div),
        ("RMS divergence / RMS of derivative terms", rel_div),
    ]
    for (label, values), ax in zip(panels, axes[:, 0]):
        ax.plot(values)
        if test_start > train_end:
            ax.axvspan(train_end, test_start, color="grey", alpha=0.4, label="buffer")
        ax.axvline(test_start, color="red", linestyle="--", label="train/test boundary")
        ax.set_title(f"{name}: {label} over time")
        ax.set_xlabel("time step")
        ax.legend()

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

        n_steps, train_end, test_start = (
            split["n_steps"],
            split["train"][1],
            split["test"][0],
        )
        snapshot_steps = sorted({0, n_steps // 2, n_steps - 1})
        dx, dy = grid_spacing(arr.shape[2], arr.shape[3])
        dx = args.dx if args.dx is not None else dx
        dy = args.dy if args.dy is not None else dy
        rms_div, rel_div, snapshots = divergence_stats(arr, dx, dy, args.chunk_t, snapshot_steps)

        print(f"\n=== {name} (dx={dx:.5g}, dy={dy:.5g}) ===")
        region_summary = {}
        for region, sl in [
            ("train", slice(0, train_end)),
            ("test", slice(test_start, n_steps)),
        ]:
            print(
                f"  {region}: RMS divergence mean={rms_div[sl].mean():.4g} "
                f"max={rms_div[sl].max():.4g} | "
                f"normalized mean={rel_div[sl].mean():.4g} max={rel_div[sl].max():.4g}"
            )
            region_summary[region] = {
                "rms_mean": rms_div[sl].mean(),
                "rms_max": rms_div[sl].max(),
                "normalized_mean": rel_div[sl].mean(),
                "normalized_max": rel_div[sl].max(),
            }
        plot_path = plot_divergence_over_time(
            name, rms_div, rel_div, train_end, test_start, args.out_dir
        )
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
