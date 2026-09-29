"""Temporal power spectrum of the spatially averaged velocity field and its
kinetic energy.

Complements check_point_spectrum.py's per-point view (does the dominant
period depend on where you look?) with the aggregate one: the spatial mean
of u_x, u_y, and the per-channel kinetic energy proxy 0.5*u^2 (matching
check_split.py's per-timestep spatial means/energy) at each time step, over
the train+val region only -- like every scripts/analysis/*.py script, this
never reads the held-out test region.

Two estimators are computed and plotted together, same as
check_point_spectrum.py: a plain periodogram and Welch's method (see
mhd_surrogate.analysis.spectral's docstring for why both).

The energy series in particular tend to show a monotonically rising
spectrum with no isolated peak, rather than an oscillation like u_y's --
dominant_periods's top "period" for those is then just the edge of the
resolvable range, not a real spectral line (see its docstring's caveat).
Check the plot, not just the printed/summary numbers.

Usage:
    uv run scripts/analysis/check_spatial_mean_spectrum.py
    uv run scripts/analysis/check_spatial_mean_spectrum.py --dataset re16k_t400_0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

from mhd_surrogate.analysis.spectral import (
    dominant_periods,
    format_peaks,
    power_spectrum,
    welch_spectrum,
)
from mhd_surrogate.analysis.summary import add_common_args, filter_datasets, write_summary
from mhd_surrogate.utils.logging_config import setup_logging

log = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path("data/processed/splits/split_manifest.json")
DEFAULT_OUT_DIR = Path("reports/figures")
CHANNEL_NAMES = ["u_x", "u_y"]
N_PEAKS = 3
# See check_point_spectrum.py's NPERSEG comment for the reasoning.
NPERSEG = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--chunk-t",
        type=int,
        default=32,
        help="Time steps per read (default: %(default)s)",
    )
    parser.add_argument(
        "--n-peaks",
        type=int,
        default=N_PEAKS,
        help="Top spectral peaks to report per series (default: %(default)s)",
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=NPERSEG,
        help="Welch segment length, in steps (default: %(default)s)",
    )
    parser.add_argument(
        "--noverlap",
        type=int,
        default=None,
        help="Welch segment overlap, in steps (default: nperseg // 2)",
    )
    add_common_args(parser)
    return parser.parse_args()


def spatial_series(arr, chunk_t: int, n_steps: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Spatial mean of each channel, and of the per-channel kinetic energy
    proxy 0.5*u^2 (matching check_split.py's convention), per time step.

    Returns (means, energy), each shape (T, C). `n_steps` bounds how much of
    `arr` is read (default: all of it) -- for reading only a leading region
    of a larger array without loading the rest.
    """
    n_steps = arr.shape[0] if n_steps is None else n_steps
    n_channels = arr.shape[1]
    means = np.empty((n_steps, n_channels), dtype=np.float64)
    energy = np.empty((n_steps, n_channels), dtype=np.float64)
    for start in range(0, n_steps, chunk_t):
        end = min(start + chunk_t, n_steps)
        block = arr[start:end]
        means[start:end] = block.mean(axis=(2, 3))
        energy[start:end] = 0.5 * (block.astype(np.float64) ** 2).mean(axis=(2, 3))
    return means, energy


def analyze(series: np.ndarray, nperseg: int, noverlap: int, n_peaks: int) -> dict:
    """Periodogram + Welch spectra and their top dominant periods for a 1D
    time series, plus its mean/std.
    """
    periodogram = power_spectrum(series)
    welch = welch_spectrum(series, nperseg, noverlap)
    return {
        "mean": series.mean(),
        "std": series.std(),
        "periodogram": periodogram,
        "welch": welch,
        "periodogram_top_periods": dominant_periods(*periodogram, n_peaks),
        "welch_top_periods": dominant_periods(*welch, n_peaks),
    }


def plot_spectrum(name: str, groups: dict[str, dict[str, dict]], out_dir: Path) -> Path:
    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 5 * len(groups)), squeeze=False)
    for ax, (title, spectra) in zip(axes[:, 0], groups.items()):
        for c, (label, s) in enumerate(spectra.items()):
            omega, power = s["periodogram"]
            ax.loglog(2 * np.pi / omega[1:], power[1:], color=colors[c], alpha=0.3, linewidth=1)
            omega, power = s["welch"]
            ax.loglog(2 * np.pi / omega[1:], power[1:], color=colors[c], label=f"{label} (Welch)")
        ax.set_title(f"{name}: {title}, temporal spectrum (train+val)")
        ax.set_xlabel("period T (snapshot steps)")
        ax.set_ylabel("power spectral density")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / f"{name}_spatial_mean_spectrum.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    noverlap = args.noverlap if args.noverlap is not None else args.nperseg // 2
    manifest = json.loads(args.manifest.read_text())
    root = zarr.open_group(store=manifest["config"]["zarr_store"], mode="r")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = filter_datasets(manifest["splits"], args.dataset)
    for name, split in splits.items():
        arr = root[name]
        if arr.ndim != 4 or arr.shape[1] != len(CHANNEL_NAMES):
            log.warning("skipping %s: expected shape (T, 2, Nx, Ny), got %s", name, arr.shape)
            continue

        train_end = split["train"][1]
        means, energy = spatial_series(arr, args.chunk_t, n_steps=train_end)
        series = {
            "u_x": means[:, 0],
            "u_y": means[:, 1],
            "energy_x": energy[:, 0],
            "energy_y": energy[:, 1],
            "energy_total": energy.sum(axis=1),
        }
        print(f"\n=== {name} (train+val: [0,{train_end})) ===")

        results = {
            label: analyze(s, args.nperseg, noverlap, args.n_peaks) for label, s in series.items()
        }
        for label, r in results.items():
            print(
                f"  {label}: mean={r['mean']:.4g} std={r['std']:.4g}\n"
                f"    periodogram top periods: {format_peaks(r['periodogram_top_periods'])}\n"
                f"    Welch top periods:       {format_peaks(r['welch_top_periods'])}"
            )

        groups = {
            "velocity": {label: results[label] for label in ("u_x", "u_y")},
            "kinetic energy 0.5*u^2": {
                label: results[label] for label in ("energy_x", "energy_y", "energy_total")
            },
        }
        plot_path = plot_spectrum(name, groups, args.out_dir)
        print(f"  plot: {plot_path}")

        summary = {
            label: {
                "mean": r["mean"],
                "std": r["std"],
                "periodogram_top_periods": r["periodogram_top_periods"],
                "welch_top_periods": r["welch_top_periods"],
            }
            for label, r in results.items()
        }
        summary_path = write_summary(
            name, "check_spatial_mean_spectrum", {"series": summary}, args.summary_dir
        )
        print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
