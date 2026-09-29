"""Temporal power spectrum of the velocity field at a few fixed spatial points.

A first, simple probe toward determining the slowest characterizing
frequency of the flow, which will inform how the train/val/test sizes and
the buffer are set (see the "Train+val / test split" and "Temporal
autocorrelation" sections of the README, and CLAUDE.md's "Data analysis"
section). check_autocorrelation.py already found a persistent quasi-periodic
component pooled over the whole domain; this complements it with a
per-point view (does the dominant period depend on where you look?) and,
unlike autocorrelation, reads off periods directly as spectral peaks rather
than inferring them from zero crossings.

For each of a few grid points, computes the FFT of u_x(t) and u_y(t) at that
point over the train+val region only -- like every scripts/analysis/*.py
script, this never reads the held-out test region.

Default points (grid indices into axis 2 = x, axis 3 = y; override with
--point label:x:y, repeatable). All downstream of the inlet transient
(x < ~150, see check_vorticity.py's docstring) -- a near-inlet point was
tried and dropped: watching the video shows that region is either backflow
or laminar, neither of which is useful for characterizing frequencies.
  core_mid=500:63        -- mid-channel (y=63 is the midpoint of the
                             127-point y grid), earlier streamwise station.
  core_downstream=900:63 -- mid-channel, further downstream; found a sharp,
                             consistent ~25-40 step u_y peak across all 9
                             datasets.
  wall_bottom=900:5      -- near the bottom wall (y=5, within the thin
                             boundary layer -- see the Grid/vorticity
                             sections), same x as core_downstream.
  wall_top=900:121       -- near the top wall (y=121, symmetric to
                             wall_bottom's distance from the wall), same x
                             -- for a symmetry check between the two walls.

Usage:
    uv run scripts/analysis/check_point_spectrum.py
    uv run scripts/analysis/check_point_spectrum.py --dataset re16k_t400_0
    uv run scripts/analysis/check_point_spectrum.py --point core:600:63 --point wall:600:5
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
from mhd_surrogate.utils.logging_config import setup_logging

log = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path("data/processed/splits/split_manifest.json")
DEFAULT_OUT_DIR = Path("reports/figures")
CHANNEL_NAMES = ["u_x", "u_y"]
DEFAULT_POINTS = "core_mid:500:63,core_downstream:900:63,wall_bottom:900:5,wall_top:900:121"
N_PEAKS = 3


def parse_points(spec: str) -> dict[str, tuple[int, int]]:
    """Parse "label:x:y,label:x:y,..." into {label: (x, y)}."""
    points = {}
    for item in spec.split(","):
        label, x, y = item.split(":")
        points[label] = (int(x), int(y))
    return points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--point",
        action="append",
        dest="points",
        default=None,
        metavar="LABEL:X:Y",
        help="Grid point to analyze (repeatable); default: the two points above",
    )
    parser.add_argument(
        "--n-peaks",
        type=int,
        default=N_PEAKS,
        help="Top spectral peaks to report per point/channel (default: %(default)s)",
    )
    add_common_args(parser)
    return parser.parse_args()


def power_spectrum(series: np.ndarray, spacing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectrum of a 1D real time series via FFT.

    Detrends (removes the mean) and applies a Hann window to limit spectral
    leakage (the series is a finite, non-periodic excerpt), then computes a
    one-sided periodogram normalized so that integrating `power` over the
    angular frequency axis returns the window-weighted variance of the
    series -- same convention as check_spectrum.py's spatial spectra.

    Returns (omega, power), each shape (n//2+1,): angular frequency in
    radians per snapshot step (the physical time step is not stored in the
    data) and power spectral density.
    """
    n = series.shape[0]
    window = np.hanning(n)
    detrended = series - series.mean()
    transform = np.fft.rfft(detrended * window)
    power = np.abs(transform) ** 2 * spacing / (window**2).sum() / (2 * np.pi)
    last = -1 if n % 2 == 0 else None
    power[1:last] *= 2
    omega = 2 * np.pi * np.fft.rfftfreq(n, d=spacing)
    return omega, power


def dominant_periods(omega: np.ndarray, power: np.ndarray, n_peaks: int) -> list[dict]:
    """Top `n_peaks` local-maximum peaks of `power`, ranked by power, excluding
    the zero-frequency bin (omega[0]).

    A local maximum is a bin whose power exceeds both immediate neighbors;
    the highest-frequency bin (Nyquist) is never counted, since it has no
    right neighbor to compare against. Returns a list of
    {"period", "omega", "power_fraction"} (power_fraction relative to the
    total power excluding the zero-frequency bin), most dominant first.
    """
    total = power[1:].sum()
    is_peak = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
    peak_idx = np.flatnonzero(is_peak) + 1
    ranked = peak_idx[np.argsort(power[peak_idx])[::-1]][:n_peaks]
    return [
        {
            "period": 2 * np.pi / omega[i],
            "omega": omega[i],
            "power_fraction": power[i] / total,
        }
        for i in ranked
    ]


def plot_point_spectra(
    name: str,
    points: dict[str, tuple[int, int]],
    spectra: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(len(points), 1, figsize=(10, 5 * len(points)), squeeze=False)
    for ax, (label, (x, y)) in zip(axes[:, 0], points.items()):
        for cname in CHANNEL_NAMES:
            omega, power = spectra[label][cname]
            periods = 2 * np.pi / omega[1:]
            ax.loglog(periods, power[1:], label=cname)
        ax.set_title(f"{name}: point '{label}' (x={x}, y={y}) temporal spectrum (train+val)")
        ax.set_xlabel("period T (snapshot steps)")
        ax.set_ylabel("power spectral density")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / f"{name}_point_spectrum.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    points = parse_points(",".join(args.points) if args.points else DEFAULT_POINTS)
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
        print(f"\n=== {name} (train+val: [0,{train_end})) ===")

        spectra: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        point_summary: dict[str, dict] = {}
        for label, (x, y) in points.items():
            series = arr[:train_end, :, x, y]  # (train_end, C); a single grid point, tiny read
            print(f"  point '{label}' (x={x}, y={y}):")
            spectra[label] = {}
            channel_summary = {}
            for c, cname in enumerate(CHANNEL_NAMES):
                omega, power = power_spectrum(series[:, c])
                spectra[label][cname] = (omega, power)
                peaks = dominant_periods(omega, power, args.n_peaks)
                peaks_text = ", ".join(
                    f"T={p['period']:.1f} ({p['power_fraction']:.1%})" for p in peaks
                )
                print(
                    f"    {cname}: mean={series[:, c].mean():.4g} std={series[:, c].std():.4g} "
                    f"| top periods: {peaks_text}"
                )
                channel_summary[cname] = {
                    "mean": series[:, c].mean(),
                    "std": series[:, c].std(),
                    "top_periods": peaks,
                }
            point_summary[label] = {"x": x, "y": y, "channels": channel_summary}

        plot_path = plot_point_spectra(name, points, spectra, args.out_dir)
        print(f"  plot: {plot_path}")

        summary_path = write_summary(
            name, "check_point_spectrum", {"points": point_summary}, args.summary_dir
        )
        print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
