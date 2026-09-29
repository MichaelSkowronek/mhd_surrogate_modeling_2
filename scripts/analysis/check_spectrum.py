"""Spatial power spectrum of the 2D velocity field, train+val region.

Computes 1D power spectra E(k) of u_x and u_y along x (axis 2, streamwise)
and along y (axis 3), each averaged over the other spatial axis and over
time, over the train+val region only. Like every scripts/analysis/*.py
script, this never reads the held-out test region.

The domain is not periodic (inlet region, walls in y), so each line has its
mean removed and a Hann window applied before the FFT to limit spectral
leakage. Spectra are one-sided and normalized so that E(k) integrated over
the angular wavenumber k = 2*pi/wavelength equals the window-weighted
variance of the signal. The grid spacing comes from the domain lengths in
configs/analysis/grid.yaml; --dx/--dy override it. Note that y was interpolated onto
a uniform grid from a non-uniform DNS grid (see configs/analysis/grid.yaml), which
smooths small scales in y.

Because the flow is not homogeneous in x (the inlet region differs from the
developed region), the x spectrum is an average over a non-stationary signal.

Usage:
    uv run scripts/analysis/check_spectrum.py
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
CHANNEL_NAMES = ["u_x", "u_y"]


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


def spectrum_sum(block: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    """One-sided spectrum along `axis`, averaged over the other spatial axis
    and summed over time. `block` has shape (t, C, Nx, Ny); returns (C, K).
    """
    n = block.shape[axis]
    shape = [1, 1, 1, 1]
    shape[axis] = n
    window = np.hanning(n).reshape(shape)

    detrended = block - block.mean(axis=axis, keepdims=True)
    transform = np.fft.rfft(detrended * window, axis=axis)
    # Per-cycle PSD, converted to a density per unit angular wavenumber.
    power = np.abs(transform) ** 2 * spacing / (window**2).sum() / (2 * np.pi)
    # One-sided: fold the negative frequencies onto the positive ones.
    last = -1 if n % 2 == 0 else None
    # Move the wavenumber axis last, so the other spatial axis is axis 2.
    power = np.moveaxis(power, axis, -1)
    power[..., 1:last] *= 2
    return power.mean(axis=2).sum(axis=0)


def compute_spectrum(
    arr, train_end: int, dx: float, dy: float, chunk_t: int
) -> dict[str, np.ndarray]:
    """Mean spectrum per direction over the train+val region: {direction: (C, K)}.

    Reads `arr` in chunks bounded by `train_end`; never reads at or past it.
    """
    directions = {"x": (2, dx), "y": (3, dy)}
    sums = {d: 0.0 for d in directions}
    count = 0

    for start in range(0, train_end, chunk_t):
        end = min(start + chunk_t, train_end)
        block = arr[start:end].astype(np.float64)
        for direction, (axis, spacing) in directions.items():
            sums[direction] = sums[direction] + spectrum_sum(block, axis, spacing)
        count += block.shape[0]

    return {d: sums[d] / count for d in directions}


def wavenumbers(n: int, spacing: float) -> np.ndarray:
    return 2 * np.pi * np.fft.rfftfreq(n, d=spacing)


def print_summary(spectrum, k_by_direction: dict[str, np.ndarray]) -> dict:
    summary = {}
    for direction, k in k_by_direction.items():
        dk = k[1] - k[0]
        summary[direction] = {}
        for c, cname in enumerate(CHANNEL_NAMES):
            power = spectrum[direction][c]
            variance = power[1:].sum() * dk
            peak = np.argmax(power[1:]) + 1
            has_peak = peak != 1
            if has_peak:
                peak_text = f"peak k={k[peak]:.3g} (wavelength {2 * np.pi / k[peak]:.3g})"
            else:
                peak_text = "no interior peak (max at lowest k)"
            print(f"  {cname} along {direction}: variance={variance:.4g}, {peak_text}")
            summary[direction][cname] = {
                "variance": variance,
                "peak_k": k[peak] if has_peak else None,
                "peak_wavelength": 2 * np.pi / k[peak] if has_peak else None,
            }
    return summary


def plot_spectrum(
    name: str, spectrum, k_by_direction: dict[str, np.ndarray], out_dir: Path
) -> Path:
    fig, axes = plt.subplots(len(k_by_direction), 1, figsize=(10, 5 * len(k_by_direction)))
    for ax, (direction, k) in zip(np.atleast_1d(axes), k_by_direction.items()):
        for c, cname in enumerate(CHANNEL_NAMES):
            ax.loglog(k[1:], spectrum[direction][c][1:], label=cname)
        ax.set_title(f"{name}: power spectrum along {direction} (train+val)")
        ax.set_xlabel("angular wavenumber k")
        ax.set_ylabel("E(k)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / f"{name}_spectrum.png"
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
        if arr.ndim != 4 or arr.shape[1] != len(CHANNEL_NAMES):
            log.warning("skipping %s: expected shape (T, 2, Nx, Ny), got %s", name, arr.shape)
            continue

        dx, dy = grid_spacing(arr.shape[2], arr.shape[3])
        dx = args.dx if args.dx is not None else dx
        dy = args.dy if args.dy is not None else dy
        train_end = split["train"][1]

        spectrum = compute_spectrum(arr, train_end, dx, dy, args.chunk_t)
        k_by_direction = {
            "x": wavenumbers(arr.shape[2], dx),
            "y": wavenumbers(arr.shape[3], dy),
        }

        print(f"\n=== {name} (dx={dx:.5g}, dy={dy:.5g}, train+val: [0,{train_end})) ===")
        spectrum_summary = print_summary(spectrum, k_by_direction)
        print(f"  plot: {plot_spectrum(name, spectrum, k_by_direction, args.out_dir)}")

        summary_path = write_summary(
            name,
            "check_spectrum",
            {"dx": dx, "dy": dy, **spectrum_summary},
            args.summary_dir,
        )
        print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
