"""2D spatial power spectrum of the velocity field, averaged over time.

check_spectrum.py already computes 1D spectra E(kx) and E(ky), each
averaged over the *other* spatial axis (and over time); it can't say
whether energy at a given kx and a given ky occur together in the same
structure or in unrelated ones. This instead keeps both wavenumbers: a full
2D FFT per snapshot, then the resulting 2D power spectrum E(kx, ky)
averaged over time, over each dataset's full recorded length -- like every
scripts/analysis/*.py script, this never reads re16k_t400_5, the model's
held-out test set (excluded entirely from configs/analysis/split.yaml).

Each snapshot has the channel's *time-mean field* subtracted (not just its
own instantaneous spatial mean -- see wavenumber_spectrum_2d's docstring
for why that distinction matters here), and the domain is not periodic
(inlet region, walls in y), so a 2D Hann window (the outer product of 1D
Hann windows along each axis) is applied before the FFT to limit spectral
leakage -- same reasoning as check_spectrum.py. The spectrum is one-sided
in ky only (kx keeps its full positive/negative range; ky's negative half
is redundant for a real field) and normalized so that integrating
E(kx, ky) over d(kx)*d(ky) returns the window-weighted variance of the
fluctuation field. The grid spacing comes from the domain lengths in
configs/analysis/grid.yaml; --dx/--dy override it.

Usage:
    uv run scripts/analysis/check_wavenumber_spectrum.py
    uv run scripts/analysis/check_wavenumber_spectrum.py --dataset re16k_t400_0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
import zarr
from matplotlib.colors import LogNorm

from mhd_surrogate.analysis.summary import add_common_args, filter_datasets, write_summary
from mhd_surrogate.data.grid import grid_spacing
from mhd_surrogate.utils.logging_config import setup_logging

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/analysis/split.yaml")
DEFAULT_OUT_DIR = Path("reports/figures")
CHANNEL_NAMES = ["u_x", "u_y"]


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


def wavenumber_spectrum_2d(
    arr,
    channel: int,
    dx: float,
    dy: float,
    chunk_t: int,
    n_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-averaged 2D power spectrum of one channel's *fluctuation about its
    time-mean field* (matching check_autocorrelation.py's `u' = u - <u>_t`
    convention), via 2D FFT.

    `arr` has shape (T, C, Nx, Ny); `channel` selects one of the C channels.
    Subtracting the time-mean *field* (not each snapshot's own scalar
    spatial mean) matters: a channel with a mean profile that varies over
    space but little over time -- e.g. u_x's cross-channel shear profile --
    would otherwise leave that (time-invariant) shape inside what gets
    called the "fluctuation", and since it is present in every snapshot
    essentially unchanged, it dominates the time-averaged spectrum as a
    broad low-wavenumber blob, drowning out the actually time-varying
    structure this analysis is after (confirmed: this was tried first, and
    that is exactly what happened).

    Each snapshot's fluctuation is windowed (a 2D Hann window) before its 2D
    FFT; power is then averaged over time. Two passes over `arr`, each in
    chunks of `chunk_t` snapshots (bounded by `n_steps`, default: all of
    it): the first computes the time-mean field, the second the fluctuation
    spectra -- keeping memory bounded rather than loading the whole region
    at once.

    Returns (kx, ky, power): kx (shape Nx, centered via fftshift, covering
    negative through positive angular wavenumbers), ky (shape Ny // 2 + 1,
    the one-sided non-negative angular wavenumbers -- its negative half is
    redundant for a real field), and power (shape (Nx, Ny // 2 + 1)),
    normalized so that integrating power over d(kx)*d(ky) returns the
    window-weighted variance of the fluctuation field.
    """
    n_steps = arr.shape[0] if n_steps is None else n_steps
    nx, ny = arr.shape[2], arr.shape[3]

    mean_field = np.zeros((nx, ny), dtype=np.float64)
    for start in range(0, n_steps, chunk_t):
        end = min(start + chunk_t, n_steps)
        mean_field += arr[start:end, channel].astype(np.float64).sum(axis=0)
    mean_field /= n_steps

    window = np.outer(np.hanning(nx), np.hanning(ny))
    window_sq_sum = (window**2).sum()
    last = -1 if ny % 2 == 0 else None

    power_sum = np.zeros((nx, ny // 2 + 1), dtype=np.float64)
    for start in range(0, n_steps, chunk_t):
        end = min(start + chunk_t, n_steps)
        block = arr[start:end, channel].astype(np.float64)  # (t, Nx, Ny)
        fluctuation = block - mean_field
        transform = np.fft.rfft2(fluctuation * window, axes=(1, 2))
        power = np.abs(transform) ** 2 * (dx * dy) / window_sq_sum / (2 * np.pi) ** 2
        power[:, :, 1:last] *= 2  # one-sided in ky only
        power_sum += power.sum(axis=0)

    power_mean = np.fft.fftshift(power_sum / n_steps, axes=0)
    kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = 2 * np.pi * np.fft.rfftfreq(ny, d=dy)
    return kx, ky, power_mean


def peak_wavenumbers(kx: np.ndarray, ky: np.ndarray, power: np.ndarray) -> dict:
    """Location of the largest bin, excluding the zero-wavenumber (DC) one."""
    masked = power.copy()
    masked[np.argmin(np.abs(kx)), 0] = -np.inf
    i, j = np.unravel_index(np.argmax(masked), masked.shape)
    return {"kx": kx[i], "ky": ky[j], "power": power[i, j]}


def plot_wavenumber_spectra(
    name: str, kx: np.ndarray, ky: np.ndarray, spectra: dict[str, np.ndarray], out_dir: Path
) -> Path:
    fig, axes = plt.subplots(1, len(spectra), figsize=(7 * len(spectra), 6), squeeze=False)
    for ax, (cname, power) in zip(axes[0], spectra.items()):
        pcm = ax.pcolormesh(kx, ky, power.T, norm=LogNorm(), shading="auto", cmap="viridis")
        ax.set_title(f"{name}: {cname} 2D wavenumber spectrum")
        ax.set_xlabel("kx")
        ax.set_ylabel("ky")
        fig.colorbar(pcm, ax=ax, label="power spectral density")

    fig.tight_layout()
    out_path = out_dir / f"{name}_wavenumber_spectrum.png"
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
        if arr.ndim != 4 or arr.shape[1] != len(CHANNEL_NAMES):
            log.warning("skipping %s: expected shape (T, 2, Nx, Ny), got %s", name, arr.shape)
            continue

        dx, dy = grid_spacing(arr.shape[2], arr.shape[3])
        dx = args.dx if args.dx is not None else dx
        dy = args.dy if args.dy is not None else dy
        n_steps = arr.shape[0]
        print(f"\n=== {name} (dx={dx:.5g}, dy={dy:.5g}, {n_steps} steps) ===")

        spectra = {}
        channel_summary = {}
        for c, cname in enumerate(CHANNEL_NAMES):
            kx, ky, power = wavenumber_spectrum_2d(arr, c, dx, dy, args.chunk_t, n_steps=n_steps)
            spectra[cname] = power
            peak = peak_wavenumbers(kx, ky, power)
            wavelength_x = 2 * np.pi / abs(peak["kx"]) if peak["kx"] != 0 else float("inf")
            wavelength_y = 2 * np.pi / peak["ky"] if peak["ky"] != 0 else float("inf")
            print(
                f"  {cname}: peak at kx={peak['kx']:.3g}, ky={peak['ky']:.3g} "
                f"(wavelength_x={wavelength_x:.3g}, wavelength_y={wavelength_y:.3g})"
            )
            channel_summary[cname] = peak

        plot_path = plot_wavenumber_spectra(name, kx, ky, spectra, args.out_dir)
        print(f"  plot: {plot_path}")

        summary_path = write_summary(
            name,
            "check_wavenumber_spectrum",
            {"dx": dx, "dy": dy, "channels": channel_summary},
            args.summary_dir,
        )
        print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
