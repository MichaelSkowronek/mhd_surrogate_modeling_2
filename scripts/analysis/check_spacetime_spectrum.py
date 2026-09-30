"""Joint wavenumber-frequency ("dispersion") spectrum of the velocity field.

check_wavenumber_spectrum.py found a consistent spatial structure
(|kx| ~ 1.76-2.26, u_x's ky ~ 3.12) across all 9 datasets; check_point_spectrum.py
and check_spatial_mean_spectrum.py found a consistent temporal period
(~25-40 steps, dataset-dependent). Those were found independently, so they
could be two unrelated features of the flow that merely coexist. This
computes the joint spectrum E(kx, ky, omega) -- a full 3D FFT over x, y and
time together -- so a peak at the *combination* of the previously found
wavenumber and frequency is direct evidence of one coherent, moving
structure (a traveling wave, with a phase velocity = omega/kx), not a
coincidence of two separate analyses.

Unlike every other scripts/analysis/*.py script, this cannot process time in
independent chunks -- time is one of the FFT axes -- so it loads the whole
train+val region for one channel into memory at once (matching
check_autocorrelation.py's field_acf), never the held-out test region.

The channel's time-mean field is subtracted first (matching
check_wavenumber_spectrum.py's u' = u - <u>_t convention -- see its
docstring for why that, not each snapshot's own scalar mean, matters), and
a 3D Hann window (the outer product of 1D Hann windows along time, x and y)
is applied before the FFT to limit leakage from the domain's non-periodic
boundaries in space and the finite (and, unlike x/y, non-cyclic-by-nature)
extent in time. The spectrum is one-sided in ky only (kx and omega keep
their full positive/negative range; ky's negative half is redundant for a
real field) and normalized so that integrating E(kx, ky, omega) over
d(kx)*d(ky)*d(omega) returns the window-weighted variance of the
fluctuation field. The grid spacing comes from the domain lengths in
configs/analysis/grid.yaml; --dx/--dy override it. Frequency is in radians
per snapshot step, like check_point_spectrum.py (the physical time step is
not stored in the data).

Peaks at ~6 GB and ~15-20s per dataset/channel (re16k_t400_0, the largest),
noticeably heavier than the other check_*.py scripts. Deliberately not
wired into run_all_checks.py's parallel dispatch: 9 datasets in parallel at
that footprint would exceed a typical machine's RAM. Run datasets one at a
time (the default; or --dataset to pick one).

Usage:
    uv run scripts/analysis/check_spacetime_spectrum.py
    uv run scripts/analysis/check_spacetime_spectrum.py --dataset re16k_t400_0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.colors import LogNorm

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
    add_common_args(parser)
    return parser.parse_args()


def spacetime_spectrum_3d(
    arr, channel: int, dx: float, dy: float, n_steps: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Joint wavenumber-frequency power spectrum of one channel's fluctuation
    about its time-mean field, via a 3D FFT over (time, x, y).

    `arr` has shape (T, C, Nx, Ny); `channel` selects one of the C channels.
    `n_steps` bounds how much of `arr` is read (default: all of it). Unlike
    the other check_*.py core functions, there is no `chunk_t`: the whole
    (bounded) region is loaded at once, since time is itself an FFT axis
    here.

    Returns (omega, kx, ky, power): omega and kx (shape T and Nx) centered
    via fftshift, covering negative through positive; ky (shape
    Ny // 2 + 1) the one-sided non-negative angular wavenumbers -- its
    negative half is redundant for a real field. power has shape
    (T, Nx, Ny // 2 + 1), normalized so that integrating it over
    d(kx)*d(ky)*d(omega) returns the window-weighted variance of the
    fluctuation field.
    """
    n_steps = arr.shape[0] if n_steps is None else n_steps
    data = arr[:n_steps, channel].astype(np.float64)  # (T, Nx, Ny)
    t, nx, ny = data.shape

    fluctuation = data - data.mean(axis=0)
    window = (
        np.hanning(t)[:, None, None] * np.hanning(nx)[None, :, None] * np.hanning(ny)[None, None, :]
    )
    window_sq_sum = (window**2).sum()
    last = -1 if ny % 2 == 0 else None

    transform = np.fft.rfftn(fluctuation * window, axes=(0, 1, 2))
    dt = 1.0  # one snapshot step; the physical time step is not stored in the data
    power = np.abs(transform) ** 2 * (dt * dx * dy) / window_sq_sum / (2 * np.pi) ** 3
    power[:, :, 1:last] *= 2  # one-sided in ky only

    power = np.fft.fftshift(power, axes=(0, 1))
    omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(t, d=dt))
    kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = 2 * np.pi * np.fft.rfftfreq(ny, d=dy)
    return omega, kx, ky, power


def peak_wavenumber_frequency(
    omega: np.ndarray, kx: np.ndarray, ky: np.ndarray, power: np.ndarray
) -> dict:
    """Location of the largest bin, excluding the zero-frequency,
    zero-wavenumber (DC) point.
    """
    masked = power.copy()
    masked[np.argmin(np.abs(omega)), np.argmin(np.abs(kx)), 0] = -np.inf
    i, j, k = np.unravel_index(np.argmax(masked), masked.shape)
    return {
        "omega": omega[i],
        "period": 2 * np.pi / abs(omega[i]) if omega[i] != 0 else float("inf"),
        "kx": kx[j],
        "ky": ky[k],
        "power": power[i, j, k],
    }


def plot_dispersion(
    name: str,
    cname: str,
    omega: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    power: np.ndarray,
    peak: dict,
    out_dir: Path,
) -> Path:
    """(kx, omega) slice through the ky bin closest to the peak's own ky --
    the classic space-time "dispersion" plot: a ridge through the peak would
    indicate a genuine traveling wave with a phase velocity = omega/kx, not
    just coincidentally-coexisting wavenumber and frequency content.
    """
    ky_idx = int(np.argmin(np.abs(ky - peak["ky"])))
    slice_power = power[:, :, ky_idx]  # (omega, kx)

    fig, ax = plt.subplots(figsize=(9, 7))
    pcm = ax.pcolormesh(kx, omega, slice_power, norm=LogNorm(), shading="auto", cmap="viridis")
    ax.scatter(
        [peak["kx"]],
        [peak["omega"]],
        color="red",
        marker="x",
        s=100,
        label=f"peak: T={peak['period']:.1f}, ky={peak['ky']:.3g}",
    )
    ax.set_title(f"{name}: {cname} dispersion, ky={ky[ky_idx]:.3g} slice (train+val)")
    ax.set_xlabel("kx")
    ax.set_ylabel("omega (rad/step)")
    ax.legend()
    fig.colorbar(pcm, ax=ax, label="power spectral density")

    fig.tight_layout()
    out_path = out_dir / f"{name}_{cname}_dispersion.png"
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
        train_end = split["trainval"][1]
        print(f"\n=== {name} (dx={dx:.5g}, dy={dy:.5g}, train+val: [0,{train_end})) ===")

        channel_summary = {}
        for c, cname in enumerate(CHANNEL_NAMES):
            omega, kx, ky, power = spacetime_spectrum_3d(arr, c, dx, dy, n_steps=train_end)
            peak = peak_wavenumber_frequency(omega, kx, ky, power)
            wavelength_x = 2 * np.pi / abs(peak["kx"]) if peak["kx"] != 0 else float("inf")
            wavelength_y = 2 * np.pi / peak["ky"] if peak["ky"] != 0 else float("inf")
            print(
                f"  {cname}: peak at kx={peak['kx']:.3g}, ky={peak['ky']:.3g}, "
                f"omega={peak['omega']:.3g} (period={peak['period']:.1f}, "
                f"wavelength_x={wavelength_x:.3g}, wavelength_y={wavelength_y:.3g})"
            )
            channel_summary[cname] = peak

            plot_path = plot_dispersion(name, cname, omega, kx, ky, power, peak, args.out_dir)
            print(f"  plot: {plot_path}")

        summary_path = write_summary(
            name,
            "check_spacetime_spectrum",
            {"dx": dx, "dy": dy, "channels": channel_summary},
            args.summary_dir,
        )
        print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
