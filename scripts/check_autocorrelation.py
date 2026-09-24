"""Temporal autocorrelation of the velocity field and of scalar diagnostics.

Two analyses, with lags measured in snapshot steps (the physical time
between snapshots is not stored in the data):

1. Field autocorrelation: the pointwise fluctuation u' = u - <u>_t (the mean
   is the time-mean field of each region) is correlated with itself at lag
   tau, pooled over all grid points and normalized by the zero-lag value,
   separately for u_x and u_y and separately for the train and test time
   steps. This says how quickly the flow decorrelates from a given snapshot,
   which bounds how far apart train and test snapshots must be to be
   effectively independent and gives the effective number of independent
   time steps.
2. Scalar autocorrelation over the whole series: the spatial means of u_x
   and u_y and the kinetic energy per direction (0.5*u^2, spatial mean),
   with the approximate +-1.96/sqrt(N) white-noise band for reference (only
   a rough guide for strongly autocorrelated series).

Per curve it reports the lag-1 correlation, the lags at which the
autocorrelation first falls below 1/e and below 0.05, the first zero
crossing, and the integral time scale tau_int = 1/2 + sum(rho) up to the
first zero crossing, giving N_eff = N / (2 * tau_int). Because the tail of
the estimate is noisy, tau_int and N_eff are rough.

Usage:
    uv run scripts/check_autocorrelation.py
    uv run scripts/check_autocorrelation.py --max-lag 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

DEFAULT_MANIFEST = Path("data/processed/splits/split_manifest.json")
DEFAULT_OUT_DIR = Path("reports/figures")
CHANNEL_NAMES = ["u_x", "u_y"]
SCALAR_NAMES = [
    "u_x spatial mean",
    "u_y spatial mean",
    "kinetic energy 0.5*u_x^2",
    "kinetic energy 0.5*u_y^2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--max-lag",
        type=int,
        default=150,
        help="Max field lag in steps (default: %(default)s)",
    )
    parser.add_argument(
        "--slab",
        type=int,
        default=64,
        help="x columns per FFT slab (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-t",
        type=int,
        default=64,
        help="Time steps per scalar pass (default: %(default)s)",
    )
    return parser.parse_args()


def acf_from_power(power: np.ndarray, n_steps: int, n_fft: int, max_lag: int) -> np.ndarray:
    """Normalized autocorrelation from a summed power spectrum along the last axis.

    The signal was zero-padded to n_fft >= 2*n_steps, so the inverse
    transform gives the linear (not circular) autocovariance sums, which are
    divided by the number of overlapping samples (n_steps - lag).
    """
    autocov_sum = np.fft.irfft(power, n=n_fft, axis=-1)[..., : max_lag + 1]
    lags = np.arange(max_lag + 1)
    autocov = autocov_sum / (n_steps - lags)
    return autocov / autocov[..., :1]


def field_acf(data: np.ndarray, max_lag: int, slab: int) -> np.ndarray:
    """Pointwise-fluctuation autocorrelation pooled over space, shape (C, max_lag+1).

    `data` has shape (T, C, Nx, Ny). Power spectra are accumulated over x
    slabs so only one slab is held as float64 at a time.
    """
    n_steps, n_channels, nx, _ = data.shape
    n_fft = 2 * n_steps
    time_mean = data.mean(axis=0, dtype=np.float64)

    power = 0.0
    for x0 in range(0, nx, slab):
        x1 = min(x0 + slab, nx)
        fluctuation = data[:, :, x0:x1, :].astype(np.float64) - time_mean[:, x0:x1, :]
        transform = np.fft.rfft(fluctuation, n=n_fft, axis=0)
        power = power + (np.abs(transform) ** 2).sum(axis=(2, 3))  # (K, C)

    return acf_from_power(np.moveaxis(power, 0, -1), n_steps, n_fft, max_lag)


def scalar_series(data: np.ndarray, chunk_t: int) -> np.ndarray:
    """Spatial means of u_x, u_y and per-direction energy 0.5*u^2, shape (T, 4).

    `data` may be a zarr array; it is read in time chunks.
    """
    n_steps = data.shape[0]
    out = np.empty((n_steps, 4), dtype=np.float64)
    for start in range(0, n_steps, chunk_t):
        end = min(start + chunk_t, n_steps)
        block = data[start:end].astype(np.float64)
        out[start:end, :2] = block.mean(axis=(2, 3))
        out[start:end, 2:] = 0.5 * (block**2).mean(axis=(2, 3))
    return out


def series_acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized autocorrelation of each column of a (T, S) series, shape (S, max_lag+1)."""
    n_steps = series.shape[0]
    n_fft = 2 * n_steps
    centered = series - series.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(centered, n=n_fft, axis=0)) ** 2
    return acf_from_power(power.T, n_steps, n_fft, max_lag)


def decorrelation_metrics(rho: np.ndarray, n_steps: int) -> dict:
    """Lags of key decorrelation thresholds; None where not reached in the window."""

    def first_below(threshold: float):
        below = np.flatnonzero(rho <= threshold)
        return int(below[0]) if below.size else None

    zero = first_below(0.0)
    tau_int = 0.5 + rho[1:zero].sum() if zero is not None else None
    return {
        "lag1": rho[1],
        "one_over_e": first_below(1 / np.e),
        "below_0.05": first_below(0.05),
        "zero": zero,
        "tau_int": tau_int,
        "n_eff": n_steps / (2 * tau_int) if tau_int else None,
    }


def format_metrics(m: dict) -> str:
    def fmt(value, spec="d"):
        return "not reached" if value is None else format(value, spec)

    return (
        f"rho(1)={m['lag1']:.3f} | lag at 1/e: {fmt(m['one_over_e'])}, "
        f"at 0.05: {fmt(m['below_0.05'])}, first zero: {fmt(m['zero'])} | "
        f"tau_int={fmt(m['tau_int'], '.1f')} steps, N_eff={fmt(m['n_eff'], '.1f')}"
    )


def plot_acf(
    name: str,
    field: dict[str, np.ndarray],
    scalars: np.ndarray,
    n_steps: int,
    out_dir: Path,
) -> Path:
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    ax = axes[0]
    for region, style in (("train", "-"), ("test", "--")):
        for c, cname in enumerate(CHANNEL_NAMES):
            rho = field[region][c]
            ax.plot(
                np.arange(len(rho)),
                rho,
                style,
                color=colors[c],
                label=f"{cname} {region}",
            )
    ax.axhline(1 / np.e, color="grey", linestyle=":", label="1/e")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"{name}: field autocorrelation (pooled over space)")
    ax.set_xlabel("lag (snapshot steps)")
    ax.set_ylabel("autocorrelation")
    ax.legend()

    ax = axes[1]
    for s, sname in enumerate(SCALAR_NAMES):
        ax.plot(np.arange(scalars.shape[1]), scalars[s], color=colors[s], label=sname)
    bound = 1.96 / np.sqrt(n_steps)
    ax.axhspan(-bound, bound, color="grey", alpha=0.2, label="+-1.96/sqrt(N)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"{name}: scalar diagnostics autocorrelation (all steps)")
    ax.set_xlabel("lag (snapshot steps)")
    ax.set_ylabel("autocorrelation")
    ax.legend()

    fig.tight_layout()
    out_path = out_dir / f"{name}_autocorrelation.png"
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
        if arr.ndim != 4 or arr.shape[1] != len(CHANNEL_NAMES):
            print(f"skipping {name}: expected shape (T, 2, Nx, Ny), got {arr.shape}")
            continue

        regions = {"train": split["train"], "test": split["test"]}
        field: dict[str, np.ndarray] = {}
        print(f"\n=== {name} (lags in snapshot steps) ===")
        print("field autocorrelation, fluctuations about each region's time-mean field:")
        for region, (start, end) in regions.items():
            data = arr[start:end]
            n_steps = end - start
            max_lag = min(args.max_lag, n_steps // 2)
            field[region] = field_acf(data, max_lag, args.slab)
            del data
            for c, cname in enumerate(CHANNEL_NAMES):
                metrics = decorrelation_metrics(field[region][c], n_steps)
                print(f"  {cname} {region} (N={n_steps}): {format_metrics(metrics)}")

        series = scalar_series(arr, args.chunk_t)
        total_steps = series.shape[0]
        scalar_lag = total_steps // 3
        scalars = series_acf(series, scalar_lag)
        print("scalar diagnostics autocorrelation, all steps:")
        for s, sname in enumerate(SCALAR_NAMES):
            metrics = decorrelation_metrics(scalars[s], total_steps)
            print(f"  {sname} (N={total_steps}): {format_metrics(metrics)}")

        print(f"  plot: {plot_acf(name, field, scalars, total_steps, args.out_dir)}")


if __name__ == "__main__":
    main()
