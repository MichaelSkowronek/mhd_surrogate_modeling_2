"""Lightweight non-stationarity check: first half vs. second half of train+val.

Every method in the frequency-analysis suite (Welch spectra, DMD, POD, SPOD)
implicitly assumes the process is statistically stationary over the analysis
window -- Welch averages segments together, DMD fits one linear operator for
the whole window, SPOD averages cross-spectral density across segments --
but none of them test that assumption. If it's false (a drifting mean, a
changing variance, or spectral content that shifts partway through), those
methods would silently blur across the change rather than flag it. This has
never been checked directly: `check_split.py`'s per-timestep plots are the
closest thing, but that's a human eyeballing one plot, not a quantified
before/after comparison.

The DNS discards a 400-step warm-up (spin-up) phase before the saved series
begins (see the README's "Origin: from the 3D DNS to this dataset" section),
so a strong startup transient isn't expected in what's recorded -- but that
was a design choice made upstream of this project, not something verified
against the actual data here. This check does that verification: split the
train+val region into two contiguous halves and compare, per channel, the
spatial mean/std, the kinetic energy proxy, and the dominant period/power of
the domain-mean series (a plain periodogram per half, not Welch -- each half
is already short, and Welch would need to shrink segments further to fit).
A meaningful shift in any of these between halves would be direct evidence
of non-stationarity within the window; if they roughly agree, that's direct
(not merely assumed) support for every other script's stationarity
assumption.

Two reading caveats, both consistent with findings already documented in
check_spatial_mean_spectrum.py: `u_x`'s domain-mean series has no clean
isolated spectral peak to begin with (a broad, low bump), so its
"dominant period" is expected to bounce around between two short,
independent halves regardless of stationarity -- `u_y`, which *does* have a
sharp, dominant peak, is the more meaningful period comparison here. And
`u_y`'s mean is ~0 by the channel's own symmetry, so a *relative* change in
it is a near-zero-denominator artifact; this reports mean shifts as an
absolute delta instead, not a percentage, for exactly that reason.

Like every scripts/analysis/*.py script, this never reads past train+val
(`split["trainval"][1]`), let alone the held-out test region.

Usage:
    uv run scripts/analysis/check_stationarity.py
    uv run scripts/analysis/check_stationarity.py --dataset re16k_t400_0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

from mhd_surrogate.analysis.spectral import dominant_periods, power_spectrum
from mhd_surrogate.analysis.summary import add_common_args, filter_datasets, write_summary
from mhd_surrogate.utils.logging_config import setup_logging

log = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path("data/processed/splits/split_manifest.json")
DEFAULT_OUT_DIR = Path("reports/figures")
CHANNEL_NAMES = ["u_x", "u_y"]
N_PEAKS = 1


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
        help="Top spectral peaks to report per half/channel (default: %(default)s)",
    )
    add_common_args(parser)
    return parser.parse_args()


def per_timestep_summary(
    arr, chunk_t: int, n_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spatial mean, spatial std, and the kinetic energy proxy 0.5*u^2
    (spatial mean), per time step and channel, shape (n_steps, n_channels)
    each. Reads `arr` in chunks bounded by `n_steps`, so this stays cheap
    regardless of how long the window is -- this is deliberately a smaller,
    non-shared cousin of check_split.py's per_timestep_stats (no min/max/
    correlation, not needed here).
    """
    n_channels = arr.shape[1]
    means = np.empty((n_steps, n_channels), dtype=np.float64)
    stds = np.empty((n_steps, n_channels), dtype=np.float64)
    energy = np.empty((n_steps, n_channels), dtype=np.float64)
    for start in range(0, n_steps, chunk_t):
        end = min(start + chunk_t, n_steps)
        block = arr[start:end]
        means[start:end] = block.mean(axis=(2, 3))
        stds[start:end] = block.std(axis=(2, 3))
        energy[start:end] = 0.5 * (block.astype(np.float64) ** 2).mean(axis=(2, 3))
    return means, stds, energy


def half_windows(n_steps: int) -> tuple[slice, slice]:
    """Split [0, n_steps) into two contiguous halves (the second gets the
    extra step for an odd `n_steps`)."""
    mid = n_steps // 2
    return slice(0, mid), slice(mid, n_steps)


def relative_change(before: float, after: float) -> float:
    """(after - before) / |before|, i.e. how much `after` differs from
    `before` as a fraction of `before`'s magnitude."""
    return (after - before) / abs(before)


def analyze_half(means: np.ndarray, stds: np.ndarray, energy: np.ndarray, n_peaks: int) -> dict:
    """Per-channel mean/std/energy averaged over a half-window, plus the
    domain-mean series' own dominant period(s) (plain periodogram)."""
    n_channels = means.shape[1]
    channels = {}
    for c, cname in enumerate(CHANNEL_NAMES[:n_channels]):
        periodogram = power_spectrum(means[:, c])
        channels[cname] = {
            "mean": means[:, c].mean(),
            "std": stds[:, c].mean(),
            "energy": energy[:, c].mean(),
            "top_periods": dominant_periods(*periodogram, n_peaks),
        }
    return channels


def plot_halves(
    name: str,
    means: np.ndarray,
    mid: int,
    first: dict,
    second: dict,
    out_dir: Path,
) -> Path:
    n_channels = means.shape[1]
    fig, axes = plt.subplots(n_channels, 2, figsize=(14, 4 * n_channels), squeeze=False)
    for c, cname in enumerate(CHANNEL_NAMES[:n_channels]):
        ax_series, ax_spectrum = axes[c]

        ax_series.plot(means[:, c], color="tab:blue", linewidth=1)
        ax_series.axvline(mid, color="grey", linestyle=":", linewidth=1)
        ax_series.set_title(f"{name}: {cname} spatial mean over time (train+val)")
        ax_series.set_xlabel("time step")

        first_omega, first_power = power_spectrum(means[:mid, c])
        second_omega, second_power = power_spectrum(means[mid:, c])
        ax_spectrum.loglog(
            2 * np.pi / first_omega[1:], first_power[1:], label="first half", color="tab:blue"
        )
        ax_spectrum.loglog(
            2 * np.pi / second_omega[1:], second_power[1:], label="second half", color="tab:orange"
        )
        ax_spectrum.set_title(f"{name}: {cname} spectrum, first vs. second half")
        ax_spectrum.set_xlabel("period T (snapshot steps)")
        ax_spectrum.set_ylabel("power spectral density")
        ax_spectrum.legend()
        ax_spectrum.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / f"{name}_stationarity.png"
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

        trainval_end = split["trainval"][1]
        means, stds, energy = per_timestep_summary(arr, args.chunk_t, trainval_end)
        first_slice, second_slice = half_windows(trainval_end)
        mid = first_slice.stop

        first = analyze_half(
            means[first_slice], stds[first_slice], energy[first_slice], args.n_peaks
        )
        second = analyze_half(
            means[second_slice], stds[second_slice], energy[second_slice], args.n_peaks
        )

        print(
            f"\n=== {name} (train+val: [0,{trainval_end}), "
            f"first half [0,{mid}), second half [{mid},{trainval_end})) ==="
        )
        for cname in first:
            f, s = first[cname], second[cname]
            # mean uses an absolute delta, not relative_change: u_y's mean is
            # ~0 by channel symmetry, so a relative change there is a
            # near-zero-denominator artifact, not a meaningful percentage.
            print(
                f"  {cname}: mean {f['mean']:.4g} -> {s['mean']:.4g} "
                f"(delta={s['mean'] - f['mean']:+.2g}), "
                f"std {f['std']:.4g} -> {s['std']:.4g} "
                f"({relative_change(f['std'], s['std']):+.1%}), "
                f"energy {f['energy']:.4g} -> {s['energy']:.4g} "
                f"({relative_change(f['energy'], s['energy']):+.1%})\n"
                f"    dominant period: first {format_period(f['top_periods'])}, "
                f"second {format_period(s['top_periods'])}"
            )

        plot_path = plot_halves(name, means, mid, first, second, args.out_dir)
        print(f"  plot: {plot_path}")

        summary_path = write_summary(
            name,
            "check_stationarity",
            {"trainval_steps": trainval_end, "mid": mid, "first": first, "second": second},
            args.summary_dir,
        )
        print(f"  summary: {summary_path}")


def format_period(peaks: list[dict]) -> str:
    if not peaks:
        return "none"
    p = peaks[0]
    return f"T={p['period']:.1f} ({p['power_fraction']:.1%})"


if __name__ == "__main__":
    main()
