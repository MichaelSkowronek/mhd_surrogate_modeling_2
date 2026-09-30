"""Spectral Proper Orthogonal Decomposition (SPOD) of the velocity field.

Bridges check_pod.py and check_dmd.py: standard POD's modes are ranked by
variance over the *whole* recorded window but each mode's time coefficient
can mix many frequencies (the near-equal-energy mode-0/mode-1 pair
check_pod.py finds in every dataset is the direct consequence -- POD has to
split one traveling structure into two spatially-orthogonal modes, since it
has no other way to represent a single oscillation); DMD's modes are
single-frequency by construction but not generally spatially orthogonal,
and a mode's whole-window importance has to be estimated separately from
its frequency (dominant_modes's rms_power). SPOD gets both: at each
frequency independently, it takes the SVD of that frequency's
Welch-blocked, Hann-windowed Fourier coefficients across segments -- same
blocking as check_point_spectrum.py's Welch estimator, same SVD machinery
as check_pod.py, just applied per-frequency instead of once over the whole
time domain. The result should show its energy concentrated at the same
period check_dmd.py and check_pod.py already found, in a single mode whose
shape looks like the coherent structure they found too -- a third,
genuinely different method (per-frequency eigendecomposition, not a
whole-window SVD or a fitted linear operator) converging on the same
answer.

`u_x` and `u_y` are stacked into one state vector per snapshot (same
convention as check_dmd.py/check_pod.py) and the time-mean field is
subtracted first via the shared `build_dmd_state`. The SPOD machinery
itself (`spod`/`leading_frequencies`) lives in
`src/mhd_surrogate/analysis/spod.py`.

Like check_dmd.py and check_pod.py, this loads the whole train+val region
into memory at once, never the held-out test region. Frequency is in
radians per snapshot step, like the other scripts here (the physical time
step is not stored in the data). The Welch segment length --nperseg
defaults to 200, matching check_point_spectrum.py's choice for this
project's ~500-730 step train+val regions.

Peaks at ~6-8 GB per dataset (similar to check_pod.py, since it starts from
the same state matrix, plus the segment FFTs). Deliberately not wired into
run_all_checks.py's parallel dispatch for the same reason as check_dmd.py/
check_pod.py: 9 datasets in parallel at that footprint would exceed a
typical machine's RAM. Run datasets one at a time (the default; or
--dataset to pick one).

Usage:
    uv run scripts/analysis/check_spod.py
    uv run scripts/analysis/check_spod.py --dataset re16k_t400_0 --nperseg 150
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

from mhd_surrogate.analysis.dmd import build_dmd_state
from mhd_surrogate.analysis.spod import leading_frequencies, spod
from mhd_surrogate.analysis.summary import add_common_args, filter_datasets, write_summary
from mhd_surrogate.utils.logging_config import setup_logging

log = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path("data/processed/splits/split_manifest.json")
DEFAULT_OUT_DIR = Path("reports/figures")
CHANNEL_NAMES = ["u_x", "u_y"]
N_PEAKS = 3
N_MODE_RANKS = 3  # leading-mode ranks shown in the eigenvalue spectrum plot
NPERSEG = 200  # see this script's docstring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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
    parser.add_argument(
        "--n-peaks",
        type=int,
        default=N_PEAKS,
        help="Top dominant frequencies to report (default: %(default)s)",
    )
    add_common_args(parser)
    return parser.parse_args()


def plot_eigenvalue_spectrum(
    name: str, omega: np.ndarray, eigenvalues: np.ndarray, n_mode_ranks: int, out_dir: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    period = 2 * np.pi / omega[1:]
    for i in range(min(n_mode_ranks, eigenvalues.shape[1])):
        ax.loglog(period, eigenvalues[1:, i], label=f"mode {i}")
    ax.set_title(f"{name}: SPOD eigenvalue spectrum (train+val)")
    ax.set_xlabel("period T (snapshot steps)")
    ax.set_ylabel("eigenvalue (energy)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"{name}_spod_spectrum.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_mode_shape(
    name: str, mode: np.ndarray, period: float, nx: int, ny: int, out_dir: Path
) -> Path:
    reshaped = mode.reshape(len(CHANNEL_NAMES), nx, ny)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cname, field in zip(axes, CHANNEL_NAMES, reshaped.real):
        limit = np.percentile(np.abs(field), 99)
        im = ax.imshow(np.rot90(field), cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_title(f"{name}: {cname}, SPOD mode 0, T={period:.1f} (Re)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out_path = out_dir / f"{name}_spod_modes.png"
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

        train_end = split["trainval"][1]
        nx, ny = arr.shape[2], arr.shape[3]
        data = arr[:train_end].astype(np.float64)  # (T, 2, Nx, Ny)
        state = build_dmd_state(data)  # (state_dim, T)

        omega, modes, eigenvalues = spod(state, args.nperseg, noverlap)
        peaks = leading_frequencies(omega, eigenvalues, args.n_peaks)

        print(f"\n=== {name} (train+val: [0,{train_end}), nperseg={args.nperseg}) ===")
        for p in peaks:
            print(
                f"  T={p['period']:.1f} (omega={p['omega']:.3g} rad/step): "
                f"eigenvalue={p['eigenvalue']:.3g} ({p['eigenvalue_fraction']:.1%} of total)"
            )

        spectrum_path = plot_eigenvalue_spectrum(
            name, omega, eigenvalues, N_MODE_RANKS, args.out_dir
        )
        print(f"  spectrum plot: {spectrum_path}")

        top = peaks[0]
        top_freq_idx = int(np.argmin(np.abs(omega - top["omega"])))
        modes_path = plot_mode_shape(
            name, modes[top_freq_idx, :, 0], top["period"], nx, ny, args.out_dir
        )
        print(f"  modes plot: {modes_path}")

        summary_path = write_summary(
            name,
            "check_spod",
            {"nperseg": args.nperseg, "noverlap": noverlap, "leading_frequencies": peaks},
            args.summary_dir,
        )
        print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
