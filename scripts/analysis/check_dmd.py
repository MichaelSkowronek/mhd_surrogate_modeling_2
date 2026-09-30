"""Dynamic Mode Decomposition (DMD) of the velocity field.

The frequency-analysis scripts so far all assume energy sits on a fixed
grid of FFT frequencies. DMD instead fits the best linear dynamical system
A (in a least-squares sense, x_{t+1} ~= A x_t) directly to the snapshot
sequence and eigendecomposes it: each eigenvalue gives a mode's frequency
*and* growth/decay rate (not just frequency), at whatever frequency the
data actually supports, not a grid-locked bin. A sustained coherent
structure should show up as an eigenvalue near the unit circle (neither
growing nor decaying) at the same frequency check_spacetime_spectrum.py
already found; this is a genuinely different method arriving at the same
answer, not another FFT variant.

Modes are selected by "power" (amplitude at the first snapshot, the
standard DMD convention), but printed sorted by "rms_power" (root-mean-
square power *over the whole recorded window*) instead: a fast-decaying
mode can have high power yet vanish within a handful of steps, contributing
far less to the actual recorded series than a lower-power but near-neutral
one that persists throughout -- see dominant_modes's docstring for the
exact definition, and this project's own re16k_t400_0 as a real example
(the top-power mode's rms_power is roughly a quarter of the second-ranked
mode's, despite having higher power).

Uses "exact DMD" (Tu et al., 2014), the standard, numerically robust
formulation. `u_x` and `u_y` are stacked into one state vector per
snapshot (DMD models the joint dynamics of the whole vector field, not each
channel independently), and the channel's time-mean field is subtracted
first (matching check_wavenumber_spectrum.py's u' = u - <u>_t convention),
so the spectrum isn't dominated by a large near-unit eigenvalue for the
persistent mean flow. The DMD machinery itself
(`build_dmd_state`/`exact_dmd`/`mode_amplitudes`/`dominant_modes`/
`reconstruct_frames`) lives in `src/mhd_surrogate/analysis/dmd.py`, shared
with `scripts/viz/make_dmd_video.py`.

Like check_spacetime_spectrum.py, this loads a dataset's whole recorded
length into memory at once (DMD needs the full snapshot sequence for its
SVD, not just chunks of it), never re16k_t400_5, the model's held-out test
set (excluded entirely from configs/analysis/split.yaml). The SVD is truncated
to --rank (default 100) for robustness against small, noise-dominated
singular values -- standard DMD practice; the energy this captures is
printed for transparency. Frequency is in radians per snapshot step, like
the other scripts here (the physical time step is not stored in the data).

Peaks at ~8 GB and ~20-25s per dataset (re16k_t400_0, the largest),
heavier even than check_spacetime_spectrum.py. Deliberately not wired into
run_all_checks.py's parallel dispatch for the same reason: 9 datasets in
parallel at that footprint would exceed a typical machine's RAM. Run
datasets one at a time (the default; or --dataset to pick one).

Usage:
    uv run scripts/analysis/check_dmd.py
    uv run scripts/analysis/check_dmd.py --dataset re16k_t400_0 --rank 50
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
import zarr

from mhd_surrogate.analysis.dmd import build_dmd_state, dominant_modes, exact_dmd, mode_amplitudes
from mhd_surrogate.analysis.summary import add_common_args, filter_datasets, write_summary
from mhd_surrogate.utils.logging_config import setup_logging

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/analysis/split.yaml")
DEFAULT_OUT_DIR = Path("reports/figures")
CHANNEL_NAMES = ["u_x", "u_y"]
DEFAULT_RANK = 100
N_MODES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--rank",
        type=int,
        default=DEFAULT_RANK,
        help="SVD truncation rank (default: %(default)s)",
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        default=N_MODES,
        help="Top DMD modes to report/plot (default: %(default)s)",
    )
    add_common_args(parser)
    return parser.parse_args()


def plot_eigenvalue_spectrum(name: str, mu: np.ndarray, power: np.ndarray, out_dir: Path) -> Path:
    """Continuous-time DMD spectrum: frequency vs. growth rate, colored by
    each mode's power. Above the growth_rate=0 line: growing; below:
    decaying; on it: a sustained oscillation or steady structure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(mu.imag, mu.real, c=np.log10(power + 1e-300), cmap="viridis", s=40)
    ax.axhline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_title(f"{name}: DMD eigenvalue spectrum")
    ax.set_xlabel("frequency (rad/step)")
    ax.set_ylabel("growth rate (per step)")
    fig.colorbar(sc, ax=ax, label="log10(power)")
    fig.tight_layout()
    out_path = out_dir / f"{name}_dmd_spectrum.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_mode_shapes(
    name: str, top: list[dict], modes: np.ndarray, nx: int, ny: int, out_dir: Path
) -> Path:
    # Selected by rms_power, not power: a high-power-but-fast-decaying mode
    # can matter far less to the recorded series than a lower-power but
    # persistent one -- see dominant_modes's docstring.
    by_rms = sorted(top, key=lambda m: m["rms_power"], reverse=True)
    oscillating = [m for m in by_rms if m["frequency"] != 0][:2]
    if not oscillating:
        oscillating = by_rms[:1]

    fig, axes = plt.subplots(len(oscillating), 2, figsize=(12, 5 * len(oscillating)), squeeze=False)
    for row, m in zip(axes, oscillating):
        mode = modes[:, m["mode_index"]].reshape(len(CHANNEL_NAMES), nx, ny)
        for ax, cname, field in zip(row, CHANNEL_NAMES, mode.real):
            limit = np.percentile(np.abs(field), 99)
            im = ax.imshow(np.rot90(field), cmap="coolwarm", vmin=-limit, vmax=limit)
            ax.set_title(f"{name}: {cname}, mode T={m['period']:.1f} (Re)")
            fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    out_path = out_dir / f"{name}_dmd_modes.png"
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

        n_steps = arr.shape[0]
        nx, ny = arr.shape[2], arr.shape[3]
        data = arr[:n_steps].astype(np.float64)  # (T, 2, Nx, Ny)
        fluctuation = build_dmd_state(data)  # (state_dim, T)
        x, xprime = fluctuation[:, :-1], fluctuation[:, 1:]

        eigenvalues, modes, energy_fraction = exact_dmd(x, xprime, rank=args.rank)
        amplitudes = mode_amplitudes(modes, x[:, 0])
        top = dominant_modes(
            eigenvalues, amplitudes, modes, dt=1.0, n_steps=n_steps, n_modes=args.n_modes
        )

        print(
            f"\n=== {name} ({n_steps} steps, rank {args.rank} "
            f"captures {energy_fraction:.1%} of variance) ==="
        )
        # Selection above is by power (the standard DMD convention, at the
        # first snapshot); displayed here sorted by rms_power (how much a
        # mode actually persists across the recorded window) instead, since
        # those two can disagree sharply -- see dominant_modes's docstring.
        for m in sorted(top, key=lambda m: m["rms_power"], reverse=True):
            print(
                f"  mode {m['mode_index']}: period={m['period']:.1f} "
                f"(freq={m['frequency']:.3g} rad/step), "
                f"growth_rate={m['growth_rate']:.3g}/step, "
                f"power={m['power']:.3g} (rms over window: {m['rms_power']:.3g})"
            )

        mu_all = np.log(eigenvalues) / 1.0
        power_all = np.abs(amplitudes) * np.linalg.norm(modes, axis=0)
        spectrum_path = plot_eigenvalue_spectrum(name, mu_all, power_all, args.out_dir)
        print(f"  spectrum plot: {spectrum_path}")
        modes_path = plot_mode_shapes(name, top, modes, nx, ny, args.out_dir)
        print(f"  modes plot: {modes_path}")

        summary_path = write_summary(
            name,
            "check_dmd",
            {"rank": args.rank, "energy_fraction": energy_fraction, "modes": top},
            args.summary_dir,
        )
        print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
