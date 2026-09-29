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

Uses "exact DMD" (Tu et al., 2014), the standard, numerically robust
formulation. `u_x` and `u_y` are stacked into one state vector per
snapshot (DMD models the joint dynamics of the whole vector field, not each
channel independently), and the channel's time-mean field is subtracted
first (matching check_wavenumber_spectrum.py's u' = u - <u>_t convention),
so the spectrum isn't dominated by a large near-unit eigenvalue for the
persistent mean flow.

Like check_spacetime_spectrum.py, this loads the whole train+val region
into memory at once (DMD needs the full snapshot sequence for its SVD, not
just chunks of it), never the held-out test region. The SVD is truncated
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
DEFAULT_RANK = 100
N_MODES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
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


def exact_dmd(
    x: np.ndarray, xprime: np.ndarray, rank: int | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact DMD (Tu et al., 2014): eigenvalues and spatial modes of the
    best-fit linear operator A satisfying `xprime ~= A @ x`, without forming
    A explicitly (the state dimension here is far larger than the number of
    snapshots, so A itself would be too large to form or even fit in
    memory -- only its action, via a rank-r SVD-based reduction, is
    computed).

    `x`, `xprime` have shape (state_dim, n_snapshots): consecutive-snapshot
    pairs, `x[:, i] -> xprime[:, i]` one step later. `rank` truncates the
    SVD of `x` (default: an automatically-determined numerical rank --
    singular values below `s[0] * max(x.shape) * eps` are dropped, matching
    `numpy.linalg.matrix_rank`'s own convention. This only avoids dividing
    by numerically-negligible singular values, which is unconditionally
    unsafe -- e.g. exactly-low-rank data, such as a single spatial pattern
    or few superposed ones, otherwise leaves near-zero values in `s_r` that
    blow up `1/s_r` into numerical garbage. It is not, by itself, denoising:
    pass an explicit smaller `rank` to actually filter out noise-dominated
    small-but-not-negligible singular values, standard DMD practice for
    real, noisy data).

    Returns (eigenvalues, modes, energy_fraction): eigenvalues (complex,
    shape (r,)) are the discrete-time DMD eigenvalues (`lambda =
    exp(mu*dt)` for the continuous-time rate `mu`); modes (complex, shape
    (state_dim, r)) are the corresponding spatial mode shapes, each defined
    only up to an arbitrary complex scale/phase; energy_fraction is the
    share of `x`'s variance the rank-r truncation retains.
    """
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    if rank is None:
        tol = max(x.shape) * np.finfo(s.dtype).eps
        r = int(np.count_nonzero(s > s[0] * tol))
    else:
        r = min(rank, s.shape[0])
    energy_fraction = float((s[:r] ** 2).sum() / (s**2).sum())
    u_r, s_r, v_r = u[:, :r], s[:r], vh[:r].conj().T

    a_tilde = u_r.conj().T @ xprime @ v_r @ np.diag(1.0 / s_r)
    eigenvalues, w = np.linalg.eig(a_tilde)
    modes = xprime @ v_r @ np.diag(1.0 / s_r) @ w
    return eigenvalues, modes, energy_fraction


def mode_amplitudes(modes: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Least-squares amplitude of each DMD mode fitting the first snapshot
    (`x0 ~= modes @ amplitudes`), the standard DMD mode-amplitude convention.
    """
    amplitudes, *_ = np.linalg.lstsq(modes, x0, rcond=None)
    return amplitudes


def dominant_modes(
    eigenvalues: np.ndarray,
    amplitudes: np.ndarray,
    modes: np.ndarray,
    dt: float,
    n_modes: int,
) -> list[dict]:
    """Top `n_modes` DMD modes ranked by `|amplitude| * ||mode||` ("power"),
    deduplicating complex-conjugate eigenvalue pairs -- which both
    represent the same real oscillation -- down to one entry each.

    For a real input, complex eigenvalues always occur in exact conjugate
    pairs (same growth rate, opposite-signed frequency): keeping only
    `frequency >= 0` keeps exactly one representative of each pair, plus
    every purely real eigenvalue (frequency exactly 0) once.

    Returns a list of {"mode_index", "frequency", "growth_rate", "period",
    "amplitude", "power"}, continuous-time (`mu = log(eigenvalue) / dt`):
    frequency = Im(mu) in radians/step, growth_rate = Re(mu) per step
    (positive: growing, negative: decaying, ~0: a sustained
    oscillation/steady structure), period = 2*pi / frequency (inf if ~0).
    """
    mu = np.log(eigenvalues) / dt
    keep = np.flatnonzero(mu.imag >= 0)
    power = np.abs(amplitudes[keep]) * np.linalg.norm(modes[:, keep], axis=0)
    order = keep[np.argsort(power)[::-1]][:n_modes]

    result = []
    for i in order:
        frequency = float(mu[i].imag)
        result.append(
            {
                "mode_index": int(i),
                "frequency": frequency,
                "growth_rate": float(mu[i].real),
                "period": 2 * np.pi / frequency if frequency != 0 else float("inf"),
                "amplitude": complex(amplitudes[i]),
                "power": float(np.abs(amplitudes[i]) * np.linalg.norm(modes[:, i])),
            }
        )
    return result


def plot_eigenvalue_spectrum(name: str, mu: np.ndarray, power: np.ndarray, out_dir: Path) -> Path:
    """Continuous-time DMD spectrum: frequency vs. growth rate, colored by
    each mode's power. Above the growth_rate=0 line: growing; below:
    decaying; on it: a sustained oscillation or steady structure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(mu.imag, mu.real, c=np.log10(power + 1e-300), cmap="viridis", s=40)
    ax.axhline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_title(f"{name}: DMD eigenvalue spectrum (train+val)")
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
    oscillating = [m for m in top if m["frequency"] != 0][:2]
    if not oscillating:
        oscillating = top[:1]

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
        nx, ny = arr.shape[2], arr.shape[3]
        data = arr[:train_end].astype(np.float64)  # (T, 2, Nx, Ny)
        state = data.reshape(train_end, -1)  # (T, state_dim), channels+space flattened
        fluctuation = (state - state.mean(axis=0)).T  # (state_dim, T)
        x, xprime = fluctuation[:, :-1], fluctuation[:, 1:]

        eigenvalues, modes, energy_fraction = exact_dmd(x, xprime, rank=args.rank)
        amplitudes = mode_amplitudes(modes, x[:, 0])
        top = dominant_modes(eigenvalues, amplitudes, modes, dt=1.0, n_modes=args.n_modes)

        print(
            f"\n=== {name} (train+val: [0,{train_end}), rank {args.rank} "
            f"captures {energy_fraction:.1%} of variance) ==="
        )
        for m in top:
            print(
                f"  mode {m['mode_index']}: period={m['period']:.1f} "
                f"(freq={m['frequency']:.3g} rad/step), "
                f"growth_rate={m['growth_rate']:.3g}/step, power={m['power']:.3g}"
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
