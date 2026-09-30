"""Proper Orthogonal Decomposition (POD, aka PCA) of the velocity field.

Complements check_dmd.py: DMD gives each mode a single frequency and
growth/decay rate but its modes are not generally orthogonal; POD gives up
mode-by-mode frequency information (a POD mode's time coefficient can, and
typically does, mix many frequencies) in exchange for modes that are
mutually orthogonal and exactly ranked by the variance ("energy") they
capture -- the classic "which spatial patterns actually account for most of
the flow's variability" answer, with no eigenvalue problem, no complex
arithmetic, and (unlike DMD) no power-vs-persistence ambiguity to worry
about, since the energy is already a whole-window quantity by construction
(see mhd_surrogate.analysis.pod's docstring). It's also cheaper than DMD:
just the SVD of the same state matrix, without DMD's extra eigendecomposition
and least-squares steps.

`u_x` and `u_y` are stacked into one state vector per snapshot (same
convention as check_dmd.py: POD, too, describes the joint spatial structure
of the whole vector field, not each channel independently), and the
channel's time-mean field is subtracted first (matching
check_wavenumber_spectrum.py's u' = u - <u>_t convention) -- otherwise the
first "mode" would just be the persistent mean-flow shape, not a fluctuation
pattern. The POD machinery itself (`pod`) lives in
`src/mhd_surrogate/analysis/pod.py`; `build_dmd_state` is reused directly
from `src/mhd_surrogate/analysis/dmd.py`.

Like check_dmd.py, this loads the whole train+val region into memory at
once, never the held-out test region. Frequency is not reported at all
(POD modes don't have one); the closest analogue -- roughly what timescale
a mode's coefficient varies on -- is a question for the coefficient's own
spectrum, not computed here.

Peaks at ~6-8 GB per dataset (similar to check_dmd.py, since it starts from
the same state matrix). Deliberately not wired into run_all_checks.py's
parallel dispatch for the same reason: 9 datasets in parallel at that
footprint would exceed a typical machine's RAM. Run datasets one at a time
(the default; or --dataset to pick one).

Usage:
    uv run scripts/analysis/check_pod.py
    uv run scripts/analysis/check_pod.py --dataset re16k_t400_0 --n-modes 5
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
from mhd_surrogate.analysis.pod import pod
from mhd_surrogate.analysis.summary import add_common_args, filter_datasets, write_summary
from mhd_surrogate.utils.logging_config import setup_logging

log = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path("data/processed/splits/split_manifest.json")
DEFAULT_OUT_DIR = Path("reports/figures")
CHANNEL_NAMES = ["u_x", "u_y"]
N_MODES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--n-modes",
        type=int,
        default=N_MODES,
        help="Top POD modes to report/plot the shape of (default: %(default)s)",
    )
    add_common_args(parser)
    return parser.parse_args()


def plot_energy_spectrum(name: str, energy_fraction: np.ndarray, out_dir: Path) -> Path:
    cumulative = np.cumsum(energy_fraction)
    fig, ax1 = plt.subplots(figsize=(8, 6))
    mode_index = np.arange(1, len(energy_fraction) + 1)
    ax1.semilogy(mode_index, energy_fraction, "o-", markersize=3, color="tab:blue")
    ax1.set_xlabel("mode")
    ax1.set_ylabel("energy fraction", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(mode_index, cumulative, color="tab:orange")
    ax2.set_ylabel("cumulative energy fraction", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, 1.05)

    ax1.set_title(f"{name}: POD energy spectrum (train+val)")
    fig.tight_layout()
    out_path = out_dir / f"{name}_pod_spectrum.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_mode_shapes(
    name: str,
    modes: np.ndarray,
    energy_fraction: np.ndarray,
    n_modes: int,
    nx: int,
    ny: int,
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(n_modes, 2, figsize=(12, 5 * n_modes), squeeze=False)
    for i, row in zip(range(n_modes), axes):
        mode = modes[:, i].reshape(len(CHANNEL_NAMES), nx, ny)
        for ax, cname, field in zip(row, CHANNEL_NAMES, mode):
            limit = np.percentile(np.abs(field), 99)
            im = ax.imshow(np.rot90(field), cmap="coolwarm", vmin=-limit, vmax=limit)
            ax.set_title(f"{name}: {cname}, POD mode {i} ({energy_fraction[i]:.1%} energy)")
            fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    out_path = out_dir / f"{name}_pod_modes.png"
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
        state = build_dmd_state(data)  # (state_dim, T)

        modes, energy_fraction, coefficients = pod(state)
        cumulative = np.cumsum(energy_fraction)

        print(f"\n=== {name} (train+val: [0,{train_end}), {len(energy_fraction)} POD modes) ===")
        for i in range(args.n_modes):
            print(
                f"  mode {i}: energy={energy_fraction[i]:.2%}, "
                f"cumulative={cumulative[i]:.2%}, "
                f"coefficient std={coefficients[i].std():.3g}"
            )
        n_for_90pct = int(np.searchsorted(cumulative, 0.9) + 1)
        print(f"  {n_for_90pct} modes capture 90% of the variance")

        spectrum_path = plot_energy_spectrum(name, energy_fraction, args.out_dir)
        print(f"  spectrum plot: {spectrum_path}")
        modes_path = plot_mode_shapes(
            name, modes, energy_fraction, args.n_modes, nx, ny, args.out_dir
        )
        print(f"  modes plot: {modes_path}")

        summary_path = write_summary(
            name,
            "check_pod",
            {
                "n_modes_total": len(energy_fraction),
                "n_modes_for_90pct_energy": n_for_90pct,
                "top_modes": [
                    {
                        "mode_index": i,
                        "energy_fraction": float(energy_fraction[i]),
                        "cumulative_energy_fraction": float(cumulative[i]),
                        "coefficient_std": float(coefficients[i].std()),
                    }
                    for i in range(args.n_modes)
                ],
            },
            args.summary_dir,
        )
        print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
