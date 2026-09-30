"""Render an animation of a DMD mode's own time evolution.

check_dmd.py's mode-shape plot shows a single static snapshot (Re(mode)) of
each top mode; this instead animates it, evaluating the fitted linear
dynamics `Re(amplitude * mode * eigenvalue**t)` frame by frame -- much more
direct than trying to read a traveling/oscillating structure off a static
image. Re-runs the same DMD decomposition as check_dmd.py (the mode vectors
themselves aren't persisted anywhere -- only summary statistics are), so
this has the same cost: ~8 GB and ~20-25s for the largest dataset.

By default, renders the top `--n-videos` *oscillating* modes from the top
`--n-modes` ranked by power (same selection as check_dmd.py's mode-shape
plot); pass one or more `--mode-index` (from check_dmd.py's printed output)
to render specific modes instead, oscillating or not.

Growth/decay is normalized out by default (`--include-growth` to keep it):
an oscillating mode's amplitude is otherwise fully decayed or blown up
within a few periods, and check_dmd.py already reports the growth rate
separately as a number. A non-oscillating mode (e.g. a slow drift) has
nothing left to animate if its growth is also normalized away, so it is
never normalized regardless of this flag -- see
mhd_surrogate.analysis.dmd.reconstruct_frames's docstring.

Requires no system ffmpeg install: encoding uses the static binary bundled
by the imageio-ffmpeg package.

Usage:
    uv run scripts/viz/make_dmd_video.py --dataset re16k_t400_0
    uv run scripts/viz/make_dmd_video.py --dataset re16k_t400_0 --mode-index 70 --include-growth
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
import zarr  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

from mhd_surrogate.analysis.dmd import (  # noqa: E402
    build_dmd_state,
    dominant_modes,
    exact_dmd,
    mode_amplitudes,
    reconstruct_frames,
)
from mhd_surrogate.analysis.fields import FIELDS, compute_field  # noqa: E402
from mhd_surrogate.data.grid import grid_spacing  # noqa: E402
from mhd_surrogate.utils.logging_config import add_log_level_arg, setup_logging  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/analysis/split.yaml")
DEFAULT_OUT_DIR = Path("reports/videos")
CHANNEL_NAMES = ["u_x", "u_y"]
DEFAULT_RANK = 100
N_MODES = 5
N_VIDEOS = 2
PERIODS = 3.0
FALLBACK_FRAMES = 150  # for a non-oscillating (period=inf) mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", required=True, help="Dataset name (must be configured)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--field", choices=sorted(FIELDS), default="vorticity")
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK, help="SVD truncation rank")
    parser.add_argument(
        "--n-modes",
        type=int,
        default=N_MODES,
        help="Size of the ranked-by-power pool to pick default modes from (default: %(default)s)",
    )
    parser.add_argument(
        "--n-videos",
        type=int,
        default=N_VIDEOS,
        help="How many top oscillating modes to render, if --mode-index is not given",
    )
    parser.add_argument(
        "--mode-index",
        action="append",
        type=int,
        dest="mode_indices",
        default=None,
        help="Render this specific mode index (repeatable); overrides --n-videos",
    )
    parser.add_argument(
        "--periods",
        type=float,
        default=PERIODS,
        help="Oscillation periods to render, for a finite-period mode (default: %(default)s)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Explicit frame count (default: derived from --periods, or "
        f"{FALLBACK_FRAMES} for a non-oscillating mode)",
    )
    parser.add_argument(
        "--include-growth",
        action="store_true",
        help="Keep growth/decay in the animation instead of normalizing it out",
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dx", type=float, default=None, help="Override spacing along axis 2")
    parser.add_argument("--dy", type=float, default=None, help="Override spacing along axis 3")
    add_log_level_arg(parser)
    return parser.parse_args()


def render_mode_video(
    name: str,
    mode_entry: dict,
    mode: np.ndarray,
    nx: int,
    ny: int,
    field: str,
    dx: float,
    dy: float,
    n_frames: int,
    normalize_growth: bool,
    fps: int,
    out_path: Path,
) -> None:
    frames = reconstruct_frames(
        mode, mode_entry["amplitude"], mode_entry["eigenvalue"], n_frames, normalize_growth
    )
    block = frames.reshape(n_frames, len(CHANNEL_NAMES), nx, ny)
    field_block = compute_field(block, field, dx, dy)

    lo, hi = np.percentile(field_block, [1, 99])
    if FIELDS[field]["symmetric"]:
        limit = max(abs(lo), abs(hi))
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = lo, hi

    fig, ax = plt.subplots(figsize=(14, 3))
    im = ax.imshow(np.zeros((ny, nx)), cmap=FIELDS[field]["cmap"], vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.8)
    title = ax.set_title("")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, str(out_path), dpi=150):
        for t in range(n_frames):
            im.set_data(np.rot90(field_block[t]))
            title.set_text(
                f"{name}: DMD mode {mode_entry['mode_index']} "
                f"(T={mode_entry['period']:.1f}), {field} at t={t}"
            )
            writer.grab_frame()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = yaml.safe_load(args.config.read_text())
    if args.dataset not in config["datasets"]:
        raise SystemExit(
            f"{args.dataset!r} is not a configured dataset (configs/analysis/split.yaml); "
            "re16k_t400_5 in particular is the model's held-out test set and must never be read"
        )
    root = zarr.open_group(store=config["zarr_store"], mode="r")

    arr = root[args.dataset]
    if arr.ndim != 4 or arr.shape[1] != len(CHANNEL_NAMES):
        raise SystemExit(f"expected shape (T, 2, Nx, Ny), got {arr.shape}")
    nx, ny = arr.shape[2], arr.shape[3]
    dx, dy = grid_spacing(nx, ny)
    dx = args.dx if args.dx is not None else dx
    dy = args.dy if args.dy is not None else dy

    n_steps = arr.shape[0]
    log.info("running DMD on %s (%d steps); this is the expensive part", args.dataset, n_steps)
    data = arr[:n_steps].astype(np.float64)
    fluctuation = build_dmd_state(data)
    x, xprime = fluctuation[:, :-1], fluctuation[:, 1:]
    eigenvalues, modes, energy_fraction = exact_dmd(x, xprime, rank=args.rank)
    amplitudes = mode_amplitudes(modes, x[:, 0])
    top = dominant_modes(
        eigenvalues, amplitudes, modes, dt=1.0, n_steps=n_steps, n_modes=args.n_modes
    )
    log.info("rank %d captures %.1f%% of variance", args.rank, 100 * energy_fraction)

    if args.mode_indices:
        by_index = {m["mode_index"]: m for m in top}
        selected = [by_index[i] for i in args.mode_indices if i in by_index]
        missing = set(args.mode_indices) - by_index.keys()
        if missing:
            raise SystemExit(
                f"mode index/indices {sorted(missing)} not among the top {args.n_modes} by "
                "power; increase --n-modes or pick from check_dmd.py's printed output"
            )
    else:
        # By rms_power, not power: a high-power-but-fast-decaying mode can
        # matter far less to the recorded series than a lower-power but
        # persistent one -- see dominant_modes's docstring.
        by_rms = sorted(top, key=lambda m: m["rms_power"], reverse=True)
        selected = [m for m in by_rms if m["frequency"] != 0][: args.n_videos]
        if not selected:
            selected = by_rms[:1]

    for m in selected:
        n_frames = args.frames
        if n_frames is None:
            n_frames = (
                round(args.periods * m["period"]) if np.isfinite(m["period"]) else FALLBACK_FRAMES
            )
        out_path = args.out_dir / f"{args.dataset}_dmd_mode{m['mode_index']}_{args.field}.mp4"
        log.info(
            "mode %d: period=%.1f, growth_rate=%.3g/step, %d frames -> %s",
            m["mode_index"],
            m["period"],
            m["growth_rate"],
            n_frames,
            out_path,
        )
        render_mode_video(
            args.dataset,
            m,
            modes[:, m["mode_index"]],
            nx,
            ny,
            args.field,
            dx,
            dy,
            n_frames,
            normalize_growth=not args.include_growth,
            fps=args.fps,
            out_path=out_path,
        )
        log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
