"""Render an animation of one or more POD modes' actual time coefficients.

check_pod.py's mode-shape plot shows a single static snapshot (the mode
shape itself, with no time information -- a POD mode has no single
frequency, unlike a DMD mode) of each top mode; this instead animates its
*recorded* contribution to the flow, `coefficients[i, t] * mode_i` frame by
frame, over the dataset's actual recorded snapshots. Unlike make_dmd_video.py, there
is no analytic model to extrapolate from (POD gives no dynamics, just a
decomposition of the given data): frames replay the real, fitted
coefficient, so `--start`/`--end`/`--stride` (same convention as
make_video.py) pick which recorded steps to render, not how many periods of
an oscillation. Re-runs the same POD decomposition as check_pod.py (the
mode vectors and coefficients aren't persisted anywhere -- only summary
statistics are), so this has the same cost: ~6-8 GB and ~20s for the
largest dataset.

By default, renders one video per top `--n-videos` mode (ranked by energy,
matching check_pod.py's mode-shape plot). Pass one or more `--mode-index`
(from check_pod.py's printed output) to render specific modes instead of
the top ones. `--combine` sums the *selected* modes into a single video
instead of rendering one each -- most useful for a near-equal-energy pair
(see check_pod.py's docstring): POD, unlike DMD, can only represent a
single traveling/oscillating structure as two modes in spatial quadrature,
so watching the pair *together* is what actually shows it travel; either
mode alone just pulses in place.

Requires no system ffmpeg install: encoding uses the static binary bundled
by the imageio-ffmpeg package.

Usage:
    uv run scripts/viz/make_pod_video.py --dataset re16k_t400_0
    uv run scripts/viz/make_pod_video.py --dataset re16k_t400_0 \\
        --mode-index 0 --mode-index 1 --combine
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

from mhd_surrogate.analysis.dmd import build_dmd_state  # noqa: E402
from mhd_surrogate.analysis.fields import FIELDS, compute_field  # noqa: E402
from mhd_surrogate.analysis.pod import pod, pod_reconstruction  # noqa: E402
from mhd_surrogate.data.grid import grid_spacing  # noqa: E402
from mhd_surrogate.utils.logging_config import add_log_level_arg, setup_logging  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/analysis/split.yaml")
DEFAULT_OUT_DIR = Path("reports/videos")
CHANNEL_NAMES = ["u_x", "u_y"]
N_VIDEOS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", required=True, help="Dataset name (must be configured)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--field", choices=sorted(FIELDS), default="vorticity")
    parser.add_argument(
        "--n-videos",
        type=int,
        default=N_VIDEOS,
        help="How many top-energy modes to render, if --mode-index is not given",
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
        "--combine",
        action="store_true",
        help="Sum the selected modes into one video instead of one each",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None, help="Default: the dataset's full length")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth time step")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dx", type=float, default=None, help="Override spacing along axis 2")
    parser.add_argument("--dy", type=float, default=None, help="Override spacing along axis 3")
    add_log_level_arg(parser)
    return parser.parse_args()


def render_video(
    name: str,
    label: str,
    frames: np.ndarray,
    steps: np.ndarray,
    nx: int,
    ny: int,
    field: str,
    dx: float,
    dy: float,
    fps: int,
    out_path: Path,
) -> None:
    n_frames = len(steps)
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
        for i, t in enumerate(steps):
            im.set_data(np.rot90(field_block[i]))
            title.set_text(f"{name}: POD {label}, {field} at t={t}")
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
    log.info("running POD on %s (%d steps); this is the expensive part", args.dataset, n_steps)
    data = arr[:n_steps].astype(np.float64)
    state = build_dmd_state(data)
    modes, energy_fraction, coefficients = pod(state)
    log.info("computed %d POD modes", modes.shape[1])

    if args.mode_indices:
        n_modes = modes.shape[1]
        invalid = [i for i in args.mode_indices if not (0 <= i < n_modes)]
        if invalid:
            raise SystemExit(f"mode index/indices {invalid} out of range [0, {n_modes})")
        selected = args.mode_indices
    else:
        selected = list(range(args.n_videos))

    end = args.end if args.end is not None else n_steps
    steps = np.arange(args.start, end, args.stride)
    if steps.size == 0:
        raise SystemExit(f"no time steps in range [{args.start}, {end}) with stride {args.stride}")

    groups = [selected] if args.combine else [[i] for i in selected]
    for group in groups:
        energies = ", ".join(f"{i}: {energy_fraction[i]:.1%}" for i in group)
        label = "+".join(str(i) for i in group)
        log.info("mode(s) %s (energy %s), %d frames", label, energies, len(steps))

        frames = pod_reconstruction(modes, coefficients, group)[:, steps].T  # (n_frames, state_dim)

        out_path = args.out_dir / f"{args.dataset}_pod_mode{label}_{args.field}.mp4"
        render_video(
            args.dataset,
            f"mode {label}",
            frames,
            steps,
            nx,
            ny,
            args.field,
            dx,
            dy,
            args.fps,
            out_path,
        )
        log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
