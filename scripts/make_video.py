"""Render an animation of the flow over time.

Fields: `vorticity` (default, out-of-plane vorticity w = du_y/dx - du_x/dy,
same convention as check_vorticity.py), `u_x`, `u_y`, or `speed`
(sqrt(u_x^2 + u_y^2)). Derivatives use the grid spacing from
configs/grid.yaml (--dx/--dy override it); irrelevant for u_x/u_y/speed.

The color scale is fixed across the whole video (from the 1st/99th
percentile of a strided subsample of frames) so brightness is comparable
frame to frame; it is not tied to the train/test split.

Requires no system ffmpeg install: encoding uses the static binary bundled
by the imageio-ffmpeg package.

Usage:
    uv run scripts/make_video.py --dataset re16k_t400_0
    uv run scripts/make_video.py --dataset re16k_t400_0 --field speed --stride 2 --fps 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

from mhd_surrogate.fields import vorticity as compute_vorticity  # noqa: E402
from mhd_surrogate.grid import grid_spacing  # noqa: E402

DEFAULT_STORE = Path("data/processed/re16k_t400.zarr")
DEFAULT_OUT_DIR = Path("reports/videos")
FIELDS = {
    "vorticity": {"cmap": "RdBu_r", "symmetric": True},
    "u_x": {"cmap": "viridis", "symmetric": False},
    "u_y": {"cmap": "viridis", "symmetric": False},
    "speed": {"cmap": "viridis", "symmetric": False},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--dataset", required=True, help="Array name in the zarr store")
    parser.add_argument("--field", choices=sorted(FIELDS), default="vorticity")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: <out-dir>/<dataset>_<field>.mp4",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None, help="Default: all time steps")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth time step")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dx", type=float, default=None, help="Override spacing along axis 2")
    parser.add_argument("--dy", type=float, default=None, help="Override spacing along axis 3")
    parser.add_argument(
        "--color-sample",
        type=int,
        default=30,
        help="Frames sampled to fix the color scale",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=32,
        help="Output frames per zarr read (default: %(default)s)",
    )
    return parser.parse_args()


def compute_field(block: np.ndarray, field: str, dx: float, dy: float) -> np.ndarray:
    """`block` is (t, 2, Nx, Ny); returns (t, Nx, Ny)."""
    if field == "vorticity":
        return compute_vorticity(block, dx, dy)
    if field == "u_x":
        return block[:, 0]
    if field == "u_y":
        return block[:, 1]
    if field == "speed":
        return np.sqrt(block[:, 0] ** 2 + block[:, 1] ** 2)
    raise ValueError(f"unknown field: {field}")


def color_limits(arr, steps: np.ndarray, field: str, dx: float, dy: float) -> tuple[float, float]:
    """1st/99th percentile color limits from a subsample of time steps."""
    sample = compute_field(arr[steps][:], field, dx, dy)
    lo, hi = np.percentile(sample, [1, 99])
    if FIELDS[field]["symmetric"]:
        limit = max(abs(lo), abs(hi))
        return -limit, limit
    return lo, hi


def main() -> None:
    args = parse_args()
    root = zarr.open_group(store=str(args.store), mode="r")
    arr = root[args.dataset]
    if arr.ndim != 4 or arr.shape[1] != 2:
        raise SystemExit(f"expected shape (T, 2, Nx, Ny), got {arr.shape}")

    dx, dy = grid_spacing(arr.shape[2], arr.shape[3])
    dx = args.dx if args.dx is not None else dx
    dy = args.dy if args.dy is not None else dy

    end = args.end if args.end is not None else arr.shape[0]
    steps = np.arange(args.start, end, args.stride)
    if steps.size == 0:
        raise SystemExit(f"no time steps in range [{args.start}, {end}) with stride {args.stride}")

    sample_stride = max(1, len(steps) // args.color_sample)
    vmin, vmax = color_limits(arr, steps[::sample_stride], args.field, dx, dy)
    print(
        f"{args.dataset}: {len(steps)} frames, field={args.field}, "
        f"color range [{vmin:.4g}, {vmax:.4g}]"
    )

    out_path = args.out or (args.out_dir / f"{args.dataset}_{args.field}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 3))
    im = ax.imshow(
        np.zeros((arr.shape[3], arr.shape[2])),
        cmap=FIELDS[args.field]["cmap"],
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, shrink=0.8)
    title = ax.set_title("")
    fig.tight_layout()

    writer = FFMpegWriter(fps=args.fps)
    with writer.saving(fig, str(out_path), dpi=150):
        for i in range(0, len(steps), args.chunk_frames):
            chunk_steps = steps[i : i + args.chunk_frames]
            # steps is a fixed-stride arange, so this slice reproduces it exactly.
            block = arr[chunk_steps[0] : chunk_steps[-1] + 1 : args.stride]
            field_block = compute_field(block, args.field, dx, dy)
            for j, t in enumerate(chunk_steps):
                im.set_data(np.rot90(field_block[j]))
                title.set_text(f"{args.dataset}: {args.field} at t={t}")
                writer.grab_frame()

    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
