"""Sanity-check whether a trailing train/test split looks representative.

Loads the split manifest written by scripts/split_data.py and, for each
dataset, computes the per-timestep spatial mean/std of each channel,
compares aggregate train vs. test statistics, and plots the per-timestep
values over time with the train/test boundary marked -- so a drift, trend,
or regime change concentrated in the held-out tail is visible rather than
hidden inside a single aggregate number.

Usage:
    uv run scripts/check_split.py
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
STD_DIFF_WARN_THRESHOLD = 0.2  # flag mean shifts larger than this many std devs
REL_DIFF_WARN_THRESHOLD = 0.1  # flag std ratio changes larger than this fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--chunk-t", type=int, default=32, help="Time steps per read (default: %(default)s)"
    )
    return parser.parse_args()


def per_timestep_stats(arr, chunk_t: int) -> tuple[np.ndarray, np.ndarray]:
    """Spatial mean and std per time step and channel, shape (T, C)."""
    n_steps, n_channels = arr.shape[0], arr.shape[1]
    means = np.empty((n_steps, n_channels), dtype=np.float64)
    stds = np.empty((n_steps, n_channels), dtype=np.float64)
    for start in range(0, n_steps, chunk_t):
        end = min(start + chunk_t, n_steps)
        block = arr[start:end]
        means[start:end] = block.mean(axis=(2, 3))
        stds[start:end] = block.std(axis=(2, 3))
    return means, stds


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def print_region_comparison(means: np.ndarray, stds: np.ndarray, train_end: int) -> None:
    n_channels = means.shape[1]
    channel_names = CHANNEL_NAMES if n_channels == len(CHANNEL_NAMES) else [
        f"channel={i}" for i in range(n_channels)
    ]
    for c, cname in enumerate(channel_names):
        train_mean = means[:train_end, c].mean()
        test_mean = means[train_end:, c].mean()
        train_std = stds[:train_end, c].mean()
        test_std = stds[train_end:, c].mean()
        # Mean shift relative to the natural fluctuation scale (train std),
        # not relative to the mean itself -- a % diff is meaningless when
        # the baseline mean is near zero (e.g. a zero-mean fluctuation).
        mean_diff_in_std = abs(test_mean - train_mean) / (train_std + 1e-8)
        std_rel_diff = abs(test_std - train_std) / (abs(train_std) + 1e-8)
        flag = (
            " <-- check this"
            if mean_diff_in_std > STD_DIFF_WARN_THRESHOLD or std_rel_diff > REL_DIFF_WARN_THRESHOLD
            else ""
        )
        print(
            f"  {cname}: train mean={train_mean:.4g} std={train_std:.4g} | "
            f"test mean={test_mean:.4g} std={test_std:.4g} | "
            f"mean shift={mean_diff_in_std:.2f} std devs, std rel diff={std_rel_diff:.1%}{flag}"
        )


def plot_over_time(name: str, means: np.ndarray, train_end: int, out_dir: Path) -> Path:
    n_steps, n_channels = means.shape
    channel_names = CHANNEL_NAMES if n_channels == len(CHANNEL_NAMES) else [
        f"channel={i}" for i in range(n_channels)
    ]
    window = max(n_steps // 50, 1)

    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 4 * n_channels), squeeze=False)
    for c in range(n_channels):
        ax = axes[c][0]
        series = means[:, c]
        ax.plot(series, alpha=0.3, label="per-step spatial mean")
        smoothed = rolling_mean(series, window)
        offset = (n_steps - len(smoothed)) // 2
        ax.plot(range(offset, offset + len(smoothed)), smoothed, label=f"rolling mean (w={window})")
        ax.axvline(train_end, color="red", linestyle="--", label="train/test boundary")
        ax.set_title(f"{name}: {channel_names[c]} spatial mean over time")
        ax.set_xlabel("time step")
        ax.legend()

    fig.tight_layout()
    out_path = out_dir / f"{name}_split_check.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = zarr.open_group(store=manifest["config"]["zarr_store"], mode="r")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, split in manifest["splits"].items():
        train_end = split["train"][1]
        means, stds = per_timestep_stats(root[name], args.chunk_t)

        print(f"\n=== {name} (train: [0,{train_end}), test: [{train_end},{split['n_steps']})) ===")
        print_region_comparison(means, stds, train_end)
        out_path = plot_over_time(name, means, train_end, args.out_dir)
        print(f"  plot: {out_path}")


if __name__ == "__main__":
    main()
