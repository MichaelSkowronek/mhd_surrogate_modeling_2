"""Descriptive statistics of the train+val region of a trailing split.

Loads the split manifest written by scripts/data/split_data.py and, for each
dataset, computes the per-timestep spatial mean/std/min/max of each channel,
the per-channel kinetic energy proxy (0.5*u^2), and the correlation between
channels, over the train+val region only -- this script (like every
scripts/analysis/*.py script) never reads the held-out test region, since
even just looking at its summary statistics would be data snooping. It plots
the per-timestep values over time and a value histogram per channel, so a
drift or trend within the region is visible rather than hidden inside a
single aggregate number.

This used to compare train against test; now that test is off limits, it
just describes train+val. Once the train/val boundary within this region is
decided, comparing train against val the way this used to compare train
against test belongs here again.

Usage:
    uv run scripts/analysis/check_split.py
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
N_HIST_BINS = 60


def channel_labels(n_channels: int) -> list[str]:
    if n_channels == len(CHANNEL_NAMES):
        return CHANNEL_NAMES
    return [f"channel={i}" for i in range(n_channels)]


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
    add_common_args(parser)
    return parser.parse_args()


def per_timestep_stats(arr, chunk_t: int, n_steps: int | None = None):
    """Spatial mean, std, min and max per time step and channel, shape (T, C).

    Also returns the per-channel kinetic energy proxy 0.5*u^2 (spatial
    mean), shape (T, C); the total is its sum over channels.

    When there are exactly 2 channels (velocity components u_x, u_y), also
    returns the spatial Pearson correlation between them per time step,
    shape (T,) (None otherwise).

    `n_steps` bounds how much of `arr` is read (default: all of it) -- for
    reading only a leading region of a larger array without loading the
    rest.
    """
    n_steps = arr.shape[0] if n_steps is None else n_steps
    n_channels = arr.shape[1]
    means = np.empty((n_steps, n_channels), dtype=np.float64)
    stds = np.empty((n_steps, n_channels), dtype=np.float64)
    mins = np.empty((n_steps, n_channels), dtype=np.float64)
    maxs = np.empty((n_steps, n_channels), dtype=np.float64)
    energy = np.empty((n_steps, n_channels), dtype=np.float64)
    has_velocity_pair = n_channels == 2
    correlation = np.empty(n_steps, dtype=np.float64) if has_velocity_pair else None

    for start in range(0, n_steps, chunk_t):
        end = min(start + chunk_t, n_steps)
        block = arr[start:end]
        means[start:end] = block.mean(axis=(2, 3))
        stds[start:end] = block.std(axis=(2, 3))
        mins[start:end] = block.min(axis=(2, 3))
        maxs[start:end] = block.max(axis=(2, 3))
        energy[start:end] = 0.5 * (block.astype(np.float64) ** 2).mean(axis=(2, 3))

        if has_velocity_pair:
            ux = block[:, 0].reshape(end - start, -1)
            uy = block[:, 1].reshape(end - start, -1)
            ux_centered = ux - ux.mean(axis=1, keepdims=True)
            uy_centered = uy - uy.mean(axis=1, keepdims=True)
            cov = (ux_centered * uy_centered).mean(axis=1)
            correlation[start:end] = cov / (ux.std(axis=1) * uy.std(axis=1) + 1e-12)

    return means, stds, mins, maxs, energy, correlation


def histogram(
    arr,
    train_end: int,
    value_range: np.ndarray,
    n_bins: int,
    chunk_t: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel value histogram of the train+val region, i.e. `arr[:train_end]`.

    Reads the array in chunks bounded by `train_end` rather than loading it
    all at once; never reads at or past `train_end`. Returns (counts,
    bin_edges), each (n_channels, ...).
    """
    n_channels = arr.shape[1]
    bin_edges = np.stack(
        [np.linspace(value_range[c, 0], value_range[c, 1], n_bins + 1) for c in range(n_channels)]
    )
    counts = np.zeros((n_channels, n_bins), dtype=np.int64)

    for start in range(0, train_end, chunk_t):
        end = min(start + chunk_t, train_end)
        block = arr[start:end]
        for c in range(n_channels):
            hist, _ = np.histogram(block[:, c], bins=bin_edges[c])
            counts[c] += hist

    return counts, bin_edges


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def print_stats(means: np.ndarray, stds: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> dict:
    n_channels = means.shape[1]
    summary = {}
    for c, cname in enumerate(channel_labels(n_channels)):
        mean, std = means[:, c].mean(), stds[:, c].mean()
        vmin, vmax = mins[:, c].min(), maxs[:, c].max()
        print(f"  {cname}: mean={mean:.4g} std={std:.4g} min={vmin:.4g} max={vmax:.4g}")
        summary[cname] = {"mean": mean, "std": std, "min": vmin, "max": vmax}
    return summary


def print_scalar_stats(label: str, series: np.ndarray) -> dict:
    mean, std = series.mean(), series.std()
    print(f"  {label}: mean={mean:.4g} std={std:.4g}")
    return {"mean": mean, "std": std}


def plot_series_over_time(name: str, series: dict[str, np.ndarray], out_dir: Path) -> Path:
    n_steps = next(iter(series.values())).shape[0]
    window = max(n_steps // 50, 1)

    fig, axes = plt.subplots(len(series), 1, figsize=(12, 4 * len(series)), squeeze=False)
    for i, (label, values) in enumerate(series.items()):
        ax = axes[i][0]
        ax.plot(values, alpha=0.3, label="per-step value")
        smoothed = rolling_mean(values, window)
        offset = (n_steps - len(smoothed)) // 2
        ax.plot(
            range(offset, offset + len(smoothed)),
            smoothed,
            label=f"rolling mean (w={window})",
        )
        ax.set_title(f"{name}: {label} over time (train+val)")
        ax.set_xlabel("time step")
        ax.legend()

    fig.tight_layout()
    out_path = out_dir / f"{name}_split_check.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_histogram(
    name: str,
    counts: np.ndarray,
    bin_edges: np.ndarray,
    out_dir: Path,
) -> Path:
    n_channels = counts.shape[0]

    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 4 * n_channels), squeeze=False)
    for c, cname in enumerate(channel_labels(n_channels)):
        ax = axes[c][0]
        centers = (bin_edges[c, :-1] + bin_edges[c, 1:]) / 2
        width = bin_edges[c, 1] - bin_edges[c, 0]
        density = counts[c] / (counts[c].sum() * width)
        ax.bar(centers, density, width=width, alpha=0.7)
        ax.set_title(f"{name}: {cname} value distribution (train+val)")
        ax.set_xlabel("value")
        ax.set_ylabel("density")

    fig.tight_layout()
    out_path = out_dir / f"{name}_split_hist.png"
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
        train_end = split["train"][1]
        arr = root[name]
        means, stds, mins, maxs, energy, correlation = per_timestep_stats(
            arr, args.chunk_t, n_steps=train_end
        )

        print(
            f"\n=== {name} (train+val: [0,{train_end}) of {split['n_steps']} total; "
            f"buffer: {split['buffer']}, test: {split['test']} held out) ==="
        )
        channel_summary = print_stats(means, stds, mins, maxs)

        labels = channel_labels(means.shape[1])
        series = {f"{cname} spatial mean": means[:, c] for c, cname in enumerate(labels)}

        energy_series = {
            f"kinetic energy 0.5*{cname}^2, spatial mean": energy[:, c]
            for c, cname in enumerate(labels)
        }
        energy_series["total kinetic energy 0.5*sum(u^2), spatial mean"] = energy.sum(axis=1)
        energy_summary = {
            label: print_scalar_stats(label, values) for label, values in energy_series.items()
        }
        series.update(energy_series)

        correlation_summary = None
        if correlation is not None:
            correlation_summary = print_scalar_stats("u_x-u_y spatial correlation", correlation)
            series["u_x-u_y spatial correlation"] = correlation

        out_path = plot_series_over_time(name, series, args.out_dir)
        print(f"  plot: {out_path}")

        value_range = np.stack([mins.min(axis=0), maxs.max(axis=0)], axis=1)
        counts, bin_edges = histogram(arr, train_end, value_range, N_HIST_BINS, args.chunk_t)
        hist_path = plot_histogram(name, counts, bin_edges, args.out_dir)
        print(f"  histogram: {hist_path}")

        summary_path = write_summary(
            name,
            "check_split",
            {
                "n_steps": split["n_steps"],
                "train_range": split["train"],
                "buffer_range": split["buffer"],
                "test_range": split["test"],
                "channels": channel_summary,
                "energy": energy_summary,
                "correlation": correlation_summary,
            },
            args.summary_dir,
        )
        print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
