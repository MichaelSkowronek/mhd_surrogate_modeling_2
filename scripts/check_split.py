"""Sanity-check whether a trailing train/test split looks representative.

Loads the split manifest written by scripts/split_data.py and, for each
dataset, computes the per-timestep spatial mean/std/min/max of each
channel, compares aggregate train vs. test statistics, plots the
per-timestep values over time with the train/test boundary marked, and
plots a train-vs-test value histogram per channel -- so a drift, trend, or
distribution shift concentrated in the held-out tail is visible rather
than hidden inside a single aggregate number.

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
        "--chunk-t", type=int, default=32, help="Time steps per read (default: %(default)s)"
    )
    return parser.parse_args()


def per_timestep_stats(arr, chunk_t: int):
    """Spatial mean, std, min and max per time step and channel, shape (T, C).

    Also returns the per-channel kinetic energy proxy 0.5*u^2 (spatial
    mean), shape (T, C); the total is its sum over channels.

    When there are exactly 2 channels (velocity components u_x, u_y), also
    returns the spatial Pearson correlation between them per time step,
    shape (T,) (None otherwise).
    """
    n_steps, n_channels = arr.shape[0], arr.shape[1]
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


def train_test_histograms(
    arr, train_end: int, test_start: int, value_range: np.ndarray, n_bins: int, chunk_t: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel value histograms for the train and test regions.

    Reads the array in chunks and accumulates counts rather than loading
    everything at once; a chunk straddling a region boundary is split so each
    region gets the right counts regardless of chunk alignment. Buffer steps
    (train_end <= step < test_start) are excluded from both.
    Returns (train_counts, test_counts, bin_edges), each (n_channels, ...).
    """
    n_steps, n_channels = arr.shape[0], arr.shape[1]
    bin_edges = np.stack(
        [np.linspace(value_range[c, 0], value_range[c, 1], n_bins + 1) for c in range(n_channels)]
    )
    train_counts = np.zeros((n_channels, n_bins), dtype=np.int64)
    test_counts = np.zeros((n_channels, n_bins), dtype=np.int64)

    for start in range(0, n_steps, chunk_t):
        end = min(start + chunk_t, n_steps)
        block = arr[start:end]
        train_n = max(0, min(train_end, end) - start)
        test_from = min(max(test_start - start, 0), block.shape[0])
        for c in range(n_channels):
            if train_n > 0:
                counts, _ = np.histogram(block[:train_n, c], bins=bin_edges[c])
                train_counts[c] += counts
            if test_from < block.shape[0]:
                counts, _ = np.histogram(block[test_from:, c], bins=bin_edges[c])
                test_counts[c] += counts

    return train_counts, test_counts, bin_edges


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def print_region_comparison(
    means: np.ndarray,
    stds: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    train_end: int,
    test_start: int,
) -> None:
    n_channels = means.shape[1]
    for c, cname in enumerate(channel_labels(n_channels)):
        train_mean = means[:train_end, c].mean()
        test_mean = means[test_start:, c].mean()
        train_std = stds[:train_end, c].mean()
        test_std = stds[test_start:, c].mean()
        train_min, train_max = mins[:train_end, c].min(), maxs[:train_end, c].max()
        test_min, test_max = mins[test_start:, c].min(), maxs[test_start:, c].max()
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
            f"  {cname}: train mean={train_mean:.4g} std={train_std:.4g} "
            f"min={train_min:.4g} max={train_max:.4g} | "
            f"test mean={test_mean:.4g} std={test_std:.4g} "
            f"min={test_min:.4g} max={test_max:.4g} | "
            f"mean shift={mean_diff_in_std:.2f} std devs, std rel diff={std_rel_diff:.1%}{flag}"
        )


def print_scalar_comparison(
    label: str, series: np.ndarray, train_end: int, test_start: int
) -> None:
    train_vals, test_vals = series[:train_end], series[test_start:]
    train_mean, test_mean = train_vals.mean(), test_vals.mean()
    train_std, test_std = train_vals.std(), test_vals.std()
    mean_diff_in_std = abs(test_mean - train_mean) / (train_std + 1e-8)
    std_rel_diff = abs(test_std - train_std) / (abs(train_std) + 1e-8)
    flag = (
        " <-- check this"
        if mean_diff_in_std > STD_DIFF_WARN_THRESHOLD or std_rel_diff > REL_DIFF_WARN_THRESHOLD
        else ""
    )
    print(
        f"  {label}: train mean={train_mean:.4g} std={train_std:.4g} | "
        f"test mean={test_mean:.4g} std={test_std:.4g} | "
        f"mean shift={mean_diff_in_std:.2f} std devs, std rel diff={std_rel_diff:.1%}{flag}"
    )


def plot_series_over_time(
    name: str, series: dict[str, np.ndarray], train_end: int, test_start: int, out_dir: Path
) -> Path:
    n_steps = next(iter(series.values())).shape[0]
    window = max(n_steps // 50, 1)

    fig, axes = plt.subplots(len(series), 1, figsize=(12, 4 * len(series)), squeeze=False)
    for i, (label, values) in enumerate(series.items()):
        ax = axes[i][0]
        ax.plot(values, alpha=0.3, label="per-step value")
        smoothed = rolling_mean(values, window)
        offset = (n_steps - len(smoothed)) // 2
        ax.plot(range(offset, offset + len(smoothed)), smoothed, label=f"rolling mean (w={window})")
        if test_start > train_end:
            ax.axvspan(train_end, test_start, color="grey", alpha=0.4, label="buffer")
        ax.axvline(test_start, color="red", linestyle="--", label="train/test boundary")
        ax.set_title(f"{name}: {label} over time")
        ax.set_xlabel("time step")
        ax.legend()

    fig.tight_layout()
    out_path = out_dir / f"{name}_split_check.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_histograms(
    name: str,
    train_counts: np.ndarray,
    test_counts: np.ndarray,
    bin_edges: np.ndarray,
    out_dir: Path,
) -> Path:
    n_channels = train_counts.shape[0]

    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 4 * n_channels), squeeze=False)
    for c, cname in enumerate(channel_labels(n_channels)):
        ax = axes[c][0]
        centers = (bin_edges[c, :-1] + bin_edges[c, 1:]) / 2
        width = bin_edges[c, 1] - bin_edges[c, 0]
        # Normalize to a density so train/test are comparable despite the
        # different number of time steps in each region.
        train_density = train_counts[c] / (train_counts[c].sum() * width)
        test_density = test_counts[c] / (test_counts[c].sum() * width)
        ax.bar(centers, train_density, width=width, alpha=0.5, label="train")
        ax.bar(centers, test_density, width=width, alpha=0.5, label="test")
        ax.set_title(f"{name}: {cname} value distribution, train vs. test")
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.legend()

    fig.tight_layout()
    out_path = out_dir / f"{name}_split_hist.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = zarr.open_group(store=manifest["config"]["zarr_store"], mode="r")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, split in manifest["splits"].items():
        train_end, test_start = split["train"][1], split["test"][0]
        arr = root[name]
        means, stds, mins, maxs, energy, correlation = per_timestep_stats(arr, args.chunk_t)

        print(
            f"\n=== {name} (train: [0,{train_end}), buffer: [{train_end},{test_start}), "
            f"test: [{test_start},{split['n_steps']})) ==="
        )
        print_region_comparison(means, stds, mins, maxs, train_end, test_start)

        labels = channel_labels(means.shape[1])
        series = {f"{cname} spatial mean": means[:, c] for c, cname in enumerate(labels)}

        energy_series = {
            f"kinetic energy 0.5*{cname}^2, spatial mean": energy[:, c]
            for c, cname in enumerate(labels)
        }
        energy_series["total kinetic energy 0.5*sum(u^2), spatial mean"] = energy.sum(axis=1)
        for label, values in energy_series.items():
            print_scalar_comparison(label, values, train_end, test_start)
        series.update(energy_series)

        if correlation is not None:
            print_scalar_comparison(
                "u_x-u_y spatial correlation", correlation, train_end, test_start
            )
            series["u_x-u_y spatial correlation"] = correlation

        out_path = plot_series_over_time(name, series, train_end, test_start, args.out_dir)
        print(f"  plot: {out_path}")

        value_range = np.stack([mins.min(axis=0), maxs.max(axis=0)], axis=1)
        train_counts, test_counts, bin_edges = train_test_histograms(
            arr, train_end, test_start, value_range, N_HIST_BINS, args.chunk_t
        )
        hist_path = plot_histograms(name, train_counts, test_counts, bin_edges, args.out_dir)
        print(f"  histogram: {hist_path}")


if __name__ == "__main__":
    main()
