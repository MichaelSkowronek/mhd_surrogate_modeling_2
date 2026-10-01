"""Per-channel normalization statistics, computed from the training datasets.

The statistics are written once (by a script) and read back later by other
processes (training, evaluation, inference), so they're a data contract that
crosses a process boundary: a stale file, a different channel order or stats
that include the test dataset would silently give wrong results. `NormalizationStats`
validates that contract at load time instead.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CHUNK_T = 32

# How the std is applied: one per channel, or a single scalar shared by all
# channels (the RMS of the per-channel stds).
StdMode = Literal["per_channel", "shared"]

# Running moments of one channel set: (count, mean, M2) with M2 the sum of
# squared deviations from the mean, each mean/M2 of shape (C,).
Moments = tuple[int, np.ndarray, np.ndarray]


def merge_moments(a: Moments, b: Moments) -> Moments:
    """Combine two sets of running moments (Chan et al.'s parallel update)."""
    n_a, mean_a, m2_a = a
    n_b, mean_b, m2_b = b
    n = n_a + n_b
    if n == 0:
        return a
    delta = mean_b - mean_a
    mean = mean_a + delta * (n_b / n)
    m2 = m2_a + m2_b + delta**2 * (n_a * n_b / n)
    return n, mean, m2


def channel_moments(arr: Any, chunk_t: int = DEFAULT_CHUNK_T) -> Moments:
    """Streaming per-channel count/mean/M2 of a (T, C, H, W) array.

    Reads `chunk_t` time steps at a time (so a zarr array is never fully
    loaded into memory) and accumulates in float64 regardless of the stored
    dtype. `count` is the number of values per channel (T * H * W).
    """
    if chunk_t < 1:
        raise ValueError(f"chunk_t must be >= 1, got {chunk_t}")
    n_channels = arr.shape[1]
    moments: Moments = (0, np.zeros(n_channels), np.zeros(n_channels))
    for t0 in range(0, arr.shape[0], chunk_t):
        block = np.asarray(arr[t0 : t0 + chunk_t], dtype=np.float64)
        count = block.shape[0] * block.shape[2] * block.shape[3]
        mean = block.mean(axis=(0, 2, 3))
        m2 = ((block - mean[None, :, None, None]) ** 2).sum(axis=(0, 2, 3))
        moments = merge_moments(moments, (count, mean, m2))
    return moments


class NormalizationStats(BaseModel):
    """Per-channel mean/std computed from `source_datasets` (population std)."""

    model_config = ConfigDict(frozen=True)

    channel_names: list[str]
    mean: list[float]
    std: list[float]
    n_samples: int = Field(gt=0, description="Values per channel the statistics are based on")
    source_datasets: list[str]

    @model_validator(mode="after")
    def _check_consistent(self) -> NormalizationStats:
        n = len(self.channel_names)
        if n == 0:
            raise ValueError("channel_names must not be empty")
        if len(set(self.channel_names)) != n:
            raise ValueError(f"duplicate channel_names: {self.channel_names}")
        if len(self.mean) != n or len(self.std) != n:
            raise ValueError(
                f"mean ({len(self.mean)}) and std ({len(self.std)}) must each have one "
                f"entry per channel ({n})"
            )
        if not all(math.isfinite(v) for v in self.mean):
            raise ValueError(f"mean must be finite, got {self.mean}")
        if not all(math.isfinite(v) and v > 0 for v in self.std):
            raise ValueError(f"std must be finite and > 0, got {self.std}")
        if not self.source_datasets:
            raise ValueError("source_datasets must not be empty")
        if len(set(self.source_datasets)) != len(self.source_datasets):
            raise ValueError(f"duplicate source_datasets: {self.source_datasets}")
        return self

    def effective_std(self, std_mode: StdMode) -> list[float]:
        """The std actually divided by, one entry per channel.

        `per_channel` equalizes how much each channel contributes to an MSE
        loss; `shared` keeps the channels' relative amplitudes (a uniform
        rescaling), so the loss stays proportional to physical kinetic-energy
        error. The shared scalar is the RMS of the per-channel stds.
        """
        if std_mode == "per_channel":
            return list(self.std)
        if std_mode == "shared":
            rms = math.sqrt(sum(v**2 for v in self.std) / len(self.std))
            return [rms] * len(self.std)
        raise ValueError(f"unknown std_mode {std_mode!r}, expected 'per_channel' or 'shared'")

    def check_compatible(
        self,
        channel_names: Sequence[str],
        train_datasets: Sequence[str],
        test_dataset: str,
    ) -> None:
        """Raise ValueError unless these stats match the current setup: same
        channels in the same order, computed from exactly the training
        datasets, and never from the held-out test dataset.
        """
        if test_dataset in self.source_datasets:
            raise ValueError(f"stats were computed using the test dataset {test_dataset!r}")
        if list(channel_names) != self.channel_names:
            raise ValueError(
                f"channel mismatch: stats have {self.channel_names}, expected {list(channel_names)}"
            )
        if set(train_datasets) != set(self.source_datasets):
            raise ValueError(
                f"stats were computed from {sorted(self.source_datasets)}, but the "
                f"training datasets are {sorted(train_datasets)}"
            )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> NormalizationStats:
        return cls.model_validate_json(Path(path).read_text())


def stats_from_moments(
    moments: Moments, channel_names: Sequence[str], source_datasets: Sequence[str]
) -> NormalizationStats:
    """Turn pooled running moments into a validated NormalizationStats."""
    n, mean, m2 = moments
    return NormalizationStats(
        channel_names=list(channel_names),
        mean=mean.tolist(),
        std=np.sqrt(m2 / n).tolist() if n else [],
        n_samples=n,
        source_datasets=list(source_datasets),
    )


def compute_normalization_stats(
    arrays: Mapping[str, Any],
    channel_names: Sequence[str],
    chunk_t: int = DEFAULT_CHUNK_T,
) -> NormalizationStats:
    """Pool per-channel mean/std over every (T, C, H, W) array in `arrays`
    (dataset name -> array). Pass the training datasets only.
    """
    pooled: Moments = (0, np.zeros(len(channel_names)), np.zeros(len(channel_names)))
    for arr in arrays.values():
        if arr.shape[1] != len(channel_names):
            raise ValueError(f"array has {arr.shape[1]} channels, expected {len(channel_names)}")
        pooled = merge_moments(pooled, channel_moments(arr, chunk_t))
    return stats_from_moments(pooled, channel_names, list(arrays))


@dataclass(frozen=True)
class Normalizer:
    """Applies (and inverts) `(x - mean) / std` on arrays shaped (..., C, H, W).

    Built from a NormalizationStats and a std mode. Works on numpy and jax
    arrays alike and keeps float32 inputs float32.
    """

    mean: np.ndarray  # (C, 1, 1), float32
    std: np.ndarray  # (C, 1, 1), float32

    @classmethod
    def from_stats(cls, stats: NormalizationStats, std_mode: StdMode) -> Normalizer:
        return cls(
            mean=np.asarray(stats.mean, dtype=np.float32).reshape(-1, 1, 1),
            std=np.asarray(stats.effective_std(std_mode), dtype=np.float32).reshape(-1, 1, 1),
        )

    def __call__(self, x: Any) -> Any:
        return (x - self.mean) / self.std

    def inverse(self, x: Any) -> Any:
        return x * self.std + self.mean
