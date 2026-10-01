"""Windowed dataset over a zarr-backed velocity time series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import zarr

from mhd_surrogate.data.normalization import Normalizer


@dataclass(frozen=True)
class WindowedDataset:
    """Consecutive-window samples (input, target) drawn from one region of a
    dataset's time series.

    Each sample is `window` consecutive time steps as input and the
    following `horizon` time steps as target, both shape (steps, C, Nx, Ny).
    Samples start every `stride` steps within [start, end).

    Values are returned as stored (float32) unless a `normalizer` is given,
    in which case both input and target are normalized with it (the zarr
    itself always stays raw).
    """

    array: zarr.Array
    start: int
    end: int
    window: int
    horizon: int
    stride: int
    normalizer: Normalizer | None = None

    def __post_init__(self) -> None:
        span = self.window + self.horizon
        if self.end - self.start < span:
            raise ValueError(
                f"region [{self.start}, {self.end}) has only {self.end - self.start} "
                f"steps, need at least {span} for window={self.window} + horizon={self.horizon}"
            )

    def __len__(self) -> int:
        span = self.window + self.horizon
        return (self.end - self.start - span) // self.stride + 1

    def __getitem__(self, i: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample `i`. Unlike a normal Python sequence, negative indices are
        out of range rather than counting from the end.
        """
        if not 0 <= i < len(self):
            raise IndexError(i)
        sample_start = self.start + i * self.stride
        x = self.array[sample_start : sample_start + self.window]
        y = self.array[sample_start + self.window : sample_start + self.window + self.horizon]
        if self.normalizer is not None:
            x, y = self.normalizer(x), self.normalizer(y)
        return jnp.asarray(x), jnp.asarray(y)


def load_full_dataset(
    zarr_store: Path | str,
    dataset: str,
    window: int,
    horizon: int,
    stride: int,
    normalizer: Normalizer | None = None,
) -> WindowedDataset:
    """Build a WindowedDataset over `dataset`'s entire recorded length -- for
    a dataset used wholesale (see configs/data/re16k.yaml's dataset-level
    train/val/test split), where there's no further internal region to look
    up.
    """
    root = zarr.open_group(store=str(zarr_store), mode="r")
    arr = root[dataset]
    return WindowedDataset(arr, 0, arr.shape[0], window, horizon, stride, normalizer)
