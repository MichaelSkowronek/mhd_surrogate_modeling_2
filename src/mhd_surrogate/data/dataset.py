"""Windowed dataset over a split of a zarr-backed velocity time series."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import zarr


@dataclass(frozen=True)
class WindowedDataset:
    """Consecutive-window samples (input, target) drawn from one train/test region.

    Each sample is `window` consecutive time steps as input and the
    following `horizon` time steps as target, both shape (steps, C, Nx, Ny).
    Samples start every `stride` steps within [start, end) and never cross
    the train/test boundary, since that range is a hard start/end from the
    split manifest.

    Values are returned as-is (float32), not normalized -- normalization is
    not implemented yet.
    """

    array: zarr.Array
    start: int
    end: int
    window: int
    horizon: int
    stride: int

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
        return jnp.asarray(x), jnp.asarray(y)


def load_dataset(
    manifest_path: Path,
    dataset: str,
    split: str,
    window: int,
    horizon: int,
    stride: int,
) -> WindowedDataset:
    """Build a WindowedDataset for `dataset`'s `split` region ("train" or "test")."""
    manifest = json.loads(Path(manifest_path).read_text())
    root = zarr.open_group(store=manifest["config"]["zarr_store"], mode="r")
    start, end = manifest["splits"][dataset][split]
    return WindowedDataset(root[dataset], start, end, window, horizon, stride)
