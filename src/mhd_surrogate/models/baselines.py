"""Baselines that bracket every real model.

- `Persistence` repeats the last context frame: the best simple forecast at
  the shortest lead times, worthless once the flow has decorrelated.
- `MeanField` predicts the per-pixel temporal mean of the training data
  ("climatology") at every step: the best constant forecast in pointwise
  error at long lead times, but with no dynamics and no small scales.

A useful model has to beat persistence early and the mean field late, while
keeping the physics diagnostics that both of them fail.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import jax
import numpy as np

from mhd_surrogate.models.base import CHECKPOINT_META

log = logging.getLogger(__name__)

DEFAULT_CHUNK_T = 64


@jax.jit
def _chunk_sum(block: jax.Array) -> jax.Array:
    return block.sum(axis=0)


class MeanField:
    """Per-pixel temporal mean of all training steps, pooled over datasets
    (each step weighs the same, so longer datasets count for more)."""

    name = "mean_field"
    window = 0

    def __init__(self, chunk_t: int = DEFAULT_CHUNK_T) -> None:
        self.chunk_t = chunk_t
        self.mean: np.ndarray | None = None
        self.n_steps = 0
        self.datasets: list[str] = []

    def fit(self, datasets: Mapping[str, Any]) -> None:
        """Stream every dataset `chunk_t` steps at a time.

        Each chunk is summed over time on JAX's default device (the GPU when
        the `gpu` extra is installed) in float32 and accumulated on the host in
        float64, in the given dataset order, so the result doesn't depend on
        how many chunks there are by more than float32 rounding of one chunk.
        """
        total = None
        n_steps = 0
        for name, array in datasets.items():
            for start in range(0, array.shape[0], self.chunk_t):
                block = np.asarray(array[start : start + self.chunk_t], dtype=np.float32)
                chunk = np.asarray(_chunk_sum(block), dtype=np.float64)
                total = chunk if total is None else total + chunk
                n_steps += block.shape[0]
            log.info("mean_field: summed %s (%d steps so far)", name, n_steps)
        if total is None or n_steps == 0:
            raise ValueError("MeanField.fit needs at least one non-empty dataset")
        self.mean = (total / n_steps).astype(np.float32)
        self.n_steps = n_steps
        self.datasets = list(datasets)

    def predict(self, context: np.ndarray, n_steps: int) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("MeanField is not fitted")
        return np.broadcast_to(self.mean, (n_steps, *self.mean.shape))

    def save(self, directory: Path) -> None:
        if self.mean is None:
            raise RuntimeError("MeanField is not fitted")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "mean.npy", self.mean)
        meta = {
            "name": self.name,
            "chunk_t": self.chunk_t,
            "n_steps": self.n_steps,
            "datasets": self.datasets,
        }
        (directory / CHECKPOINT_META).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> Self:
        directory = Path(directory)
        meta = json.loads((directory / CHECKPOINT_META).read_text())
        model = cls(chunk_t=meta["chunk_t"])
        model.mean = np.load(directory / "mean.npy")
        model.n_steps = meta["n_steps"]
        model.datasets = meta["datasets"]
        return model


class Persistence:
    """Repeats the last context frame; nothing to fit."""

    name = "persistence"
    window = 1

    def fit(self, datasets: Mapping[str, Any]) -> None:
        pass

    def predict(self, context: np.ndarray, n_steps: int) -> np.ndarray:
        return np.broadcast_to(context[-1], (n_steps, *context.shape[1:]))

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / CHECKPOINT_META).write_text(json.dumps({"name": self.name}))

    @classmethod
    def load(cls, directory: Path) -> Self:
        return cls()
