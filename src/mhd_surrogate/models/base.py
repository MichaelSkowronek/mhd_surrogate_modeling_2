"""The interface every surrogate model implements.

A model is fitted on the training datasets, saved as a checkpoint directory
and scored by `mhd_surrogate.evaluation.protocol.forecast`, which only needs
`window` and `predict`. Inputs and outputs are raw (unnormalized) velocity
fields of shape (T, C, Nx, Ny); a model that wants normalized data applies
the normalizer itself, so every model is scored in the same physical units.

A checkpoint is a directory holding `model.json` (the registry name plus the
model's own metadata) and whatever arrays the model needs; `registry.load_model`
reads the name and dispatches to the right class.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, Self

import numpy as np

CHECKPOINT_META = "model.json"


class SurrogateModel(Protocol):
    name: str
    window: int

    def fit(self, datasets: Mapping[str, Any]) -> None:
        """Fit on the training datasets, {name: array of shape (T, C, Nx, Ny)}
        (zarr arrays are fine: models read them in chunks)."""

    def predict(self, context: np.ndarray, n_steps: int) -> np.ndarray:
        """`n_steps` forecast frames from the last `window` context frames."""

    def save(self, directory: Path) -> None:
        """Write the checkpoint into `directory` (created if missing)."""

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Rebuild the fitted model from a checkpoint written by `save`."""
