"""The interface every surrogate model implements.

A model is fitted on the training datasets, saved as a checkpoint directory
and scored by `mhd_surrogate.evaluation.protocol.forecast`, which only needs
`window` and `predict`. Inputs and outputs are raw (unnormalized) velocity
fields of shape (T, C, Nx, Ny); a model that wants normalized data applies
the normalizer itself, so every model is scored in the same physical units.

A checkpoint is a directory holding `model.json` (the registry name plus the
model's own metadata) and whatever arrays the model needs; `registry.load_model`
reads the name and dispatches to the right class.

A generative model sets `stochastic = True` and takes
`predict(context, n_steps, seed=...)`: each seed gives one sample of the
forecast distribution, and an ensemble is the samples for seeds 0, 1, ...
(`evaluation.protocol.forecast_members`). A deterministic model has one
answer: it takes no seed and is scored as a one-member ensemble.

A model trained iteratively (a neural network) sets `iterative = True` and
takes `fit(datasets, hooks)`: the `FitHooks` give it what a closed-form fit
doesn't need -- a validation score to early-stop on, somewhere to log its
training curves, and a directory for resumable training state. `fit_model`
calls either kind the right way.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
        """`n_steps` forecast frames from the last `window` context frames
        (a stochastic model also takes `seed=`, see the module docstring)."""

    def save(self, directory: Path) -> None:
        """Write the checkpoint into `directory` (created if missing)."""

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Rebuild the fitted model from a checkpoint written by `save`."""


def _discard(metrics: dict[str, float], step: int) -> None:
    pass


@dataclass(frozen=True)
class FitHooks:
    """What the training entry point gives an iterative model's `fit`.

    - `validate(model)`: scores the model in its current state on the
      validation data; must include `selection_score` (higher is better,
      `evaluation.metrics.selection_score`), which early stopping maximizes.
      None: no early stopping, the last state is kept.
    - `log_metrics(metrics, step)`: training curves (loss, learning rate,
      validation scores during training), e.g. `mlflow.log_metrics`.
    - `state_dir`: where resumable training state is kept; a `fit` that finds
      state there continues from it. None: not resumable.
    """

    validate: Callable[[Any], dict[str, float]] | None = None
    log_metrics: Callable[[dict[str, float], int], None] = _discard
    state_dir: Path | None = None


def fit_model(model: SurrogateModel, datasets: Mapping[str, Any], hooks: FitHooks) -> None:
    """`model.fit(datasets, hooks)` for an iterative model, else `model.fit(datasets)`."""
    if getattr(model, "iterative", False):
        model.fit(datasets, hooks)
    else:
        model.fit(datasets)
