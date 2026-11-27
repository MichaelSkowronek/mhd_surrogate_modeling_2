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
(`evaluation.protocol.forecast_members`). It may also implement
`predict_members(context, n_steps, seeds)` to sample several members in one
call (throughput on a GPU); member k must be the sample for `seeds[k]`
whatever else is in the batch (see `evaluation.protocol.ForecastModel`). A
deterministic model has one answer: it takes no seed and is scored as a
one-member ensemble.

A forecast too long to hold at once (a long rollout, a served forecast) is
streamed in blocks by `evaluation.rollout.rollout_blocks`, which restarts
the model at every block from the last `window` frames of the forecast so
far. That's the same forecast only if those frames are all of the model's
state. A model with more -- a hidden state, a reduced state it projects its
input onto, a random key that advances with the rollout -- implements
`predict_blocks(context, n_steps, block_steps)` (a stochastic one with
`seed=`): the frames its `predict` call would return, yielded `block_steps`
at a time (the last block may be shorter), up to float rounding (a GPU
compiles each block length separately).

A model that forecasts on a JAX device may also implement
`predict_device_blocks(context, n_steps, block_steps)` (a stochastic one
with `seeds=`, a list): the same blocks, each a `DeviceBlock` that leaves
the frames on the device, where the scoring reduces them
(`evaluation.protocol.MemberBlocks`), for every seed in one call. Its
forecast is `predict`'s (`predict_members`' for several seeds), up to the
rounding of its last step on the device, e.g. a latent model's decoding:
the scores of a model are those of the forecast it serves. One without it
is scored on its `predict` frames, copied to the device.

Such a model may also implement `predict_device_batch(contexts, n_steps,
block_steps)` (a stochastic one with `seeds=`, a list of seeds per
context): the forecasts from several contexts in one call, as
`DeviceBlock`s of all their trajectories, context by context. It is for
throughput, the multi-start validations during training
(`evaluation.protocol.StartBlocks`); batched, a forecast can round
differently from its own call's, so the scoring forecasts each one alone.

A model trained iteratively (a neural network) sets `iterative = True` and
takes `fit(datasets, hooks)`: the `FitHooks` give it what a closed-form fit
doesn't need -- a validation score to early-stop on, somewhere to log its
training curves, and a directory for resumable training state. `fit_model`
calls either kind the right way.

`parameter_count()` answers "how many parameters does the model have": the
real numbers it learned from the training data and forecasts with -- a
network's weights, DMD's modes and dynamics, the mean field's field --
complex entries counting twice (`real_size`). Normalization statistics (the
mean and scale a model normalizes its input with, the neural models'
constant-pixel values) don't count, as a network's input normalization
usually doesn't. `training/run.py` logs it with every fitted model.
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

    def parameter_count(self) -> int:
        """How many numbers the fitted model learned (see the module docstring)."""

    def save(self, directory: Path) -> None:
        """Write the checkpoint into `directory` (created if missing)."""

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Rebuild the fitted model from a checkpoint written by `save`."""


def _identity(states: Any) -> Any:
    return states


@dataclass(frozen=True)
class DeviceBlock:
    """One block of a batch of members' forecasts on the device
    (`predict_device_blocks`): `states`, (members, t, ...), and `decode`,
    which maps one member's states, (t, ...), to its frames,
    (t, C, Nx, Ny). For a model forecasting frames they are the frames and
    `decode` is the identity; a latent model's are its latents, decoded
    one member at a time when read (`frames`), so a batch of many members
    never has all its frames on the device at once."""

    states: Any
    decode: Callable[[Any], Any] = _identity

    def __len__(self) -> int:
        return len(self.states)

    @property
    def steps(self) -> int:
        return self.states.shape[1]

    def frames(self, member: int) -> Any:
        return self.decode(self.states[member])


def real_size(array: np.ndarray) -> int:
    """The entries of `array` as real numbers: a complex entry is two."""
    return array.size * (2 if np.iscomplexobj(array) else 1)


def _discard(metrics: dict[str, float], step: int) -> None:
    pass


def _ignore(scores: dict[str, float]) -> None:
    pass


@dataclass(frozen=True)
class FitHooks:
    """What the training entry point gives an iterative model's `fit`.

    - `validate(model, index)`: scores the model in its current state on
      the validation data, `index` the validation's in the run (0 for the
      first; what a validation's rollouts draw new seeds from); must
      include `selection_score` (higher is better,
      `evaluation.metrics.selection_score`), and with `neighbors` above 0
      also `rollout_stable_steps` and `rollout_steps`
      (`evaluation.selection.selection_scores` gives them). Early stopping
      keeps the validation `evaluation.selection.RunSelection` ranks best,
      each judged with its `neighbors` nearest validations either side and
      the usable fraction of their rollouts against `usable_threshold`.
      None: no early stopping, the last state is kept.
    - `report(scores)`: called after every validation with its scores plus
      the run's best so far (`best_selection_score`, `best_epoch` and, with
      `neighbors` above 0, `best_usable_fraction`), e.g. for Ray Tune.
    - `log_metrics(metrics, step)`: training curves (loss, learning rate,
      validation scores during training), e.g. `mlflow.log_metrics`.
    - `state_dir`: where resumable training state is kept; a `fit` that finds
      state there continues from it. None: not resumable.
    """

    validate: Callable[[Any, int], dict[str, float]] | None = None
    neighbors: int = 0
    usable_threshold: float = 0.8
    report: Callable[[dict[str, float]], None] = _ignore
    log_metrics: Callable[[dict[str, float], int], None] = _discard
    state_dir: Path | None = None


def fit_model(model: SurrogateModel, datasets: Mapping[str, Any], hooks: FitHooks) -> None:
    """`model.fit(datasets, hooks)` for an iterative model, else `model.fit(datasets)`."""
    if getattr(model, "iterative", False):
        model.fit(datasets, hooks)
    else:
        model.fit(datasets)
