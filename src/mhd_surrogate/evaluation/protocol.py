"""The forecast-evaluation protocol shared by every model.

A surrogate is evaluated by forecasting a held-out dataset's whole series.
The first `context_steps` steps are context only: the model may look at (the
last `window` of) them but they are never scored, so every model -- whatever
its window, or none -- is scored on exactly the same targets, steps
`context_steps` onward.

A stochastic model (a generative one, `stochastic = True`) forecasts one
sample per `seed`; an ensemble is the samples for seeds 0, 1, ...
(`forecast_members`), and `forecast` is the member for seed 0. A
deterministic model has one answer and takes no seed. A stochastic model
may also sample several members in one call (`predict_members`), which a
GPU with room to spare turns into throughput; `forecast_members` uses it
when asked to batch.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol

import numpy as np


class ForecastModel(Protocol):
    """What the protocol needs from a model.

    `window` is how many of the most recent context steps it reads (0 for a
    model that ignores the context, like a climatological mean).
    `predict(context, n_steps)` returns `n_steps` forecast frames, shape
    (n_steps, *context.shape[1:]), given a context of shape (window, ...).
    A model with `stochastic = True` also takes `seed=` and returns the
    sample for it (call it through `predict_member`), and may have
    `predict_members(context, n_steps, seeds)`, shape (len(seeds), n_steps,
    ...): the samples for `seeds` in one call. Member k must be the sample
    for `seeds[k]` whatever else is in the batch -- its randomness drawn from
    its own seed, not split from a key shared by the batch -- so it equals
    `predict(..., seed=seeds[k])` up to float rounding (GPU kernels can
    differ by batch shape, and a long chaotic rollout amplifies that).
    """

    window: int

    def predict(self, context: np.ndarray, n_steps: int) -> np.ndarray: ...


def is_stochastic(model) -> bool:
    return bool(getattr(model, "stochastic", False))


def predict_member(model: ForecastModel, context: np.ndarray, n_steps: int, seed: int = 0):
    """`model.predict(context, n_steps)`, passing `seed` on to a stochastic
    model (the sample for that seed); a deterministic model ignores it."""
    if is_stochastic(model):
        return model.predict(context, n_steps, seed=seed)
    return model.predict(context, n_steps)


def split_context(series, context_steps: int):
    """(context, targets) = the first `context_steps` steps and the rest.

    Works on anything sliceable along axis 0 (a NumPy or zarr array); the
    series must have at least one scored step.
    """
    n = series.shape[0]
    if context_steps < 1:
        raise ValueError(f"context_steps must be >= 1, got {context_steps}")
    if n <= context_steps:
        raise ValueError(f"series has {n} steps, need more than context_steps={context_steps}")
    return series[:context_steps], series[context_steps:]


def check_window(window: int, context_steps: int) -> None:
    """A model reading more than the context would see scored targets."""
    if not 0 <= window <= context_steps:
        raise ValueError(f"window={window} must be between 0 and context_steps={context_steps}")


def forecast(
    model: ForecastModel, series, context_steps: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Run `model` under the protocol: returns (prediction, targets), both of
    shape (T - context_steps, ...).

    The model sees only the last `model.window` context steps; its prediction
    must have the targets' shape. A stochastic model's prediction is its
    sample for `seed`.
    """
    check_window(model.window, context_steps)
    context, targets = split_context(series, context_steps)
    visible = np.asarray(context[context_steps - model.window :])
    prediction = _checked(predict_member(model, visible, targets.shape[0], seed), targets.shape)
    return prediction, np.asarray(targets)


def ensemble_size(model: ForecastModel, n_members: int) -> int:
    """How many members an ensemble of `n_members` really has: a
    deterministic model has a single answer, so one."""
    if n_members < 1:
        raise ValueError(f"n_members must be >= 1, got {n_members}")
    return n_members if is_stochastic(model) else 1


def forecast_members(
    model: ForecastModel,
    series,
    context_steps: int,
    seeds: Iterable[int],
    batch_size: int = 1,
) -> Iterator[np.ndarray]:
    """Ensemble members under the protocol, one at a time (an 837-step
    member is ~1 GB): the prediction for each of `seeds`, as `forecast`
    makes it.

    With `batch_size` > 1, a model that has `predict_members` samples up to
    that many members per call; one that hasn't falls back to one call per
    member. Members are still yielded one at a time, so a caller holds one
    batch at most. Batching is for throughput, but since it can change the
    rounding, a batch size is part of what reproduces an ensemble's scores.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    check_window(model.window, context_steps)
    n = series.shape[0]
    if not 1 <= context_steps < n:
        split_context(series, context_steps)  # raises with the reason
    # Only the context is read: the targets aren't needed, just their shape.
    visible = np.asarray(series[context_steps - model.window : context_steps])
    shape = (n - context_steps, *series.shape[1:])
    seeds = list(seeds)
    if batch_size == 1 or not hasattr(model, "predict_members"):
        for seed in seeds:
            yield _checked(predict_member(model, visible, shape[0], seed), shape)
        return
    for start in range(0, len(seeds), batch_size):
        batch = seeds[start : start + batch_size]
        yield from _checked(model.predict_members(visible, shape[0], batch), (len(batch), *shape))


def _checked(prediction, shape: tuple[int, ...]) -> np.ndarray:
    prediction = np.asarray(prediction)
    if prediction.shape != shape:
        raise ValueError(f"model predicted shape {prediction.shape}, expected {shape}")
    return prediction


def check_readable(name: str, data_config: dict) -> None:
    """For a script outside the training run (a video, a rollout check):
    `name` must be the validation dataset or a training one (`data_config`
    is `configs/data/re16k.yaml`). Never the test dataset: it is read once,
    by the final evaluation."""
    if name == data_config["test_dataset"]:
        raise ValueError(
            f"{name} is the test dataset: it is read once, by the final evaluation, not here"
        )
    allowed = [data_config["val_dataset"], *data_config["train_datasets"]]
    if name not in allowed:
        raise ValueError(f"{name} is not a validation or training dataset ({allowed})")
