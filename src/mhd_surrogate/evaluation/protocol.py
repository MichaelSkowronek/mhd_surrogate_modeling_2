"""The forecast-evaluation protocol shared by every model.

A surrogate is evaluated by forecasting a held-out dataset's whole series.
The first `context_steps` steps are context only: the model may look at (the
last `window` of) them but they are never scored, so every model -- whatever
its window, or none -- is scored on exactly the same targets, steps
`context_steps` onward.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ForecastModel(Protocol):
    """What the protocol needs from a model.

    `window` is how many of the most recent context steps it reads (0 for a
    model that ignores the context, like a climatological mean).
    `predict(context, n_steps)` returns `n_steps` forecast frames, shape
    (n_steps, *context.shape[1:]), given a context of shape (window, ...).
    """

    window: int

    def predict(self, context: np.ndarray, n_steps: int) -> np.ndarray: ...


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


def forecast(model: ForecastModel, series, context_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Run `model` under the protocol: returns (prediction, targets), both of
    shape (T - context_steps, ...).

    The model sees only the last `model.window` context steps; its prediction
    must have the targets' shape.
    """
    check_window(model.window, context_steps)
    context, targets = split_context(series, context_steps)
    visible = np.asarray(context[context_steps - model.window :])
    prediction = np.asarray(model.predict(visible, targets.shape[0]))
    if prediction.shape != targets.shape:
        raise ValueError(
            f"model predicted shape {prediction.shape}, expected the targets' {targets.shape}"
        )
    return prediction, np.asarray(targets)


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
