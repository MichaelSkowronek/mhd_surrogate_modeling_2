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

The scoring reads an ensemble's forecast on the device, a block of lead
times at a time (`MemberBlocks`): the members advance through the blocks
together, so the pairwise ensemble scores see every member at the same
lead times while the device holds only a block of each. The forecasts are
the ones `forecast_members` makes, taken from a model's
`predict_device_blocks` where it has one (a JAX model, which then never
copies them to the host) and otherwise from its `predict` calls, copied to
the device a block at a time.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import jax
import numpy as np

from mhd_surrogate.evaluation.quantities import DEFAULT_CHUNK_T, to_device
from mhd_surrogate.models.base import DeviceBlock


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


@dataclass(frozen=True)
class EnsembleBlock:
    """Lead times `first + 1` to `first + steps` of an ensemble's forecast
    on the device: iterating it yields every member's frames for them,
    (steps, C, Nx, Ny), in seed order, a member decoded when it's reached
    (`DeviceBlock.frames`), so a consumer that reads them one by one holds
    as many as it keeps."""

    first: int
    steps: int
    groups: tuple[DeviceBlock, ...]
    frame_shape: tuple[int, ...]

    def __len__(self) -> int:
        return sum(len(group) for group in self.groups)

    def __iter__(self) -> Iterator[jax.Array]:
        expected = (self.steps, *self.frame_shape)
        for group in self.groups:
            for member in range(len(group)):
                frames = group.frames(member)
                if tuple(frames.shape) != expected:
                    raise ValueError(
                        f"model predicted frames of shape {frames.shape}, expected {expected}"
                    )
                yield frames


class MemberBlocks:
    """An ensemble's members under the protocol, the forecasts
    `forecast_members` makes for `seeds`, on the device and `block_steps`
    lead times at a time: iterating it yields an `EnsembleBlock` per block,
    in order, all members advancing together.

    Members are forecast `batch_size` per call where the model can: by its
    `predict_device_blocks` (a stochastic model's with several seeds),
    otherwise as `forecast_members` does, its frames then copied to the
    device a block at a time. With `time_first`, the first member is
    forecast alone and `first_member_seconds` is the wall time of its
    forecast -- the model's calls and its decoding, waited for on the
    device -- without the reading or scoring around it."""

    def __init__(
        self,
        model: ForecastModel,
        series,
        context_steps: int,
        seeds: Iterable[int],
        block_steps: int = DEFAULT_CHUNK_T,
        batch_size: int = 1,
        time_first: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if block_steps < 1:
            raise ValueError(f"block_steps must be >= 1, got {block_steps}")
        check_window(model.window, context_steps)
        n = series.shape[0]
        if not 1 <= context_steps < n:
            split_context(series, context_steps)  # raises with the reason
        seeds = list(seeds)
        if not seeds:
            raise ValueError("no seeds to forecast")
        self.model = model
        # Only the context is read: the targets aren't needed, just their shape.
        self.visible = np.asarray(series[context_steps - model.window : context_steps])
        self.n_steps = n - context_steps
        self.frame_shape = tuple(series.shape[1:])
        self.block_steps = block_steps
        head, rest = (seeds[:1], seeds[1:]) if time_first else ([], seeds)
        # (seeds, sampled in a batch) per call.
        self.groups = ([(head, False)] if head else []) + [
            (rest[i : i + batch_size], batch_size > 1) for i in range(0, len(rest), batch_size)
        ]
        self.time_first = time_first
        self.first_member_seconds = 0.0

    def __len__(self) -> int:
        return -(-self.n_steps // self.block_steps)

    def __iter__(self) -> Iterator[EnsembleBlock]:
        streams = [self._group_blocks(*group) for group in self.groups]
        for first in range(0, self.n_steps, self.block_steps):
            steps = min(self.block_steps, self.n_steps - first)
            blocks = []
            for i, ((group, _), stream) in enumerate(zip(self.groups, streams)):
                start = time.perf_counter()
                block = next(stream, None)
                if block is None or len(block) != len(group) or block.steps != steps:
                    raise ValueError(
                        f"model gave the block from lead {first + 1} as "
                        f"{None if block is None else (len(block), block.steps)}, "
                        f"expected {(len(group), steps)} (members, steps)"
                    )
                if i == 0 and self.time_first:
                    block = DeviceBlock(jax.block_until_ready(block.frames(0))[None])
                    self.first_member_seconds += time.perf_counter() - start
                blocks.append(block)
            yield EnsembleBlock(first, steps, tuple(blocks), self.frame_shape)
        if any(next(stream, None) is not None for stream in streams):
            raise ValueError(f"model gave more blocks than the forecast's {len(self)}")

    def _group_blocks(self, group: Sequence[int], batched: bool) -> Iterator[DeviceBlock]:
        model, n = self.model, self.n_steps
        if hasattr(model, "predict_device_blocks"):
            seeded = {"seeds": list(group)} if is_stochastic(model) else {}
            yield from model.predict_device_blocks(self.visible, n, self.block_steps, **seeded)
            return
        shape = (n, *self.frame_shape)
        if batched and is_stochastic(model) and hasattr(model, "predict_members"):
            frames = _checked(
                model.predict_members(self.visible, n, list(group)), (len(group), *shape)
            )
        else:
            frames = [_checked(predict_member(model, self.visible, n, s), shape) for s in group]
        for start in range(0, n, self.block_steps):
            stop = start + self.block_steps
            yield DeviceBlock(to_device(np.stack([f[start:stop] for f in frames])))


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
