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

The skill is measured from many starts of a dataset, not only from the end
of its context (`skill_starts`, CLAUDE.md "Data analysis"): every 20 frames
from there, each forecast for the same number of lead times and scored
against the truth at the same frames. `StartBlocks` streams those
forecasts like `MemberBlocks`, a start's members advancing together; a
model with `predict_device_batch` forecasts several starts in one call,
for throughput during training. Every start's members have seeds of their
own (`start_seeds`), and the validations during training other seeds than
the scoring.
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


def skill_starts(n_frames: int, context_steps: int, stride: int, leads: int) -> list[int]:
    """The frames the multi-start skill forecasts from: every `stride`
    frames from `context_steps` (the end of the context, the first scored
    frame), as many as are followed by `leads` frames of truth. On the
    validation dataset (917 frames, context 80, stride 20, 237 leads) that
    is 31 starts, frames 80 to 680, the last one's forecast ending with the
    dataset. A rule rather than a list, so it gives the test dataset its
    starts without it being read before the final evaluation."""
    for name, value in (("context_steps", context_steps), ("stride", stride), ("leads", leads)):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    starts = list(range(context_steps, n_frames - leads + 1, stride))
    if not starts:
        raise ValueError(
            f"a series of {n_frames} frames has no start for {leads} leads: the first, "
            f"frame {context_steps}, needs {context_steps + leads} frames"
        )
    return starts


# A stochastic model's seeds for the multi-start skill (`start_seeds`): a
# range of SEEDS_PER_START per start, the scoring's members from its first
# and the validations' during training from MONITOR_SEEDS on.
SEEDS_PER_START = 1000
MONITOR_SEEDS = 500


def start_seeds(n_starts: int, n_members: int, monitor: bool = False) -> list[list[int]]:
    """The seeds of `n_members` members at each of `n_starts` starts, in
    `skill_starts`' order: member m of start k draws `SEEDS_PER_START * k +
    m` when scored, so start 0, the end of the context, has seeds 0, 1, ...
    (the ones served by default, checked by the gate), and `SEEDS_PER_START
    * k + MONITOR_SEEDS + m` at a validation during training (`monitor`),
    the same at every validation.

    Each start its own: a stochastic model draws a step's noise from the
    seed and the step's index, so a seed shared by the starts would replay
    one noise sequence from every start, and the starts would not be the
    independent draws their average and its bootstrap take them for (run
    1's rollout failures clustered by seed, not by start). And the
    validations' own: early stopping keeps the epoch whose members look
    best, so scoring those members again would flatter it (the kept
    flow-matching runs' selection seeds scored 0.7-1.2 sd better at lead 10
    than fresh ones, 2026-11-25)."""
    first, room = (
        (MONITOR_SEEDS, SEEDS_PER_START - MONITOR_SEEDS) if monitor else (0, MONITOR_SEEDS)
    )
    if n_starts < 1:
        raise ValueError(f"n_starts must be >= 1, got {n_starts}")
    if not 1 <= n_members <= room:
        who = "a validation" if monitor else "the scoring"
        raise ValueError(f"{n_members} members per start: {who} has seeds for 1 to {room}")
    return [[SEEDS_PER_START * k + first + m for m in range(n_members)] for k in range(n_starts)]


def rollout_seed(start_index: int, members: int, validation: int) -> int:
    """The seed of validation `validation`'s rollout from the
    `start_index`-th start (`skill_starts`' order): in that start's range
    for the validations (`start_seeds`, `monitor`), past its `members`
    skill members, and a new one at every validation, so that the rollouts
    of the validations around an epoch are independent draws: failures
    follow the seed (a step's noise comes from the seed and the step's
    index; run 1's rollouts failed by seed, not by start), and with one
    seed at every validation the usable fraction would measure that seed.
    The skill's members keep theirs, so epochs compare on the same draws."""
    room = SEEDS_PER_START - MONITOR_SEEDS - members
    if start_index < 0 or validation < 0:
        raise ValueError(
            f"start_index and validation must be >= 0, got {start_index}, {validation}"
        )
    if room < 1:
        raise ValueError(f"{members} members leave no validation seeds for rollouts")
    return SEEDS_PER_START * start_index + MONITOR_SEEDS + members + validation % room


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
    as many as it keeps. `groups` are the calls' blocks the members are in
    (a `DeviceBlock`, or some of its members: `MemberGroup`)."""

    first: int
    steps: int
    groups: tuple[DeviceBlock | MemberGroup, ...]
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


@dataclass(frozen=True)
class MemberGroup:
    """Some of a `DeviceBlock`'s members, by index: one start's, of a call
    that forecast several starts (`StartBlocks`)."""

    block: DeviceBlock
    indices: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.indices)

    def frames(self, member: int):
        return self.block.frames(self.indices[member])


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
        _check_streaming(block_steps, batch_size)
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
        streams = [
            _member_blocks(
                self.model,
                self.visible,
                self.n_steps,
                self.block_steps,
                group,
                batched,
                self.frame_shape,
            )
            for group, batched in self.groups
        ]
        for first in range(0, self.n_steps, self.block_steps):
            steps = min(self.block_steps, self.n_steps - first)
            blocks = []
            for i, ((group, _), stream) in enumerate(zip(self.groups, streams)):
                start = time.perf_counter()
                block = _next_block(stream, len(group), first, steps)
                if i == 0 and self.time_first:
                    block = DeviceBlock(jax.block_until_ready(block.frames(0))[None])
                    self.first_member_seconds += time.perf_counter() - start
                blocks.append(block)
            yield EnsembleBlock(first, steps, tuple(blocks), self.frame_shape)
        _check_exhausted(streams, len(self))


@dataclass(frozen=True)
class StartsBlock:
    """Lead times `first + 1` to `first + steps` of the forecasts from
    several starts (`StartBlocks`): `ensembles` maps the index of each
    start it covers (into `StartBlocks.starts`) to that start's members, an
    `EnsembleBlock`."""

    first: int
    steps: int
    ensembles: dict[int, EnsembleBlock]


class StartBlocks:
    """Forecasts of `n_steps` lead times from each of `starts`, frames of
    `series` (a model sees the `window` frames before its start), each an
    ensemble of its own `seeds` (a list per start, `start_seeds`; one member
    for a deterministic model), on the device and `block_steps` lead times
    at a time: iterating it yields a
    `StartsBlock` per block of the starts being forecast, a start's members
    advancing together as in `MemberBlocks`.

    A trajectory (a start and a seed) is forecast as `MemberBlocks`
    forecasts a member, a start's seeds `batch_size` per call where the
    model can, unless the model has `predict_device_batch`: then
    `batch_size` trajectories, of several starts, share a call, each
    start's context passed once with its seeds in the call, at most the
    model's `max_device_batch` where it has one (what a call of it can hold
    on the device). That is for
    throughput, the validations during training: batched, a trajectory can
    round differently from its own call, and a long forecast amplifies
    rounding. Starts that share no call are streamed one after another, so
    the device (and the host, for a model that forecasts there) holds only
    the calls of the starts being streamed."""

    def __init__(
        self,
        model: ForecastModel,
        series,
        starts: Iterable[int],
        n_steps: int,
        seeds: Iterable[Iterable[int]],
        block_steps: int = DEFAULT_CHUNK_T,
        batch_size: int = 1,
    ) -> None:
        _check_streaming(block_steps, batch_size)
        self.starts, seeds = list(starts), [list(start) for start in seeds]
        if not self.starts:
            raise ValueError("no starts to forecast")
        if len(seeds) != len(self.starts):
            raise ValueError(f"{len(seeds)} lists of seeds for {len(self.starts)} starts")
        if not all(seeds):
            raise ValueError("no seeds to forecast")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        window = model.window
        early = [s for s in self.starts if s < max(window, 1)]
        if early:
            raise ValueError(f"starts {early} leave no context for a window of {window} frames")
        self.model = model
        self.n_steps = n_steps
        self.frame_shape = tuple(series.shape[1:])
        self.block_steps = block_steps
        self.batch_size = batch_size
        self.contexts = [np.asarray(series[s - window : s]) for s in self.starts]
        # (trajectories as (start index, seed), several starts in the call) per call.
        self.across_starts = batch_size > 1 and hasattr(model, "predict_device_batch")
        if self.across_starts:
            trajectories = [(k, seed) for k, own in enumerate(seeds) for seed in own]
            size = min(batch_size, getattr(model, "max_device_batch", batch_size))
            calls = [trajectories[i : i + size] for i in range(0, len(trajectories), size)]
        else:
            calls = [
                [(k, seed) for seed in own[i : i + batch_size]]
                for k, own in enumerate(seeds)
                for i in range(0, len(own), batch_size)
            ]
        # The calls streamed together: consecutive ones that share a start.
        self.waves: list[list[list[tuple[int, int]]]] = []
        for call in calls:
            if self.waves and self.waves[-1][-1][-1][0] == call[0][0]:
                self.waves[-1].append(call)
            else:
                self.waves.append([call])

    def __len__(self) -> int:
        """How many blocks it yields: every wave's."""
        return len(self.waves) * -(-self.n_steps // self.block_steps)

    def __iter__(self) -> Iterator[StartsBlock]:
        for wave in self.waves:
            streams = [self._call_blocks(call) for call in wave]
            for first in range(0, self.n_steps, self.block_steps):
                steps = min(self.block_steps, self.n_steps - first)
                groups: dict[int, list[MemberGroup]] = {}
                for call, stream in zip(wave, streams):
                    block = _next_block(stream, len(call), first, steps)
                    for k in dict.fromkeys(k for k, _ in call):
                        indices = tuple(i for i, (start, _) in enumerate(call) if start == k)
                        groups.setdefault(k, []).append(MemberGroup(block, indices))
                yield StartsBlock(
                    first,
                    steps,
                    {
                        k: EnsembleBlock(first, steps, tuple(members), self.frame_shape)
                        for k, members in groups.items()
                    },
                )
            _check_exhausted(streams, -(-self.n_steps // self.block_steps))

    def _call_blocks(self, call: Sequence[tuple[int, int]]) -> Iterator[DeviceBlock]:
        model, seeds = self.model, [seed for _, seed in call]
        if not self.across_starts:
            context = self.contexts[call[0][0]]
            yield from _member_blocks(
                model,
                context,
                self.n_steps,
                self.block_steps,
                seeds,
                self.batch_size > 1,
                self.frame_shape,
            )
            return
        # Each start's context once, with its seeds in the call (in order:
        # a call's trajectories are its starts', start by start).
        starts = list(dict.fromkeys(k for k, _ in call))
        contexts = np.stack([self.contexts[k] for k in starts])
        seeded = (
            {"seeds": [[seed for start, seed in call if start == k] for k in starts]}
            if is_stochastic(model)
            else {}
        )
        yield from model.predict_device_batch(contexts, self.n_steps, self.block_steps, **seeded)


def _check_streaming(block_steps: int, batch_size: int) -> None:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if block_steps < 1:
        raise ValueError(f"block_steps must be >= 1, got {block_steps}")


def _member_blocks(
    model: ForecastModel,
    visible: np.ndarray,
    n_steps: int,
    block_steps: int,
    seeds: Sequence[int],
    batched: bool,
    frame_shape: tuple[int, ...],
) -> Iterator[DeviceBlock]:
    """The forecasts from `visible` for `seeds`, in one call where the model
    can (`batched`), as `DeviceBlock`s: a device model's own, any other's
    `predict` frames copied to the device a block at a time."""
    if hasattr(model, "predict_device_blocks"):
        seeded = {"seeds": list(seeds)} if is_stochastic(model) else {}
        yield from model.predict_device_blocks(visible, n_steps, block_steps, **seeded)
        return
    shape = (n_steps, *frame_shape)
    if batched and is_stochastic(model) and hasattr(model, "predict_members"):
        batch = model.predict_members(visible, n_steps, list(seeds))
        frames = _checked(batch, (len(seeds), *shape))
    else:
        frames = [_checked(predict_member(model, visible, n_steps, s), shape) for s in seeds]
    for start in range(0, n_steps, block_steps):
        stop = start + block_steps
        yield DeviceBlock(to_device(np.stack([f[start:stop] for f in frames])))


def _next_block(stream: Iterator[DeviceBlock], members: int, first: int, steps: int) -> DeviceBlock:
    """A call's next block, which must hold its `members` for `steps` lead
    times from `first`."""
    block = next(stream, None)
    if block is None or len(block) != members or block.steps != steps:
        raise ValueError(
            f"model gave the block from lead {first + 1} as "
            f"{None if block is None else (len(block), block.steps)}, "
            f"expected {(members, steps)} (members, steps)"
        )
    return block


def _check_exhausted(streams: Sequence[Iterator[DeviceBlock]], n_blocks: int) -> None:
    if any(next(stream, None) is not None for stream in streams):
        raise ValueError(f"model gave more blocks than the forecast's {n_blocks}")


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
