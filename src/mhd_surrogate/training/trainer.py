"""Gradient-descent training loop for neural surrogates.

Generic over the network: a model hands `train` an Equinox module, a loss
`loss_fn(network, batch, key)` and a `WindowSampler` over its (normalized)
training frames, and gets back the trained network. What a batch means --
`window` input frames followed by `rollout_steps` targets, unrolled -- is the
loss function's business. `key` is a JAX PRNG key for a loss that's random
(e.g. noise on its inputs): a different one every optimizer step, derived
from the seed and the step count, so a resumed run draws the same ones.

- Optimizer: AdamW, gradients clipped by global norm, learning rate with a
  linear warmup and cosine decay over `max_epochs`.
- An epoch is `samples_per_epoch` windows drawn without replacement (all of
  them by default); it's also the unit of validation and checkpointing, so
  it's set short enough for early stopping and the hyperparameter search's
  pruning to see the curve.
- Early stopping: `validate(network)` scores the network every `eval_every`
  epochs; training stops after `patience` evaluations without a strictly
  better `selection_score`, and the best network is returned, not the last.
- Divergence: a non-finite or blown-up loss raises `DivergenceError`
  (`training/tracking.py` tags the run).
- Resumable: with a `state_dir`, the network, optimizer state, sampler RNG
  and early-stopping bookkeeping are saved after every epoch, and a later
  `train` call with the same configuration continues from the last one, with
  the same batches it would have drawn. Weights go to per-epoch files and
  `state.json`, written last and atomically, names the current ones, so a
  crash mid-save leaves the previous epoch's state intact.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from mhd_surrogate.models.base import FitHooks
from mhd_surrogate.training.tracking import DivergenceError

log = logging.getLogger(__name__)

STATE_FILE = "state.json"


@dataclass(frozen=True)
class TrainerConfig:
    max_epochs: int = 50
    samples_per_epoch: int | None = None  # None: every window once per epoch
    batch_size: int = 8
    learning_rate: float = 1e-3
    warmup_steps: int = 200
    final_lr_fraction: float = 0.01  # the cosine decays to this fraction of learning_rate
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 5  # evaluations without improvement before stopping
    eval_every: int = 1  # epochs
    log_every: int = 20  # optimizer steps
    max_loss: float = 100.0  # a loss above this (or non-finite) is divergence
    seed: int = 0
    prefetch: int = 2  # batches prepared ahead on a background thread; 0: none

    def __post_init__(self) -> None:
        for name in ("max_epochs", "batch_size", "patience", "eval_every", "log_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")


class WindowSampler:
    """Batches of `span` consecutive frames from a list of (T, ...) arrays.

    A window never crosses from one array (dataset) into the next. Each
    epoch draws `samples` windows without replacement (cycling through
    fresh permutations if more are asked for than exist); a final partial
    batch is dropped, so every batch has one shape and compiles once.
    """

    def __init__(self, arrays: Sequence[np.ndarray], span: int) -> None:
        self.arrays = list(arrays)
        self.span = span
        self.index = np.array(
            [(d, s) for d, a in enumerate(self.arrays) for s in range(len(a) - span + 1)],
            dtype=np.int64,
        ).reshape(-1, 2)
        if len(self.index) == 0:
            raise ValueError(f"no array has the {span} frames a window needs")

    def __len__(self) -> int:
        return len(self.index)

    def samples(self, samples_per_epoch: int | None) -> int:
        return len(self) if samples_per_epoch is None else samples_per_epoch

    def steps_per_epoch(self, batch_size: int, samples_per_epoch: int | None) -> int:
        return self.samples(samples_per_epoch) // batch_size

    def epoch(
        self, rng: np.random.Generator, batch_size: int, samples_per_epoch: int | None = None
    ) -> Iterator[np.ndarray]:
        n = self.samples(samples_per_epoch)
        if n // batch_size == 0:
            raise ValueError(f"an epoch of {n} windows has no full batch of {batch_size}")
        reps = -(-n // len(self))
        order = np.concatenate([rng.permutation(len(self)) for _ in range(reps)])[:n]
        for start in range(0, n - batch_size + 1, batch_size):
            picks = self.index[order[start : start + batch_size]]
            yield np.stack([self.arrays[d][s : s + self.span] for d, s in picks])


def prefetch(iterator: Iterator[np.ndarray], size: int) -> Iterator[jax.Array]:
    """Move batches to the device on a background thread, `size` ahead, so
    gathering the next batch overlaps the current step."""
    if size < 1:
        yield from (jax.device_put(x) for x in iterator)
        return
    buffer: queue.Queue = queue.Queue(maxsize=size)
    done = object()

    def produce() -> None:
        try:
            for x in iterator:
                buffer.put(jax.device_put(x))
        except BaseException as error:  # re-raised in the consumer
            buffer.put(error)
        buffer.put(done)

    threading.Thread(target=produce, daemon=True).start()
    while (item := buffer.get()) is not done:
        if isinstance(item, BaseException):
            raise item
        yield item


@dataclass
class TrainingState:
    """Bookkeeping that is saved with the weights (JSON)."""

    epoch: int = 0  # completed epochs
    step: int = 0  # completed optimizer steps
    best_score: float = -math.inf
    best_epoch: int = 0  # 0: no evaluation yet
    evals_since_best: int = 0
    stopped_early: bool = False
    rng_state: dict[str, Any] = field(default_factory=dict)
    latest_file: str = ""
    best_file: str = ""
    fingerprint: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainResult:
    network: eqx.Module
    info: dict[str, float]


def _inexact(tree):
    return eqx.filter(tree, eqx.is_inexact_array)


def make_optimizer(config: TrainerConfig, total_steps: int) -> tuple[Any, Any]:
    """(optimizer, learning-rate schedule)."""
    warmup = min(config.warmup_steps, max(total_steps - 1, 0))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=warmup,
        decay_steps=max(total_steps, warmup + 1),
        end_value=config.learning_rate * config.final_lr_fraction,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(schedule, weight_decay=config.weight_decay),
    )
    return optimizer, schedule


def train(
    network: eqx.Module,
    loss_fn: Callable[[eqx.Module, jax.Array, jax.Array], jax.Array],
    sampler: WindowSampler,
    config: TrainerConfig,
    hooks: FitHooks,
    validate: Callable[[eqx.Module], dict[str, float]] | None = None,
    fingerprint: dict[str, Any] | None = None,
) -> TrainResult:
    """Train `network` to minimize `loss_fn` over the sampler's batches.

    `validate` (None: no early stopping) scores a network; `hooks` gives the
    metrics sink and the state directory. `fingerprint` (e.g. the model's
    hyperparameters) is saved with the state, and a resume is refused if it
    or `config` differ: the saved weights would belong to another run.
    """
    fingerprint = {"trainer": asdict(config), "model": fingerprint or {}}
    steps_per_epoch = sampler.steps_per_epoch(config.batch_size, config.samples_per_epoch)
    if steps_per_epoch == 0:
        raise ValueError(
            f"samples_per_epoch {sampler.samples(config.samples_per_epoch)} is less than one "
            f"batch of {config.batch_size}"
        )
    optimizer, schedule = make_optimizer(config, steps_per_epoch * config.max_epochs)
    opt_state = optimizer.init(_inexact(network))
    rng = np.random.default_rng(config.seed)
    state = TrainingState(fingerprint=fingerprint)
    best = network

    state_dir = hooks.state_dir
    if state_dir is not None and (state_dir / STATE_FILE).exists():
        state, network, opt_state, best = _restore(state_dir, fingerprint, network, opt_state)
        rng.bit_generator.state = state.rng_state
        log.info(
            "resumed from %s at epoch %d (step %d, best score %.4g at epoch %d)",
            state_dir,
            state.epoch,
            state.step,
            state.best_score,
            state.best_epoch,
        )
    resumed_at = state.epoch

    base_key = jax.random.key(config.seed)

    @eqx.filter_jit
    def step(network, opt_state, batch, key):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(network, batch, key)
        updates, opt_state = optimizer.update(grads, opt_state, _inexact(network))
        return eqx.apply_updates(network, updates), opt_state, loss, optax.tree.norm(grads)

    for epoch in range(state.epoch + 1, config.max_epochs + 1):
        if state.stopped_early:
            break
        start = time.perf_counter()
        window_start = start
        losses, epoch_losses, norms = [], [], []
        batches = sampler.epoch(rng, config.batch_size, config.samples_per_epoch)
        for batch in prefetch(batches, config.prefetch):
            key = jax.random.fold_in(base_key, state.step)
            network, opt_state, loss, grad_norm = step(network, opt_state, batch, key)
            losses.append(loss)
            norms.append(grad_norm)
            state.step += 1
            if state.step % config.log_every == 0:
                now = time.perf_counter()
                throughput = len(losses) * config.batch_size / (now - window_start)
                mean_loss = _check_loss(losses, config.max_loss, state.step)
                hooks.log_metrics(
                    {
                        "loss": mean_loss,
                        "grad_norm": float(jnp.mean(jnp.stack(norms))),
                        "lr": float(schedule(state.step)),
                        "samples_per_second": throughput,
                    },
                    state.step,
                )
                epoch_losses.append(mean_loss * len(losses))
                losses, norms, window_start = [], [], now
        if losses:
            epoch_losses.append(_check_loss(losses, config.max_loss, state.step) * len(losses))
        state.epoch = epoch
        epoch_metrics = {
            "epoch": float(epoch),
            "epoch_loss": sum(epoch_losses) / steps_per_epoch,
            "epoch_seconds": time.perf_counter() - start,
        }

        if validate is not None and (epoch % config.eval_every == 0 or epoch == config.max_epochs):
            scores = validate(network)
            epoch_metrics.update({f"val_monitor.{k}": v for k, v in scores.items()})
            score = scores["selection_score"]
            if score > state.best_score:
                state.best_score, state.best_epoch, state.evals_since_best = score, epoch, 0
                best = network
            else:
                state.evals_since_best += 1
                state.stopped_early = state.evals_since_best >= config.patience
            log.info(
                "epoch %d: loss %.4g, val %s (best score %.10g at epoch %d)",
                epoch,
                epoch_metrics["epoch_loss"],
                ", ".join(f"{k} {v:.4g}" for k, v in scores.items()),
                state.best_score,
                state.best_epoch,
            )
        else:
            log.info("epoch %d: loss %.4g", epoch, epoch_metrics["epoch_loss"])
        hooks.log_metrics(epoch_metrics, state.step)

        if state_dir is not None:
            state.rng_state = rng.bit_generator.state
            _save(state_dir, state, network, opt_state, best)
        if state.stopped_early:
            log.info("early stop: no improvement in %d evaluations", config.patience)

    result = best if validate is not None and state.best_epoch > 0 else network
    info = {
        "epochs": float(state.epoch),
        "steps": float(state.step),
        "best_epoch": float(state.best_epoch),
        "stopped_early": float(state.stopped_early),
        "resumed_at_epoch": float(resumed_at),
    }
    if state.best_epoch > 0:
        info["best_selection_score"] = state.best_score
    return TrainResult(network=result, info=info)


def _check_loss(losses: list[jax.Array], max_loss: float, step: int) -> float:
    """The mean of `losses`; raises DivergenceError if any is non-finite or
    above `max_loss`."""
    values = np.asarray(jnp.stack(losses))
    if not np.all(np.isfinite(values)) or values.max() > max_loss:
        raise DivergenceError(f"loss diverged by step {step}: {values.max()!r} (max {max_loss})")
    return float(values.mean())


def _save(state_dir: Path, state: TrainingState, network, opt_state, best) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    old = _load_state(state_dir)
    state.latest_file = f"latest-epoch-{state.epoch:04d}.eqx"
    eqx.tree_serialise_leaves(state_dir / state.latest_file, (network, opt_state))
    if state.best_epoch > 0:
        state.best_file = f"best-epoch-{state.best_epoch:04d}.eqx"
        if not (state_dir / state.best_file).exists():
            eqx.tree_serialise_leaves(state_dir / state.best_file, best)
    tmp = state_dir / (STATE_FILE + ".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2))
    os.replace(tmp, state_dir / STATE_FILE)
    # Only now are the previous epoch's files no longer referenced.
    if old is not None:
        for name in {old.latest_file, old.best_file} - {state.latest_file, state.best_file, ""}:
            (state_dir / name).unlink(missing_ok=True)


def _load_state(state_dir: Path) -> TrainingState | None:
    path = state_dir / STATE_FILE
    if not path.exists():
        return None
    return TrainingState(**json.loads(path.read_text()))


def _restore(state_dir: Path, fingerprint: dict, network, opt_state):
    state = _load_state(state_dir)
    if state.fingerprint != json.loads(json.dumps(fingerprint)):
        raise ValueError(
            f"training state in {state_dir} was saved with a different configuration:\n"
            f"saved {state.fingerprint}\nnow   {fingerprint}"
        )
    network, opt_state = eqx.tree_deserialise_leaves(
        state_dir / state.latest_file, (network, opt_state)
    )
    best = network
    if state.best_file:
        best = eqx.tree_deserialise_leaves(state_dir / state.best_file, network)
    return state, network, opt_state, best
