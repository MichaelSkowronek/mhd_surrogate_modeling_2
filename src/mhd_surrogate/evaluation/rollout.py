"""Long rollouts: a forecast streamed in blocks, and whether it stays bounded.

A surrogate is meant to be rolled out for as long as one likes, longer than
a forecast that fits in memory (an 837-step one is ~1 GB), so a long
forecast is streamed a block at a time (`rollout_blocks`): for the rollout
checks here, and for a served forecast. The selection rule's stability check
(`stability.py`) only sees the scored forecast, 837 steps of the validation
dataset; `rollout_stability` rolls a model out for any number of steps from
a dataset's context and rates each block against the truth with the same
blocks and threshold (a stochastic model one member at a time: the rule
takes the median over them, like every per-member score). Used by the
canonical model's gate (`gate.py`) and
`scripts/evaluation/check_rollout_stability.py`; kept out of
`stability.py`, which early stopping imports, so the DVC `train` stage
doesn't depend on it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from mhd_surrogate.evaluation.protocol import ForecastModel, is_stochastic, predict_member
from mhd_surrogate.evaluation.quantities import DEFAULT_CHUNK_T
from mhd_surrogate.evaluation.stability import (
    QUANTITIES,
    block_means,
    energy_and_enstrophy,
    first_failing_step,
)


def rollout_blocks(
    model: ForecastModel, context: np.ndarray, n_steps: int, block_steps: int, seed: int = 0
) -> Iterator[np.ndarray]:
    """An `n_steps` forecast from the last `window` frames of `context`,
    yielded `block_steps` frames at a time (the last block may be shorter),
    so only one block is held at once; a stochastic model's is its sample
    for `seed`.

    A model with `predict_blocks` streams the forecast its one `predict`
    call would make (see `models/base.py`). Any other is restarted at every
    block from the last `window` frames of the forecast so far: for a model
    whose state is just those frames the same forecast, up to float
    rounding at the block boundaries, for any other not. Restarted, a
    stochastic model samples each block with its own seed (`block_seed`).
    """
    if block_steps < 1:
        raise ValueError(f"block_steps must be >= 1, got {block_steps}")
    window = np.asarray(context[len(context) - model.window :])
    sizes = [min(block_steps, n_steps - start) for start in range(0, n_steps, block_steps)]
    if hasattr(model, "predict_blocks"):
        seeded = {"seed": seed} if is_stochastic(model) else {}
        blocks = model.predict_blocks(window, n_steps, block_steps, **seeded)
    else:
        blocks = _restarted(model, window, sizes, seed)
    for n, block in zip(sizes, blocks, strict=True):
        block = np.asarray(block)
        expected = (n, *window.shape[1:])
        if block.shape != expected:
            raise ValueError(f"model yielded a block of shape {block.shape}, expected {expected}")
        yield block


def block_seed(seed: int, block: int) -> int:
    """The seed a restarted rollout samples block `block` of member `seed`
    with: `seed` itself for the first block, so it starts as the member's
    `predict` does, then one drawn from both (`SeedSequence`), so no two
    blocks of any members share a seed but by a ~2**-32 chance."""
    if block == 0:
        return seed
    return int(np.random.SeedSequence([seed, block]).generate_state(1)[0])


def _restarted(
    model: ForecastModel, window: np.ndarray, sizes: list[int], seed: int
) -> Iterator[np.ndarray]:
    for k, n in enumerate(sizes):
        block = np.asarray(predict_member(model, window, n, block_seed(seed, k)))
        if model.window:
            window = np.concatenate([window, block])[-model.window :]
        yield block


def rollout_block_means(
    model: ForecastModel,
    context: np.ndarray,
    n_steps: int,
    block_steps: int,
    dx: float,
    dy: float,
    chunk_t: int = DEFAULT_CHUNK_T,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Block means of the energy and enstrophy of an `n_steps` rollout from
    `context` (`rollout_blocks`; a stochastic model's sample for `seed`),
    one value per `block_steps`-step block (the last may be shorter), for
    rollouts longer than a dataset."""
    means = {q: [] for q in QUANTITIES}
    for block in rollout_blocks(model, context, n_steps, block_steps, seed=seed):
        for quantity, values in energy_and_enstrophy(block, dx, dy, chunk_t).items():
            means[quantity].append(values.mean())
    return {q: np.array(values) for q, values in means.items()}


@dataclass(frozen=True)
class RolloutStability:
    """A long rollout's block means against the truth's.

    `blocks` maps each of `QUANTITIES` to the rollout's block means (one per
    `block_steps`-step block of its `n_steps`), `reference` to the truth's
    block means over the scored steps; a block's ratio is its mean over the
    truth's largest, the same reference the selection rule's
    `stable_steps` uses, so the first blocks reproduce it.
    """

    blocks: dict[str, np.ndarray]
    reference: dict[str, np.ndarray]
    n_steps: int
    block_steps: int
    max_ratio: float

    @property
    def ratios(self) -> dict[str, np.ndarray]:
        return {q: self.blocks[q] / self.reference[q].max() for q in QUANTITIES}

    @property
    def stable_steps(self) -> int:
        return first_failing_step(self.ratios, self.block_steps, self.max_ratio, self.n_steps)

    def scores(self) -> dict[str, float]:
        """`stable_steps` and each quantity's peak ratio (NaN if the rollout
        produced NaN), as `stability_scores` names them."""
        return {
            "stable_steps": float(self.stable_steps),
            **{f"{q}_peak_ratio": float(np.max(r)) for q, r in self.ratios.items()},
        }


def rollout_stability(
    model: ForecastModel,
    context: np.ndarray,
    targets,
    n_steps: int,
    block_steps: int,
    max_ratio: float,
    dx: float,
    dy: float,
    chunk_t: int = DEFAULT_CHUNK_T,
    seed: int = 0,
) -> RolloutStability:
    """Roll `model` out for `n_steps` from `context` (`rollout_block_means`;
    a stochastic model's sample for `seed`) and compare its block means
    with those of `targets`, the scored steps that follow the context."""
    truth = energy_and_enstrophy(targets, dx, dy, chunk_t)
    return RolloutStability(
        blocks=rollout_block_means(model, context, n_steps, block_steps, dx, dy, chunk_t, seed),
        reference={q: block_means(truth[q], block_steps) for q in QUANTITIES},
        n_steps=n_steps,
        block_steps=block_steps,
        max_ratio=max_ratio,
    )
