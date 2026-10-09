"""Long rollouts: does a model stay bounded far past a dataset's length?

The selection rule's stability check (`stability.py`) only sees the scored
forecast, 837 steps of the validation dataset. A surrogate is meant to be
rolled out for as long as one likes, so these roll a model out for any
number of steps from a dataset's context, one block at a time, and rate each
block against the truth with the same blocks and threshold. Used by the
canonical model's gate (`gate.py`) and
`scripts/evaluation/check_rollout_stability.py`; kept out of `stability.py`,
which early stopping imports, so the DVC `train` stage doesn't depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mhd_surrogate.evaluation.protocol import ForecastModel, predict_member
from mhd_surrogate.evaluation.quantities import DEFAULT_CHUNK_T
from mhd_surrogate.evaluation.stability import (
    QUANTITIES,
    block_means,
    energy_and_enstrophy,
    first_failing_step,
)


def rollout_block_means(
    model: ForecastModel,
    context: np.ndarray,
    n_steps: int,
    block_steps: int,
    dx: float,
    dy: float,
    chunk_t: int = DEFAULT_CHUNK_T,
) -> dict[str, np.ndarray]:
    """Block means of the energy and enstrophy of an `n_steps` rollout from
    `context`, one value per `block_steps`-step block (the last may be
    shorter), for rollouts longer than a dataset.

    The model predicts one block at a time from the last `window` frames so
    far (its context, then its own predictions), so only one block is ever
    held in memory. For a model whose state is its last `window` frames (the
    autoregressive networks) that's the same rollout as one long `predict`
    up to float rounding at the block boundaries (amplified by a rollout
    that is blowing up: ~1% by step 200 for the first U-Nets); a model that
    projects its input (DMD) restarts from the projection of its own last
    frames at every block. A stochastic model samples block i with seed i.
    """
    window = np.asarray(context[len(context) - model.window :])
    means = {q: [] for q in QUANTITIES}
    for start in range(0, n_steps, block_steps):
        n = min(block_steps, n_steps - start)
        block = np.asarray(predict_member(model, window, n, seed=start // block_steps))
        for quantity, values in energy_and_enstrophy(block, dx, dy, chunk_t).items():
            means[quantity].append(values.mean())
        if model.window:
            window = np.concatenate([window, block])[-model.window :]
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
) -> RolloutStability:
    """Roll `model` out for `n_steps` from `context` (`rollout_block_means`)
    and compare its block means with those of `targets`, the scored steps
    that follow the context."""
    truth = energy_and_enstrophy(targets, dx, dy, chunk_t)
    return RolloutStability(
        blocks=rollout_block_means(model, context, n_steps, block_steps, dx, dy, chunk_t),
        reference={q: block_means(truth[q], block_steps) for q in QUANTITIES},
        n_steps=n_steps,
        block_steps=block_steps,
        max_ratio=max_ratio,
    )
