"""Rollout stability: how long does a forecast stay bounded?

An autoregressive model can track the flow closely for tens of steps and
then blow up, its energy growing without bound, while the short-horizon
scores (skill horizon, RMSE at lead 10) never notice. This check catches it
from the scored forecast itself: the forecast is cut into consecutive blocks
of `block_steps` steps, and a block fails when its mean kinetic energy or
enstrophy exceeds `max_ratio` times the truth's largest block mean of the
same quantity. The forecast is stable up to the first failing block.

Block means rather than single steps, so a short excursion doesn't fail a
block; the default threshold (2x) sits well above the truth's own
variability (its 100-step block means vary by ~5% in energy and ~18% in
enstrophy on the validation dataset). One-sided on purpose: a forecast that
damps toward a smooth state (Hankel DMD's enstrophy settles at ~0.2x the
truth's) is bounded, just wrong, and the physics guardrails judge that.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from mhd_surrogate.evaluation.diagnostics import DEFAULT_CHUNK_T, enstrophy, kinetic_energy
from mhd_surrogate.evaluation.protocol import ForecastModel

QUANTITIES = ("energy", "enstrophy")


def energy_and_enstrophy(
    series, dx: float, dy: float, chunk_t: int = DEFAULT_CHUNK_T
) -> dict[str, np.ndarray]:
    """Per-step domain-mean kinetic energy and enstrophy of a (T, 2, Nx, Ny)
    series, shape (T,) each, computed `chunk_t` steps at a time in float64."""
    energy, ens = [], []
    for start in range(0, series.shape[0], chunk_t):
        block = np.asarray(series[start : start + chunk_t], dtype=np.float64)
        energy.append(kinetic_energy(block))
        ens.append(enstrophy(block, dx, dy))
    return {"energy": np.concatenate(energy), "enstrophy": np.concatenate(ens)}


def block_means(values: np.ndarray, block_steps: int) -> np.ndarray:
    """Means of consecutive `block_steps`-step blocks of a (T,) series; the
    last block holds the remainder, so it may be shorter."""
    if block_steps < 1:
        raise ValueError(f"block_steps must be >= 1, got {block_steps}")
    values = np.asarray(values, dtype=np.float64)
    return np.array(
        [values[s : s + block_steps].mean() for s in range(0, len(values), block_steps)]
    )


def first_failing_step(
    ratios: Mapping[str, np.ndarray], block_steps: int, max_ratio: float, n_steps: int
) -> int:
    """The first step of the first block where any quantity's ratio (block
    mean over the truth's largest block mean, one value per block) exceeds
    `max_ratio` or is NaN; `n_steps` if none does."""
    failed = np.zeros(-(-n_steps // block_steps), dtype=bool)
    for ratio in ratios.values():
        failed |= ~(np.asarray(ratio) <= max_ratio)  # NaN fails too
    first = np.flatnonzero(failed)
    return int(first[0]) * block_steps if first.size else n_steps


def stability_scores(
    predicted: Mapping[str, np.ndarray],
    true: Mapping[str, np.ndarray],
    block_steps: int,
    max_ratio: float,
) -> dict[str, float]:
    """Stability of a forecast against its targets. `predicted` and `true`
    map each of `QUANTITIES` to its per-step series, shape (T,) (as returned
    by `energy_and_enstrophy`).

    - `stable_steps`: the first step of the first failing block, i.e. how many
      leading steps stay bounded; the whole length T if no block fails. A
      block fails when a quantity's block mean exceeds `max_ratio` times the
      truth's largest block mean, or is NaN (a forecast that blew up).
    - `energy_peak_ratio`, `enstrophy_peak_ratio`: the forecast's largest
      block mean over the truth's, i.e. how far it strayed (at most ~1 for a
      forecast within the truth's range; NaN if it produced NaN).
    """
    n_steps = len(true[QUANTITIES[0]])
    ratios = {}
    for quantity in QUANTITIES:
        if len(predicted[quantity]) != n_steps or len(true[quantity]) != n_steps:
            raise ValueError(f"{quantity}: predicted and true series must have {n_steps} steps")
        reference = block_means(true[quantity], block_steps).max()
        ratios[quantity] = block_means(predicted[quantity], block_steps) / reference
    stable = first_failing_step(ratios, block_steps, max_ratio, n_steps)
    return {
        "stable_steps": float(stable),
        **{f"{q}_peak_ratio": float(ratio.max()) for q, ratio in ratios.items()},
    }


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
    frames at every block.
    """
    window = np.asarray(context[len(context) - model.window :])
    means = {q: [] for q in QUANTITIES}
    for start in range(0, n_steps, block_steps):
        block = np.asarray(model.predict(window, min(block_steps, n_steps - start)))
        for quantity, values in energy_and_enstrophy(block, dx, dy, chunk_t).items():
            means[quantity].append(values.mean())
        if model.window:
            window = np.concatenate([window, block])[-model.window :]
    return {q: np.array(values) for q, values in means.items()}
