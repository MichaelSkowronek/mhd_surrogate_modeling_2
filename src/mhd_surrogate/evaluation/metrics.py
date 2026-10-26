"""Pointwise forecast-error metrics.

Arrays are (T, C, Nx, Ny): T forecast steps, C velocity channels. The flow is
chaotic, so pointwise error saturates after some lead time for any model;
these metrics are the short-horizon half of the evaluation, and
`diagnostics` is the long-horizon half (does the forecast stay on the
attractor).
"""

from __future__ import annotations

import math

import numpy as np


def rmse_per_step(
    prediction: np.ndarray, target: np.ndarray, scale: np.ndarray | None = None
) -> np.ndarray:
    """RMSE over channels and space at every lead time, shape (T,).

    `scale` is an optional per-channel (C,) divisor (e.g. the training std),
    so the error is relative to the flow's natural amplitude and channels of
    different magnitude count equally.
    """
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if scale is not None:
        error = error / np.asarray(scale, dtype=np.float64).reshape(1, -1, 1, 1)
    return np.sqrt((error**2).mean(axis=(1, 2, 3)))


def skill_horizon(curve: np.ndarray, threshold: float) -> int:
    """Number of leading steps whose error stays at or below `threshold`
    (the whole length if it never exceeds it). A NaN error counts as above
    it: a forecast that blew up has no skill from there on."""
    above = np.flatnonzero(~(np.asarray(curve) <= threshold))
    return int(above[0]) if above.size else len(curve)


def selection_score(
    skill: float,
    rmse_at_tie_break_lead: float,
    stable_steps: float,
    n_steps: int,
    peak_ratio: float,
    stable_fraction: float = 0.0,
    n_forecasts: int = 1,
) -> float:
    """One number that ranks candidates like the selection rule (CLAUDE.md,
    "Data analysis"): by stable steps (how long the forecast stays bounded,
    `evaluation.stability`); then a stable forecast by stable fraction, by
    skill horizon and, as a tie-break, by the RMSE at the tie-break lead,
    and an unstable one by its peak ratio, lower first. A higher score is a
    better candidate under the rule; early stopping and the hyperparameter
    search maximize it.

    The RMSE term r / (1 + r) lies in [0, 1) and grows with r, so it orders
    equal skill horizons without ever outweighing one step of skill (an
    infinite RMSE scores 1, at worst tying the next skill down). Skill minus
    that term lies in (-1, n_steps] for a forecast of `n_steps` steps, so
    weighting the stable steps by n_steps + 1 makes one more stable step
    outweigh any skill and RMSE (again tying at worst for an infinite RMSE).
    Scores are therefore comparable only between forecasts of the same
    length, i.e. on the same dataset.

    `peak_ratio` is how far the forecast strayed: its largest block mean of
    energy or enstrophy over the truth's (`stability.stability_scores`'
    peak ratios, the larger of the two; NaN for a forecast that produced
    NaN). An unstable forecast is ranked by it instead of by skill: the
    stable steps of most unstable forecasts are 0 or 100 while their peaks
    range over decades (3x to 60,000x in the U-Net runs), and a forecast
    that strays 3x is far closer to a usable one than one that blows up
    1000x, whatever its short-range skill. Its term (`peak_ratio_term`)
    lies in [0, n_steps], so one more stable step still outweighs it, and
    an unstable forecast never outscores a stable one (tying at worst). A
    stable forecast's peak ratio isn't used: below the truth's range it
    means damping, which the stability check deliberately doesn't reward.

    `stable_fraction` is the share of stable forecasts among the at most
    `n_forecasts` the candidate is judged with (`selection.RunSelection`:
    the validations nearest it in its training run), and counts only for a
    forecast stable over all `n_steps`: it orders candidates that are,
    which the stable steps can't. The default 0 scores a forecast alone.
    Distinct fractions of at most `n_forecasts` forecasts differ by at least
    1 / n_forecasts^2, and a stable forecast's skill and RMSE terms span
    n_steps + 1, so weighting the fraction by (n_steps + 1) n_forecasts^2
    makes a larger one outweigh them (tying at worst); a stable forecast
    with no bonus still scores at least any unstable one. Scores with a
    fraction are comparable only for the same `n_forecasts` too."""
    bonus = stable_fraction_bonus(stable_fraction, stable_steps, n_steps, n_forecasts)
    r = rmse_at_tie_break_lead
    if not r >= 0:  # also catches NaN
        return float("-inf")
    if stable_steps < n_steps:
        return float(stable_steps) * (n_steps + 1) + peak_ratio_term(peak_ratio, n_steps)
    return (
        float(stable_steps) * (n_steps + 1)
        + bonus
        + float(skill)
        - (1.0 if math.isinf(r) else r / (1.0 + r))
    )


def peak_ratio_term(peak_ratio: float, n_steps: int) -> float:
    """What `selection_score` adds for an unstable forecast's peak ratio:
    n_steps / (1 + log10 peak_ratio), in [0, n_steps] and decreasing as the
    peak grows (a peak ratio at most 1 counts as 1); 0 for a NaN or
    infinite one, the worst. Only its order matters, not its shape."""
    if not math.isfinite(peak_ratio):
        return 0.0
    return n_steps / (1.0 + math.log10(max(peak_ratio, 1.0)))


def stable_fraction_bonus(
    stable_fraction: float, stable_steps: float, n_steps: int, n_forecasts: int
) -> float:
    """What `selection_score` adds for a stable fraction: the fraction times
    (n_steps + 1) n_forecasts^2 for a forecast stable over all `n_steps`,
    0 for one that isn't."""
    if not 0.0 <= stable_fraction <= 1.0:
        raise ValueError(f"stable_fraction must be in [0, 1], got {stable_fraction}")
    if n_forecasts < 1:
        raise ValueError(f"n_forecasts must be >= 1, got {n_forecasts}")
    if stable_steps < n_steps:
        return 0.0
    return stable_fraction * (n_steps + 1) * n_forecasts**2
