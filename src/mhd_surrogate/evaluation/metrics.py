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


def selection_score(skill: float, rmse_at_tie_break_lead: float) -> float:
    """One number that ranks candidates like the selection rule: by skill
    horizon, ties broken by the RMSE at the tie-break lead (CLAUDE.md, "Data
    analysis"). The RMSE term r / (1 + r) lies in [0, 1) and grows with r, so
    it orders equal skill horizons without ever outweighing one step of
    skill (an infinite RMSE scores 1, at worst tying the next skill down): a
    higher score is a better candidate under the rule. Early stopping and the
    hyperparameter search maximize it."""
    r = rmse_at_tie_break_lead
    if not r >= 0:  # also catches NaN
        return float("-inf")
    return float(skill) - (1.0 if math.isinf(r) else r / (1.0 + r))
