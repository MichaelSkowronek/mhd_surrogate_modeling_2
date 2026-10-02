"""Pointwise forecast-error metrics.

Arrays are (T, C, Nx, Ny): T forecast steps, C velocity channels. The flow is
chaotic, so pointwise error saturates after some lead time for any model;
these metrics are the short-horizon half of the evaluation, and
`diagnostics` is the long-horizon half (does the forecast stay on the
attractor).
"""

from __future__ import annotations

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
    (the whole length if it never exceeds it)."""
    above = np.flatnonzero(np.asarray(curve) > threshold)
    return int(above[0]) if above.size else len(curve)
