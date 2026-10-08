"""Per-step domain-mean quantities of a velocity series.

Shared by the long-horizon diagnostics (`diagnostics.py`, scored once after
training) and the stability check (`stability.py`, part of the selection
score that early stopping uses every epoch). Kept in a module of their own
so that training -- and the DVC `train` stage's dependencies -- only reach
what early stopping needs, not the diagnostics.
"""

from __future__ import annotations

import numpy as np

from mhd_surrogate.analysis.fields import vorticity

# Steps processed at a time, so a long series is never in float64 at once.
DEFAULT_CHUNK_T = 32


def kinetic_energy(block: np.ndarray) -> np.ndarray:
    """Spatial mean of 0.5 |u|^2 per step, shape (t,)."""
    return 0.5 * (block**2).sum(axis=1).mean(axis=(1, 2))


def enstrophy(block: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Spatial mean of 0.5 w^2 per step, shape (t,)."""
    return 0.5 * (vorticity(block, dx, dy) ** 2).mean(axis=(1, 2))
