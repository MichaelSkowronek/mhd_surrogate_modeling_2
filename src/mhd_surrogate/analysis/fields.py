"""Derived fields from the (T, 2, Nx, Ny) velocity array."""

from __future__ import annotations

import numpy as np


def vorticity(block: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Out-of-plane vorticity w = du_y/dx - du_x/dy for a (t, 2, Nx, Ny) block.

    Second-order central differences (one-sided at the boundaries).
    """
    return np.gradient(block[:, 1], dx, axis=1) - np.gradient(block[:, 0], dy, axis=2)
