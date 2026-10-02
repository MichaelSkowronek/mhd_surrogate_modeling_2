"""Derived fields from the (T, 2, Nx, Ny) velocity array."""

from __future__ import annotations

import numpy as np


def vorticity(block: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Out-of-plane vorticity w = du_y/dx - du_x/dy for a (t, 2, Nx, Ny) block.

    Second-order central differences (one-sided at the boundaries).
    """
    return np.gradient(block[:, 1], dx, axis=1) - np.gradient(block[:, 0], dy, axis=2)


def divergence(block: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Divergence du_x/dx + du_y/dy for a (t, 2, Nx, Ny) block (same scheme as
    `vorticity`)."""
    return np.gradient(block[:, 0], dx, axis=1) + np.gradient(block[:, 1], dy, axis=2)


def speed(block: np.ndarray) -> np.ndarray:
    """sqrt(u_x^2 + u_y^2) for a (t, 2, Nx, Ny) block."""
    return np.sqrt(block[:, 0] ** 2 + block[:, 1] ** 2)


# Shared between scripts/viz/make_video.py and scripts/viz/make_dmd_video.py:
# which derived field to show and how to color it.
FIELDS = {
    "vorticity": {"cmap": "RdBu_r", "symmetric": True},
    "u_x": {"cmap": "viridis", "symmetric": False},
    "u_y": {"cmap": "viridis", "symmetric": False},
    "speed": {"cmap": "viridis", "symmetric": False},
}


def compute_field(block: np.ndarray, field: str, dx: float, dy: float) -> np.ndarray:
    """`block` is (t, 2, Nx, Ny); returns (t, Nx, Ny)."""
    if field == "vorticity":
        return vorticity(block, dx, dy)
    if field == "u_x":
        return block[:, 0]
    if field == "u_y":
        return block[:, 1]
    if field == "speed":
        return speed(block)
    raise ValueError(f"unknown field: {field}")
