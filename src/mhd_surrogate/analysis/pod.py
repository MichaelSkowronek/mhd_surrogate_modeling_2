"""Proper Orthogonal Decomposition (POD, aka PCA), shared by
scripts/analysis/check_pod.py.

Unlike DMD (see mhd_surrogate.analysis.dmd), a POD mode carries no single
frequency or growth rate of its own -- its time coefficient can (and
typically does) contain a mix of frequencies -- but POD modes are mutually
orthogonal and exactly ranked by the variance ("energy") they capture, with
no eigenvalue problem or complex arithmetic needed: they are just the left
singular vectors of the (mean-subtracted) snapshot matrix. Energy is also
already a whole-window quantity by construction (it comes from the full
data matrix's variance, not a single snapshot), so there is no DMD-style
power-vs-persistence ambiguity to worry about here.
"""

from __future__ import annotations

import numpy as np


def pod(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """POD of a mean-subtracted snapshot matrix (see
    mhd_surrogate.analysis.dmd.build_dmd_state), via its SVD.

    `state` has shape (state_dim, n_snapshots).

    Returns (modes, energy_fraction, coefficients): modes (state_dim, r)
    are the (real) spatial POD modes, most energetic first and mutually
    orthonormal (`modes.T @ modes` is the identity), `r = min(state_dim,
    n_snapshots)`; energy_fraction (r,) is each mode's share of total
    variance (`s**2 / sum(s**2)`, `s` the singular values); coefficients
    (r, n_snapshots) are each mode's time coefficient (`s_i * vh[i]`), so
    that `state ~= modes @ coefficients` (exact, since no rank is dropped).
    """
    u, s, vh = np.linalg.svd(state, full_matrices=False)
    energy_fraction = s**2 / (s**2).sum()
    coefficients = s[:, None] * vh
    return u, energy_fraction, coefficients


def pod_reconstruction(
    modes: np.ndarray, coefficients: np.ndarray, mode_indices: list[int]
) -> np.ndarray:
    """Time series of the summed contribution of just the given POD modes:
    `sum_i outer(modes[:, i], coefficients[i, :])`, shape (state_dim,
    n_snapshots). A single mode's own time progression is the `len(
    mode_indices) == 1` case; passing every mode index reconstructs the
    original state exactly (matching `pod`'s own `state ~= modes @
    coefficients` property) -- this is a partial version of that same sum.
    Useful for e.g. watching a pair of near-equal-energy modes (the
    signature of a single traveling structure that POD, unlike DMD, can
    only represent as two modes in spatial quadrature -- see check_pod.py)
    travel together, isolated from the rest of the flow.
    """
    return modes[:, mode_indices] @ coefficients[mode_indices, :]
