"""Dynamic Mode Decomposition, shared by scripts/analysis/check_dmd.py and
scripts/viz/make_dmd_video.py.

DMD fits the best linear dynamical system A (in a least-squares sense,
x_{t+1} ~= A x_t) directly to a snapshot sequence and eigendecomposes it:
each eigenvalue gives a mode's frequency *and* growth/decay rate, at
whatever frequency the data actually supports, not a grid-locked FFT bin.
See check_dmd.py's docstring for the full rationale.
"""

from __future__ import annotations

import numpy as np


def build_dmd_state(data: np.ndarray) -> np.ndarray:
    """(T, C, Nx, Ny) array -> (state_dim, T) fluctuation-about-time-mean
    matrix, ready for `exact_dmd` -- channels and space flattened into one
    state per snapshot (DMD models the joint dynamics of the whole vector
    field, not each channel independently), with the time-mean subtracted
    (matching check_wavenumber_spectrum.py's `u' = u - <u>_t` convention),
    so the spectrum isn't dominated by a large near-unit eigenvalue for the
    persistent mean flow.
    """
    t = data.shape[0]
    state = data.reshape(t, -1)
    return (state - state.mean(axis=0)).T


def exact_dmd(
    x: np.ndarray, xprime: np.ndarray, rank: int | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact DMD (Tu et al., 2014): eigenvalues and spatial modes of the
    best-fit linear operator A satisfying `xprime ~= A @ x`, without forming
    A explicitly (the state dimension here is far larger than the number of
    snapshots, so A itself would be too large to form or even fit in
    memory -- only its action, via a rank-r SVD-based reduction, is
    computed).

    `x`, `xprime` have shape (state_dim, n_snapshots): consecutive-snapshot
    pairs, `x[:, i] -> xprime[:, i]` one step later. `rank` truncates the
    SVD of `x` (default: an automatically-determined numerical rank --
    singular values below `s[0] * max(x.shape) * eps` are dropped, matching
    `numpy.linalg.matrix_rank`'s own convention. This only avoids dividing
    by numerically-negligible singular values, which is unconditionally
    unsafe -- e.g. exactly-low-rank data, such as a single spatial pattern
    or few superposed ones, otherwise leaves near-zero values in `s_r` that
    blow up `1/s_r` into numerical garbage. It is not, by itself, denoising:
    pass an explicit smaller `rank` to actually filter out noise-dominated
    small-but-not-negligible singular values, standard DMD practice for
    real, noisy data).

    Returns (eigenvalues, modes, energy_fraction): eigenvalues (complex,
    shape (r,)) are the discrete-time DMD eigenvalues (`lambda =
    exp(mu*dt)` for the continuous-time rate `mu`); modes (complex, shape
    (state_dim, r)) are the corresponding spatial mode shapes, each defined
    only up to an arbitrary complex scale/phase; energy_fraction is the
    share of `x`'s variance the rank-r truncation retains.
    """
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    if rank is None:
        tol = max(x.shape) * np.finfo(s.dtype).eps
        r = int(np.count_nonzero(s > s[0] * tol))
    else:
        r = min(rank, s.shape[0])
    energy_fraction = float((s[:r] ** 2).sum() / (s**2).sum())
    u_r, s_r, v_r = u[:, :r], s[:r], vh[:r].conj().T

    a_tilde = u_r.conj().T @ xprime @ v_r @ np.diag(1.0 / s_r)
    eigenvalues, w = np.linalg.eig(a_tilde)
    modes = xprime @ v_r @ np.diag(1.0 / s_r) @ w
    return eigenvalues, modes, energy_fraction


def mode_amplitudes(modes: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Least-squares amplitude of each DMD mode fitting the first snapshot
    (`x0 ~= modes @ amplitudes`), the standard DMD mode-amplitude convention.
    """
    amplitudes, *_ = np.linalg.lstsq(modes, x0, rcond=None)
    return amplitudes


def dominant_modes(
    eigenvalues: np.ndarray,
    amplitudes: np.ndarray,
    modes: np.ndarray,
    dt: float,
    n_steps: int,
    n_modes: int,
) -> list[dict]:
    """Top `n_modes` DMD modes ranked by `|amplitude| * ||mode||` ("power"),
    deduplicating complex-conjugate eigenvalue pairs -- which both
    represent the same real oscillation -- down to one entry each.

    "power" is the mode's amplitude *at the first snapshot* (the standard
    DMD convention) -- not how much it actually contributes across the
    whole recorded window. A fast-decaying mode can have large power yet
    vanish within a handful of steps, contributing far less to the
    recorded series than a lower-power but near-neutral (barely decaying
    or growing) mode that persists throughout; ranking by power alone can
    be misleading about which mode actually matters. "rms_power" answers
    that instead: the root-mean-square of the mode's power over all
    `n_steps` of the actual window (`power * |eigenvalue|^t`, `t = 0 ..
    n_steps - 1`) -- equal to "power" exactly for a perfectly sustained
    mode (growth_rate 0), and smaller/larger than it for a decaying/growing
    one. Modes are still ranked (and `n_modes` selected) by "power", the
    standard convention; compare "rms_power" across the returned entries to
    judge which ones actually persist.

    For a real input, complex eigenvalues always occur in exact conjugate
    pairs (same growth rate, opposite-signed frequency): keeping only
    `frequency >= 0` keeps exactly one representative of each pair, plus
    every purely real eigenvalue (frequency exactly 0) once.

    Returns a list of {"mode_index", "eigenvalue", "frequency",
    "growth_rate", "period", "amplitude", "power", "rms_power"},
    continuous-time (`mu = log(eigenvalue) / dt`): frequency = Im(mu) in
    radians/step, growth_rate = Re(mu) per step (positive: growing,
    negative: decaying, ~0: a sustained oscillation/steady structure),
    period = 2*pi / frequency (inf if ~0). "eigenvalue" is the raw
    discrete-time value (`exp(mu*dt)`), for reconstructing the mode's time
    evolution directly (see `reconstruct_frames`) without re-deriving it
    from frequency/growth.
    """
    mu = np.log(eigenvalues) / dt
    keep = np.flatnonzero(mu.imag >= 0)
    power = np.abs(amplitudes[keep]) * np.linalg.norm(modes[:, keep], axis=0)
    order = keep[np.argsort(power)[::-1]][:n_modes]

    t = np.arange(n_steps)
    result = []
    for i in order:
        frequency = float(mu[i].imag)
        growth_rate = float(mu[i].real)
        mode_power = float(np.abs(amplitudes[i]) * np.linalg.norm(modes[:, i]))
        result.append(
            {
                "mode_index": int(i),
                "eigenvalue": complex(eigenvalues[i]),
                "frequency": frequency,
                "growth_rate": growth_rate,
                "period": 2 * np.pi / frequency if frequency != 0 else float("inf"),
                "amplitude": complex(amplitudes[i]),
                "power": mode_power,
                "rms_power": float(np.sqrt(np.mean((mode_power * np.exp(growth_rate * t)) ** 2))),
            }
        )
    return result


def reconstruct_frames(
    mode: np.ndarray,
    amplitude: complex,
    eigenvalue: complex,
    n_frames: int,
    normalize_growth: bool = True,
) -> np.ndarray:
    """Re(amplitude * mode * eigenvalue**t) for t = 0, ..., n_frames - 1:
    the mode's own time evolution under the fitted linear dynamics, as a
    real (n_frames, state_dim) array of frames.

    If `normalize_growth`, `eigenvalue`'s magnitude is divided out first (so
    only its phase rotation drives the frames) whenever it has one (i.e. the
    mode oscillates, `eigenvalue.imag != 0`) -- growth/decay is already
    reported separately (see `dominant_modes`), and would otherwise make a
    decaying oscillation fade to invisible, or a growing one blow up, within
    a few periods. A non-oscillating mode (real `eigenvalue`, e.g. the mean
    flow or a pure transient) has nothing left to animate if its only
    dynamics -- growth/decay -- were also normalized away, so this leaves it
    untouched regardless of `normalize_growth`.
    """
    if normalize_growth and eigenvalue.imag != 0:
        eigenvalue = eigenvalue / abs(eigenvalue)
    t = np.arange(n_frames)
    coefficients = amplitude * eigenvalue**t
    return (coefficients[:, None] * mode[None, :]).real
