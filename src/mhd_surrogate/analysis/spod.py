"""Spectral Proper Orthogonal Decomposition (SPOD), shared by
scripts/analysis/check_spod.py.

Standard POD (mhd_surrogate.analysis.pod) ranks spatial patterns by their
share of *total* variance over the whole recorded window, but each mode's
time coefficient can (and typically does) mix many frequencies -- the
mode-0/mode-1 quadrature pair check_pod.py finds is the direct consequence:
POD, given only spatial orthogonality to work with, has to split one
traveling structure into two co-equal modes rather than describing it as a
single oscillation. Standard DMD (mhd_surrogate.analysis.dmd) goes the other
way: modes are single-frequency by construction, but are not generally
spatially orthogonal, and a mode's whole-window importance has to be
estimated separately from its frequency (see dominant_modes's rms_power).

SPOD (Towne, Schmidt & Colonius, 2018, "Spectral proper orthogonal
decomposition and its relationship to dynamic mode decomposition and
resolvent analysis") gets both at once: split the snapshot sequence into
overlapping, Hann-windowed segments (same convention as
mhd_surrogate.analysis.spectral.welch_spectrum's Welch estimator), Fourier
transform each segment in time, then -- independently at each frequency --
treat the segments' Fourier coefficients as a "snapshot matrix" and take its
SVD, exactly as mhd_surrogate.analysis.pod.pod does in the time domain. The
result is a set of spatial modes *at each frequency*, mutually orthogonal
within that frequency and ranked by the share of that frequency's energy
they capture; each mode oscillates at exactly that one frequency, like a
DMD mode, but assumes the underlying process is statistically stationary --
there is no growth/decay to estimate, and no power-vs-persistence ambiguity
to worry about, since a mode's importance and its frequency are the same
question here.

The per-frequency eigenvalues (`eigenvalues[f]`, summed over modes) equal
the total of every channel's mhd_surrogate.analysis.spectral.welch_spectrum
power spectral density at that frequency -- SPOD doesn't add or remove any
energy, it just reorganizes each frequency's cross-channel covariance into
orthogonal modes (checked directly in tests/analysis/test_spod.py).
"""

from __future__ import annotations

import numpy as np


def spod(
    state: np.ndarray, nperseg: int, noverlap: int, spacing: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SPOD via Welch-blocked cross-spectral density eigendecomposition.

    `state` (state_dim, n_snapshots) is a real-valued, mean-subtracted
    snapshot matrix (see mhd_surrogate.analysis.dmd.build_dmd_state), split
    into overlapping segments of length `nperseg` (stepping by `nperseg -
    noverlap`, same convention as
    mhd_surrogate.analysis.spectral.welch_spectrum). Each segment is
    independently detrended (per spatial/channel component) and
    Hann-windowed, then Fourier-transformed in time; at each frequency bin,
    the segments' Fourier coefficients form a (state_dim, n_segments)
    complex matrix whose SVD gives that frequency's SPOD modes and
    eigenvalues, normalized (like `mhd_surrogate.analysis.spectral.power_spectrum`)
    so that summing `eigenvalues[f]` over modes equals the total of every
    channel's Welch power spectral density at that frequency.

    Returns (omega, modes, eigenvalues): omega (n_freq,) angular frequency
    bins, same convention as `power_spectrum`; modes (n_freq, state_dim,
    n_modes) complex spatial SPOD modes, most energetic first at each
    frequency and mutually orthonormal within it (`modes[f].conj().T @
    modes[f]` is the identity), `n_modes = min(state_dim, n_segments)`;
    eigenvalues (n_freq, n_modes) each mode's share of that frequency's
    cross-channel energy (non-negative, descending within each frequency;
    not normalized across frequencies, since -- unlike POD's
    energy_fraction over the whole window -- energy here is naturally split
    across both frequency and mode).
    """
    if noverlap >= nperseg:
        raise ValueError(f"noverlap ({noverlap}) must be < nperseg ({nperseg})")
    state_dim, n_snapshots = state.shape
    if n_snapshots < nperseg:
        raise ValueError(f"n_snapshots ({n_snapshots}) must be >= nperseg ({nperseg})")

    step = nperseg - noverlap
    starts = list(range(0, n_snapshots - nperseg + 1, step))
    n_segments = len(starts)
    window = np.hanning(nperseg)
    scale = np.sqrt(spacing / (window**2).sum() / (2 * np.pi))

    n_freq = nperseg // 2 + 1
    transforms = np.empty((n_segments, state_dim, n_freq), dtype=complex)
    for k, start in enumerate(starts):
        segment = state[:, start : start + nperseg]
        detrended = segment - segment.mean(axis=1, keepdims=True)
        transforms[k] = np.fft.rfft(detrended * window, axis=1)

    last = -1 if nperseg % 2 == 0 else None
    transforms[:, :, 1:last] *= np.sqrt(2)  # one-sided spectrum, as in power_spectrum
    transforms *= scale

    omega = 2 * np.pi * np.fft.rfftfreq(nperseg, d=spacing)
    n_modes = min(state_dim, n_segments)
    modes = np.empty((n_freq, state_dim, n_modes), dtype=complex)
    eigenvalues = np.empty((n_freq, n_modes))
    for f in range(n_freq):
        block_matrix = transforms[:, :, f].T  # (state_dim, n_segments)
        u, s, _ = np.linalg.svd(block_matrix, full_matrices=False)
        modes[f] = u
        eigenvalues[f] = s**2 / n_segments

    return omega, modes, eigenvalues


def leading_frequencies(omega: np.ndarray, eigenvalues: np.ndarray, n_peaks: int) -> list[dict]:
    """Top `n_peaks` frequency bins by leading-mode (mode 0) eigenvalue,
    excluding the zero-frequency bin -- unlike
    mhd_surrogate.analysis.spectral.dominant_periods, this ranks by raw
    magnitude rather than requiring a local maximum: SPOD's eigenvalue
    spectrum concentrates around a dominant frequency far more cleanly than
    a raw periodogram does (it's already a Welch-averaged, cross-channel
    quantity), so the local-maximum edge-bin caveat documented on
    `dominant_periods` is not needed here.

    Returns a list of {"period", "omega", "eigenvalue", "eigenvalue_fraction"}
    (fraction relative to the total leading-mode eigenvalue, excluding the
    zero-frequency bin), most dominant first.
    """
    leading = eigenvalues[:, 0]
    total = leading[1:].sum()
    order = np.argsort(leading[1:])[::-1][:n_peaks] + 1
    return [
        {
            "period": 2 * np.pi / omega[i],
            "omega": omega[i],
            "eigenvalue": leading[i],
            "eigenvalue_fraction": leading[i] / total,
        }
        for i in order
    ]
