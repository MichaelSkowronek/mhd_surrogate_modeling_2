import numpy as np
import pytest

from mhd_surrogate.analysis.spectral import welch_spectrum
from mhd_surrogate.analysis.spod import leading_frequencies, spod


def test_spod_eigenvalue_sum_matches_welch_power_summed_over_channels():
    # SPOD reorganizes each frequency's cross-channel energy into orthogonal
    # modes; it must not add or remove any of it. Summing the eigenvalues at
    # a frequency must equal summing that frequency's Welch power spectral
    # density over every channel.
    rng = np.random.default_rng(0)
    state_dim, n = 3, 800
    nperseg, noverlap = 100, 50
    state = rng.standard_normal((state_dim, n))

    omega, _, eigenvalues = spod(state, nperseg, noverlap)

    channel_power = np.zeros_like(omega)
    for c in range(state_dim):
        channel_omega, power = welch_spectrum(state[c], nperseg, noverlap)
        np.testing.assert_allclose(channel_omega, omega)
        channel_power += power

    np.testing.assert_allclose(eigenvalues.sum(axis=1), channel_power, rtol=1e-9)


def test_spod_modes_are_unitary_at_each_frequency():
    rng = np.random.default_rng(0)
    state = rng.standard_normal((5, 600))

    _, modes, _ = spod(state, nperseg=100, noverlap=50)

    for f in range(modes.shape[0]):
        np.testing.assert_allclose(modes[f].conj().T @ modes[f], np.eye(modes.shape[2]), atol=1e-9)


def test_spod_recovers_a_single_frequency_patterns_shape_and_frequency():
    # A spatial pattern modulated by one pure tone is exactly rank-1 at that
    # tone's frequency bin (every channel shares the same time series, just
    # scaled by the pattern), so SPOD must recover the pattern as the sole
    # leading mode there, with essentially all of that frequency's energy
    # concentrated in it and negligible leading-mode energy elsewhere.
    nperseg, t0 = 100, 20  # t0 divides nperseg exactly: lands on a bin
    n = 500
    t = np.arange(n)
    pattern = np.array([1.0, 2.0, -1.0, 0.5, -2.0])
    state = pattern[:, None] * np.cos(2 * np.pi * t / t0)[None, :]

    omega, modes, eigenvalues = spod(state, nperseg, noverlap=50)

    bin_idx = round(nperseg / t0)
    leading = eigenvalues[:, 0]
    # Exactly rank-1 at the tone's own bin: all of that frequency's energy
    # is in the leading mode.
    assert leading[bin_idx] == pytest.approx(eigenvalues[bin_idx].sum(), rel=1e-9)
    # The Hann window's mainlobe spreads some energy into neighboring bins
    # (expected, not leakage from anything else here), so this only checks
    # bin_idx is the dominant one, not that it holds ~all the energy.
    assert bin_idx == np.argmax(leading[1:]) + 1

    cosine_similarity = abs(np.vdot(modes[bin_idx, :, 0], pattern)) / (
        np.linalg.norm(modes[bin_idx, :, 0]) * np.linalg.norm(pattern)
    )
    assert cosine_similarity == pytest.approx(1.0, rel=1e-6)


def test_leading_frequencies_recovers_a_known_tone():
    nperseg, t0 = 100, 20
    n = 500
    t = np.arange(n)
    pattern = np.array([1.0, -1.0])
    state = pattern[:, None] * np.cos(2 * np.pi * t / t0)[None, :]

    omega, _, eigenvalues = spod(state, nperseg, noverlap=50)
    peaks = leading_frequencies(omega, eigenvalues, n_peaks=1)

    assert peaks[0]["period"] == pytest.approx(t0, rel=1e-6)


def test_spod_rejects_noverlap_not_less_than_nperseg():
    with pytest.raises(ValueError):
        spod(np.zeros((2, 100)), nperseg=50, noverlap=50)


def test_spod_rejects_fewer_snapshots_than_nperseg():
    with pytest.raises(ValueError):
        spod(np.zeros((2, 30)), nperseg=50, noverlap=0)
