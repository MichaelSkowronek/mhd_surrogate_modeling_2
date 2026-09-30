import numpy as np
import pytest

from mhd_surrogate.analysis.pod import pod, pod_reconstruction


def test_pod_recovers_two_orthogonal_patterns_ranked_by_energy():
    # Two orthonormal spatial patterns, two orthonormal time coefficients:
    # this is already an exact SVD by construction (matched orthonormal
    # bases scaled by amp_a > amp_b), so pod() must recover it exactly.
    pattern_a = np.array([1.0, 1.0, -1.0, -1.0]) / 2
    pattern_b = np.array([1.0, -1.0, 1.0, -1.0]) / 2
    ca = np.array([1.0, 1.0, -1.0, -1.0]) / 2
    cb = np.array([1.0, -1.0, 1.0, -1.0]) / 2
    amp_a, amp_b = 5.0, 2.0
    state = amp_a * np.outer(pattern_a, ca) + amp_b * np.outer(pattern_b, cb)

    modes, energy_fraction, coefficients = pod(state)

    total = amp_a**2 + amp_b**2
    assert energy_fraction[0] == pytest.approx(amp_a**2 / total, rel=1e-9)
    assert energy_fraction[1] == pytest.approx(amp_b**2 / total, rel=1e-9)

    # Up to an arbitrary sign, mode 0 must match pattern_a and mode 1 pattern_b.
    assert abs(np.dot(modes[:, 0], pattern_a)) == pytest.approx(1.0, rel=1e-9)
    assert abs(np.dot(modes[:, 1], pattern_b)) == pytest.approx(1.0, rel=1e-9)

    # Coefficients must match amp * c, up to the same sign as their mode.
    sign_a = np.sign(np.dot(modes[:, 0], pattern_a))
    sign_b = np.sign(np.dot(modes[:, 1], pattern_b))
    np.testing.assert_allclose(coefficients[0], sign_a * amp_a * ca, atol=1e-9)
    np.testing.assert_allclose(coefficients[1], sign_b * amp_b * cb, atol=1e-9)


def test_pod_modes_are_orthonormal():
    # The defining property distinguishing POD modes from DMD modes, which
    # are not generally orthogonal.
    rng = np.random.default_rng(0)
    state = rng.standard_normal((10, 6))

    modes, _, _ = pod(state)

    np.testing.assert_allclose(modes.T @ modes, np.eye(modes.shape[1]), atol=1e-9)


def test_pod_reconstructs_the_state_exactly():
    rng = np.random.default_rng(0)
    state = rng.standard_normal((10, 6))

    modes, _, coefficients = pod(state)

    np.testing.assert_allclose(modes @ coefficients, state, atol=1e-9)


def test_pod_energy_fractions_sum_to_one():
    rng = np.random.default_rng(0)
    state = rng.standard_normal((8, 5))

    _, energy_fraction, _ = pod(state)

    assert energy_fraction.sum() == pytest.approx(1.0, rel=1e-9)


def test_pod_reconstruction_of_all_modes_matches_the_full_state():
    rng = np.random.default_rng(0)
    state = rng.standard_normal((10, 6))
    modes, _, coefficients = pod(state)

    full = pod_reconstruction(modes, coefficients, list(range(modes.shape[1])))

    np.testing.assert_allclose(full, state, atol=1e-9)


def test_pod_reconstruction_of_a_subset_matches_a_manual_sum():
    rng = np.random.default_rng(0)
    state = rng.standard_normal((10, 6))
    modes, _, coefficients = pod(state)

    pair = pod_reconstruction(modes, coefficients, [0, 2])

    expected = np.outer(modes[:, 0], coefficients[0]) + np.outer(modes[:, 2], coefficients[2])
    np.testing.assert_allclose(pair, expected, atol=1e-9)


def test_pod_reconstruction_of_a_single_mode_is_its_own_outer_product():
    rng = np.random.default_rng(0)
    state = rng.standard_normal((10, 6))
    modes, _, coefficients = pod(state)

    single = pod_reconstruction(modes, coefficients, [1])

    np.testing.assert_allclose(single, np.outer(modes[:, 1], coefficients[1]), atol=1e-9)
