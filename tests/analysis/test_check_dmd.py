import numpy as np
import pytest
from check_dmd import dominant_modes, exact_dmd, mode_amplitudes


def _traveling_wave_1d(
    n_space: int, n_snapshots: int, dt: float, k0: float, omega0: float, growth_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """exp(growth_rate*t) * cos(k0*x - omega0*t): a 1D traveling, growing or
    decaying wave. Unlike a single static spatial pattern with an
    oscillating scalar coefficient (which is exactly rank-1 and therefore
    cannot support a complex DMD eigenvalue pair -- a harmonic oscillator
    needs a 2D state), this spans cos(k0*x) and sin(k0*x) as two genuinely
    independent spatial directions, giving DMD the 2D subspace a real
    oscillation requires. No requirement to land on an FFT bin (unlike the
    FFT tests elsewhere in this project) -- DMD isn't grid-locked.

    Returns (snapshots, mode_shape): mode_shape = exp(i*k0*x) is the
    complex spatial pattern the recovered DMD mode should be proportional
    to (up to arbitrary complex scale, and up to a sign of k0 -- the
    conjugate exp(-i*k0*x) is an equally valid representative, matching
    which of the two conjugate eigenvalues dominant_modes happens to keep).
    """
    x = np.arange(n_space)
    t = np.arange(n_snapshots) * dt
    snapshots = np.exp(growth_rate * t)[None, :] * np.cos(k0 * x[:, None] - omega0 * t[None, :])
    mode_shape = np.exp(1j * k0 * x)
    return snapshots, mode_shape


def _mode_shape_match(mode: np.ndarray, shape: np.ndarray) -> float:
    """1.0 iff `mode` (complex) is exactly proportional to `shape`, up to an
    arbitrary complex scale -- DMD modes are only defined that way.
    """
    return np.abs(np.vdot(mode, shape)) / (np.linalg.norm(mode) * np.linalg.norm(shape))


def _best_shape_match(mode: np.ndarray, shape: np.ndarray) -> float:
    """Like _mode_shape_match, but also accepts the complex-conjugate shape
    (a real field's decomposition is sign-of-frequency-ambiguous; whichever
    conjugate eigenvalue dominant_modes kept, its mode should match one or
    the other).
    """
    return max(_mode_shape_match(mode, shape), _mode_shape_match(mode, shape.conj()))


def test_exact_dmd_recovers_a_traveling_waves_frequency_growth_and_shape():
    n_space, n_snapshots, dt = 40, 60, 1.0
    k0, omega0, growth_rate = 2 * np.pi * 3 / n_space, 0.3, -0.02
    snapshots, shape = _traveling_wave_1d(n_space, n_snapshots, dt, k0, omega0, growth_rate)
    x, xprime = snapshots[:, :-1], snapshots[:, 1:]

    eigenvalues, modes, energy_fraction = exact_dmd(x, xprime)
    assert energy_fraction == pytest.approx(1.0, rel=1e-6)

    amplitudes = mode_amplitudes(modes, x[:, 0])
    top = dominant_modes(eigenvalues, amplitudes, modes, dt, n_modes=1)

    assert abs(top[0]["frequency"]) == pytest.approx(omega0, rel=0.02)
    assert top[0]["growth_rate"] == pytest.approx(growth_rate, abs=0.005)
    assert top[0]["period"] == pytest.approx(2 * np.pi / omega0, rel=0.02)

    mode = modes[:, top[0]["mode_index"]]
    assert _best_shape_match(mode, shape) == pytest.approx(1.0, rel=1e-4)


def test_exact_dmd_separates_two_superposed_traveling_waves():
    n_space, n_snapshots, dt = 60, 80, 1.0
    k_slow, k_fast = 2 * np.pi * 2 / n_space, 2 * np.pi * 5 / n_space
    slow, shape_slow = _traveling_wave_1d(n_space, n_snapshots, dt, k_slow, 0.15, -0.01)
    fast, shape_fast = _traveling_wave_1d(n_space, n_snapshots, dt, k_fast, 0.6, 0.01)
    snapshots = slow + fast
    x, xprime = snapshots[:, :-1], snapshots[:, 1:]

    eigenvalues, modes, _ = exact_dmd(x, xprime)
    amplitudes = mode_amplitudes(modes, x[:, 0])
    top = dominant_modes(eigenvalues, amplitudes, modes, dt, n_modes=2)

    by_freq = sorted(top, key=lambda m: abs(m["frequency"]))
    assert abs(by_freq[0]["frequency"]) == pytest.approx(0.15, rel=0.02)
    assert by_freq[0]["growth_rate"] == pytest.approx(-0.01, abs=0.005)
    assert abs(by_freq[1]["frequency"]) == pytest.approx(0.6, rel=0.02)
    assert by_freq[1]["growth_rate"] == pytest.approx(0.01, abs=0.005)

    mode_slow = modes[:, by_freq[0]["mode_index"]]
    mode_fast = modes[:, by_freq[1]["mode_index"]]
    assert _best_shape_match(mode_slow, shape_slow) == pytest.approx(1.0, rel=1e-3)
    assert _best_shape_match(mode_fast, shape_fast) == pytest.approx(1.0, rel=1e-3)


def test_exact_dmd_default_rank_drops_negligible_singular_values():
    """A single traveling wave is exactly rank-2 (it spans cos(k0*x) and
    sin(k0*x), nothing else); the other ~58 singular values are pure
    floating-point noise. Dividing by them (the naive "use the full
    economy-SVD rank" default) would blow up into numerical garbage --
    the automatic rank must drop them instead.
    """
    n_space, n_snapshots, dt = 40, 60, 1.0
    snapshots, _ = _traveling_wave_1d(n_space, n_snapshots, dt, 2 * np.pi * 3 / n_space, 0.3, 0.0)
    x, xprime = snapshots[:, :-1], snapshots[:, 1:]

    eigenvalues, _, energy_fraction = exact_dmd(x, xprime)

    assert energy_fraction == pytest.approx(1.0, rel=1e-6)
    assert len(eigenvalues) == 2
    assert np.all(np.isfinite(eigenvalues))


def test_exact_dmd_explicit_rank_truncates_and_reduces_energy_fraction():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((30, 20))
    xprime = rng.standard_normal((30, 20))

    _, _, full_energy = exact_dmd(x, xprime, rank=20)
    _, _, truncated_energy = exact_dmd(x, xprime, rank=5)

    assert full_energy == pytest.approx(1.0, rel=1e-6)
    assert truncated_energy < full_energy


def test_dominant_modes_deduplicates_conjugate_pairs():
    # Two eigenvalues that are exact conjugates must collapse to one entry.
    eigenvalues = np.array([np.exp(1j * 0.3), np.exp(-1j * 0.3), 0.9 + 0j])
    modes = np.eye(3, dtype=complex)
    amplitudes = np.array([1.0, 1.0, 0.5])

    top = dominant_modes(eigenvalues, amplitudes, modes, dt=1.0, n_modes=3)

    assert len(top) == 2
    frequencies = sorted(abs(m["frequency"]) for m in top)
    assert frequencies[0] == pytest.approx(0.0, abs=1e-9)
    assert frequencies[1] == pytest.approx(0.3, rel=1e-6)
