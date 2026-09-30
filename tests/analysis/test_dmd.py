import numpy as np
import pytest

from mhd_surrogate.analysis.dmd import (
    build_dmd_state,
    dominant_modes,
    exact_dmd,
    mode_amplitudes,
    reconstruct_frames,
)


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


def test_build_dmd_state_shape_and_mean_subtraction():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((5, 2, 3, 4))  # (T, C, Nx, Ny)

    state = build_dmd_state(data)

    assert state.shape == (2 * 3 * 4, 5)
    time_mean = data.mean(axis=0)
    for t in range(5):
        np.testing.assert_allclose(state[:, t].reshape(2, 3, 4), data[t] - time_mean)


def test_exact_dmd_recovers_a_traveling_waves_frequency_growth_and_shape():
    n_space, n_snapshots, dt = 40, 60, 1.0
    k0, omega0, growth_rate = 2 * np.pi * 3 / n_space, 0.3, -0.02
    snapshots, shape = _traveling_wave_1d(n_space, n_snapshots, dt, k0, omega0, growth_rate)
    x, xprime = snapshots[:, :-1], snapshots[:, 1:]

    eigenvalues, modes, energy_fraction = exact_dmd(x, xprime)
    assert energy_fraction == pytest.approx(1.0, rel=1e-6)

    amplitudes = mode_amplitudes(modes, x[:, 0])
    top = dominant_modes(eigenvalues, amplitudes, modes, dt, n_steps=n_snapshots, n_modes=1)

    assert abs(top[0]["frequency"]) == pytest.approx(omega0, rel=0.02)
    assert top[0]["growth_rate"] == pytest.approx(growth_rate, abs=0.005)
    assert top[0]["period"] == pytest.approx(2 * np.pi / omega0, rel=0.02)
    assert top[0]["eigenvalue"] == pytest.approx(eigenvalues[top[0]["mode_index"]])
    # growth_rate < 0 here, so the mode's RMS power over the window must be
    # strictly below its (first-snapshot) power.
    assert top[0]["rms_power"] < top[0]["power"]

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
    top = dominant_modes(eigenvalues, amplitudes, modes, dt, n_steps=n_snapshots, n_modes=2)

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

    top = dominant_modes(eigenvalues, amplitudes, modes, dt=1.0, n_steps=50, n_modes=3)

    assert len(top) == 2
    frequencies = sorted(abs(m["frequency"]) for m in top)
    assert frequencies[0] == pytest.approx(0.0, abs=1e-9)
    assert frequencies[1] == pytest.approx(0.3, rel=1e-6)


def test_dominant_modes_rms_power_reflects_persistence_over_the_window():
    """A mode's amplitude at the first snapshot ("power") can rank it above
    a mode that actually contributes far more across the whole recorded
    window: this is exactly the discrepancy a fast-decaying-but-initially-
    strong mode creates, and "rms_power" must reflect the window, not the
    first snapshot, to catch it.
    """
    modes = np.eye(2, dtype=complex)
    # A fast-decaying mode with higher power at t=0 than a near-neutral one.
    eigenvalues = np.array([0.85 + 0j, 0.999 + 0j])
    amplitudes = np.array([10.0, 8.0])

    top = dominant_modes(eigenvalues, amplitudes, modes, dt=1.0, n_steps=500, n_modes=2)
    by_power = sorted(top, key=lambda m: m["power"], reverse=True)
    by_rms = sorted(top, key=lambda m: m["rms_power"], reverse=True)

    assert by_power[0]["power"] > by_power[1]["power"]  # the fast-decaying mode ranks first...
    assert by_rms[0]["mode_index"] != by_power[0]["mode_index"]  # ...but not by rms_power


def test_dominant_modes_rms_power_equals_power_for_a_sustained_mode():
    # growth_rate = 0 (|eigenvalue| = 1): power is identical at every step,
    # so its RMS over the window must equal the first-snapshot power exactly.
    eigenvalues = np.array([1.0 + 0j])
    modes = np.array([[1.0 + 0j]])
    amplitudes = np.array([3.0 + 0j])

    top = dominant_modes(eigenvalues, amplitudes, modes, dt=1.0, n_steps=200, n_modes=1)

    assert top[0]["rms_power"] == pytest.approx(top[0]["power"], rel=1e-9)


def test_reconstruct_frames_normalizes_out_growth_for_an_oscillating_mode():
    # Re(e^{i*theta}*(1,i)) = (cos theta, -sin theta), a unit vector for any
    # theta: with growth normalized away, only a pure rotation remains, so
    # every frame's norm should be exactly equal.
    mode = np.array([1.0, 1j]) / np.sqrt(2)
    eigenvalue = np.exp(-0.05 + 0.4j)
    amplitude = 2.0 + 0j

    frames = reconstruct_frames(mode, amplitude, eigenvalue, n_frames=30, normalize_growth=True)

    norms = np.linalg.norm(frames, axis=1)
    np.testing.assert_allclose(norms, abs(amplitude) / np.sqrt(2), rtol=1e-10)


def test_reconstruct_frames_includes_growth_when_requested():
    mode = np.array([1.0, 1j]) / np.sqrt(2)
    eigenvalue = np.exp(-0.1 + 0.3j)
    amplitude = 1.0 + 0j
    n_frames = 20

    frames = reconstruct_frames(mode, amplitude, eigenvalue, n_frames, normalize_growth=False)

    t = np.arange(n_frames)
    expected = np.abs(amplitude) * np.abs(eigenvalue) ** t / np.sqrt(2)
    np.testing.assert_allclose(np.linalg.norm(frames, axis=1), expected, rtol=1e-10)


def test_reconstruct_frames_does_not_normalize_a_non_oscillating_mode():
    # A real eigenvalue has no phase to normalize away; growth/decay is its
    # only dynamics, so normalize_growth must leave it untouched.
    mode = np.array([1.0, -1.0])
    eigenvalue = 0.9 + 0j
    amplitude = 1.0 + 0j
    n_frames = 10

    frames = reconstruct_frames(mode, amplitude, eigenvalue, n_frames, normalize_growth=True)

    t = np.arange(n_frames)
    expected = ((amplitude * eigenvalue**t)[:, None] * mode[None, :]).real
    np.testing.assert_allclose(frames, expected, rtol=1e-10)
