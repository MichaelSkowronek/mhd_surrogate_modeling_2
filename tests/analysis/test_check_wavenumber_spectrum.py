import numpy as np
import pytest
from check_wavenumber_spectrum import peak_wavenumbers, wavenumber_spectrum_2d


def _plane_wave(nx: int, ny: int, dx: float, dy: float, n_periods_x: int, n_periods_y: int):
    """cos(kx0*x + ky0*y), with kx0/ky0 chosen to land exactly on an FFT bin."""
    lx, ly = nx * dx, ny * dy
    kx0 = 2 * np.pi * n_periods_x / lx
    ky0 = 2 * np.pi * n_periods_y / ly
    x = np.arange(nx)[:, None] * dx
    y = np.arange(ny)[None, :] * dy
    return np.cos(kx0 * x + ky0 * y), kx0, ky0


def _oscillating_series(pattern: np.ndarray, n_steps: int) -> np.ndarray:
    """`pattern` at alternating +/- sign, so its time mean is exactly zero and
    it survives time-mean-field detrending as a genuine fluctuation.
    """
    signs = np.where(np.arange(n_steps) % 2 == 0, 1.0, -1.0)
    return pattern[None] * signs[:, None, None]


def test_wavenumber_spectrum_2d_recovers_amplitude_and_wavenumber_of_an_oscillating_wave():
    """A unit-amplitude oscillating plane wave has variance 0.5 (same every
    frame, since the sign alternation squares away) and all its power at
    (kx0, ky0).
    """
    nx, ny, dx, dy, n_steps = 64, 48, 1.0, 1.0, 20
    wave, kx0, ky0 = _plane_wave(nx, ny, dx, dy, n_periods_x=3, n_periods_y=2)
    arr = _oscillating_series(wave, n_steps)[:, None]  # (T, C=1, Nx, Ny)

    kx, ky, power = wavenumber_spectrum_2d(arr, channel=0, dx=dx, dy=dy, chunk_t=7)

    dkx, dky = kx[1] - kx[0], ky[1] - ky[0]
    integral = power.sum() * dkx * dky
    assert integral == pytest.approx(0.5, rel=0.02)

    peak = peak_wavenumbers(kx, ky, power)
    assert peak["kx"] == pytest.approx(kx0, rel=0.02)
    assert peak["ky"] == pytest.approx(ky0, rel=0.02)


def test_wavenumber_spectrum_2d_removes_a_large_time_invariant_mean_profile():
    """A large-amplitude but time-invariant "mean profile" (same in every
    snapshot, at a wavenumber well separated from the oscillating wave) must
    contribute ~nothing to the spectrum: only the true time-mean field is
    subtracted, not each snapshot's own scalar mean, so a time-invariant
    spatial structure is fully removed regardless of its amplitude or shape.
    """
    nx, ny, dx, dy, n_steps = 64, 48, 1.0, 1.0, 20
    mean_profile, kx_mean, ky_mean = _plane_wave(nx, ny, dx, dy, n_periods_x=1, n_periods_y=1)
    wave, kx0, ky0 = _plane_wave(nx, ny, dx, dy, n_periods_x=5, n_periods_y=3)
    field = 10.0 * mean_profile[None] + _oscillating_series(wave, n_steps)
    arr = field[:, None]

    kx, ky, power = wavenumber_spectrum_2d(arr, channel=0, dx=dx, dy=dy, chunk_t=6)

    peak = peak_wavenumbers(kx, ky, power)
    assert peak["kx"] == pytest.approx(kx0, rel=0.02)
    assert peak["ky"] == pytest.approx(ky0, rel=0.02)

    # No leftover power at the mean profile's own (much larger-amplitude) wavenumber.
    i = np.argmin(np.abs(kx - kx_mean))
    j = np.argmin(np.abs(ky - ky_mean))
    assert power[i, j] < 1e-6 * peak["power"]


def test_wavenumber_spectrum_2d_chunking_does_not_change_result():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((10, 2, 16, 12))

    _, _, whole = wavenumber_spectrum_2d(arr, channel=1, dx=0.1, dy=0.2, chunk_t=100)
    _, _, chunked = wavenumber_spectrum_2d(arr, channel=1, dx=0.1, dy=0.2, chunk_t=3)

    np.testing.assert_allclose(whole, chunked)


def test_wavenumber_spectrum_2d_respects_n_steps_bound():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((20, 1, 16, 12))

    _, _, bounded = wavenumber_spectrum_2d(arr, channel=0, dx=1.0, dy=1.0, chunk_t=4, n_steps=8)
    _, _, whole = wavenumber_spectrum_2d(arr[:8], channel=0, dx=1.0, dy=1.0, chunk_t=4)

    np.testing.assert_allclose(bounded, whole)


def test_peak_wavenumbers_excludes_zero_wavenumber_bin():
    # A residual constant offset (leftover after the time-mean-field
    # subtraction rounds imperfectly, or just as a robustness check) would
    # dominate at (kx=0, ky=0); the peak found here must still be the wave.
    nx, ny = 64, 48
    power = np.full((nx, ny // 2 + 1), 1e-6)
    power[np.argmin(np.abs(np.arange(nx) - nx // 2)), 0] = 1.0  # huge DC-ish bin
    power[10, 5] = 0.5  # the "real" peak
    kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=1.0))
    ky = 2 * np.pi * np.fft.rfftfreq(ny, d=1.0)

    peak = peak_wavenumbers(kx, ky, power)

    assert peak["kx"] == pytest.approx(kx[10])
    assert peak["ky"] == pytest.approx(ky[5])
