import numpy as np
import pytest
from check_spacetime_spectrum import peak_wavenumber_frequency, spacetime_spectrum_3d


def _traveling_wave(
    nt: int, nx: int, ny: int, dt: float, dx: float, dy: float, n_t: int, n_x: int, n_y: int
):
    """cos(kx0*x + ky0*y - omega0*t): a wave moving through space over time,
    with omega0/kx0/ky0 chosen to land exactly on FFT bins.
    """
    omega0 = 2 * np.pi * n_t / (nt * dt)
    kx0 = 2 * np.pi * n_x / (nx * dx)
    ky0 = 2 * np.pi * n_y / (ny * dy)
    t = np.arange(nt)[:, None, None] * dt
    x = np.arange(nx)[None, :, None] * dx
    y = np.arange(ny)[None, None, :] * dy
    field = np.cos(kx0 * x + ky0 * y - omega0 * t)
    return field, omega0, kx0, ky0


def test_spacetime_spectrum_3d_recovers_a_traveling_wave():
    """A unit-amplitude traveling wave has variance 0.5 and all its power at
    the single (omega0, kx0, ky0) triple -- not just at kx0/ky0 pooled over
    all omega, or at omega0 pooled over all (kx, ky).
    """
    nt, nx, ny = 24, 32, 16
    dt, dx, dy = 1.0, 1.0, 1.0
    field, omega0, kx0, ky0 = _traveling_wave(nt, nx, ny, dt, dx, dy, n_t=3, n_x=4, n_y=2)
    arr = field[:, None]  # (T, C=1, Nx, Ny)

    omega, kx, ky, power = spacetime_spectrum_3d(arr, channel=0, dx=dx, dy=dy)

    domega = omega[1] - omega[0]
    dkx = kx[1] - kx[0]
    dky = ky[1] - ky[0]
    integral = power.sum() * domega * dkx * dky
    assert integral == pytest.approx(0.5, rel=0.05)

    peak = peak_wavenumber_frequency(omega, kx, ky, power)
    # cos(kx0*x + ky0*y - omega0*t) = 0.5*exp(i(kx0*x+ky0*y-omega0*t)) + its
    # conjugate; with ky0 > 0 chosen, the conjugate pair's ky < 0 half is
    # folded away (one-sided in ky), leaving this one deterministically.
    assert peak["omega"] == pytest.approx(-omega0, rel=0.05)
    assert peak["kx"] == pytest.approx(kx0, rel=0.05)
    assert peak["ky"] == pytest.approx(ky0, rel=0.05)


def test_spacetime_spectrum_3d_removes_a_large_time_invariant_pattern():
    """A large-amplitude but time-invariant spatial pattern (present
    unchanged in every snapshot, at a wavenumber well separated from the
    traveling wave) must contribute ~nothing: subtracting the time-mean
    field removes anything with omega=0 regardless of its spatial structure
    or amplitude.
    """
    nt, nx, ny = 24, 32, 16
    dt, dx, dy = 1.0, 1.0, 1.0
    wave, omega0, kx0, ky0 = _traveling_wave(nt, nx, ny, dt, dx, dy, n_t=3, n_x=4, n_y=2)
    x = np.arange(nx)[:, None] * dx
    y = np.arange(ny)[None, :] * dy
    static_kx0, static_ky0 = 2 * np.pi * 8 / (nx * dx), 2 * np.pi * 5 / (ny * dy)
    static_pattern = 10.0 * np.cos(static_kx0 * x + static_ky0 * y)
    arr = (wave + static_pattern[None])[:, None]

    omega, kx, ky, power = spacetime_spectrum_3d(arr, channel=0, dx=dx, dy=dy)
    peak = peak_wavenumber_frequency(omega, kx, ky, power)

    assert abs(peak["omega"]) == pytest.approx(omega0, rel=0.05)

    i = np.argmin(np.abs(omega))
    j = np.argmin(np.abs(kx - static_kx0))
    k = np.argmin(np.abs(ky - static_ky0))
    assert power[i, j, k] < 1e-6 * peak["power"]


def test_spacetime_spectrum_3d_respects_n_steps_bound():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((20, 1, 12, 10))

    omega_b, kx_b, ky_b, bounded = spacetime_spectrum_3d(arr, channel=0, dx=1.0, dy=1.0, n_steps=8)
    omega_w, kx_w, ky_w, whole = spacetime_spectrum_3d(arr[:8], channel=0, dx=1.0, dy=1.0)

    np.testing.assert_allclose(bounded, whole)
    np.testing.assert_allclose(omega_b, omega_w)
    np.testing.assert_allclose(kx_b, kx_w)
    np.testing.assert_allclose(ky_b, ky_w)


def test_peak_wavenumber_frequency_excludes_dc_point():
    nt, nx, ny = 10, 8, 6
    power = np.full((nt, nx, ny // 2 + 1), 1e-6)
    power[nt // 2, nx // 2, 0] = 1.0  # huge DC point
    power[7, 5, 2] = 0.5  # the "real" peak
    omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nt, d=1.0))
    kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=1.0))
    ky = 2 * np.pi * np.fft.rfftfreq(ny, d=1.0)

    peak = peak_wavenumber_frequency(omega, kx, ky, power)

    assert peak["omega"] == pytest.approx(omega[7])
    assert peak["kx"] == pytest.approx(kx[5])
    assert peak["ky"] == pytest.approx(ky[2])
