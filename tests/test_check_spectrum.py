import numpy as np
import pytest
from check_spectrum import spectrum_sum, wavenumbers


def test_spectrum_sum_recovers_amplitude_and_wavenumber_of_a_sine():
    """A unit-amplitude sine has variance 0.5 and all its power at k0."""
    nx = 300
    lx = 30.0
    dx = lx / (nx - 1)
    n_periods = 10
    k0 = 2 * np.pi * n_periods / lx

    x = np.arange(nx) * dx
    ux = np.sin(k0 * x)[:, None] * np.ones((1, 4))  # constant along the other axis
    uy = np.zeros_like(ux)
    block = np.stack([ux, uy])[None]  # (t=1, C=2, nx, ny=4)

    power = spectrum_sum(block, axis=2, spacing=dx)  # (C, K)
    k = wavenumbers(nx, dx)
    dk = k[1] - k[0]

    integral = power[0][1:].sum() * dk
    assert integral == pytest.approx(0.5, rel=0.02)

    peak_k = k[np.argmax(power[0][1:]) + 1]
    assert peak_k == pytest.approx(k0, rel=0.02)

    # u_y is exactly zero, so it should carry ~no power.
    assert power[1][1:].sum() * dk < 1e-8


def test_wavenumbers_start_at_zero_and_step_matches_domain_length():
    n, spacing = 100, 0.1
    k = wavenumbers(n, spacing)
    assert k[0] == 0.0
    length = n * spacing
    assert k[1] == pytest.approx(2 * np.pi / length)
