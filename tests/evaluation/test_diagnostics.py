import numpy as np
import pytest

from mhd_surrogate.evaluation.diagnostics import (
    compare_diagnostics,
    enstrophy,
    kinetic_energy,
    log_spectral_distance,
    rms_divergence,
    series_summary,
)

DX, DY = 0.5, 0.25


def grid_field(n_steps, nx=16, ny=12):
    """u = (x, -y)-style linear fields on the grid: (t, 2, nx, ny)."""
    x = np.arange(nx)[:, None] * DX * np.ones((1, ny))
    y = np.arange(ny)[None, :] * DY * np.ones((nx, 1))
    return np.broadcast_to(np.stack([x, y]), (n_steps, 2, nx, ny)).copy()


def test_kinetic_energy_of_a_uniform_flow_is_half_the_squared_speed():
    block = np.zeros((3, 2, 5, 6))
    block[:, 0] = 3.0
    block[:, 1] = 4.0

    assert kinetic_energy(block) == pytest.approx(np.full(3, 12.5))


def test_enstrophy_of_a_solid_body_rotation_is_half_the_squared_vorticity():
    # u_x = -y, u_y = x has vorticity 2 everywhere.
    nx, ny = 8, 6
    x = np.arange(nx)[:, None] * DX * np.ones((1, ny))
    y = np.arange(ny)[None, :] * DY * np.ones((nx, 1))
    block = np.stack([-y, x])[None]

    assert enstrophy(block, DX, DY) == pytest.approx([0.5 * 2.0**2])
    assert rms_divergence(block, DX, DY) == pytest.approx([0.0], abs=1e-12)


def test_rms_divergence_of_a_uniformly_expanding_field_is_its_known_divergence():
    # u = (x, y) has divergence 2.
    block = grid_field(2)

    assert rms_divergence(block, DX, DY) == pytest.approx([2.0, 2.0])


def test_log_spectral_distance_is_the_mean_absolute_log10_ratio():
    true = np.array([[5.0, 1.0, 2.0, 4.0]])  # k=0 is ignored
    assert log_spectral_distance(true, true) == pytest.approx(0.0)
    assert log_spectral_distance(true * 10.0, true) == pytest.approx(1.0)
    assert log_spectral_distance(true / 2.0, true) == pytest.approx(np.log10(2.0))


def test_series_summary_is_independent_of_the_chunk_size():
    rng = np.random.default_rng(0)
    series = rng.normal(size=(7, 2, 16, 12))

    a = series_summary(series, DX, DY, chunk_t=7)
    b = series_summary(series, DX, DY, chunk_t=3)

    assert a.keys() == b.keys()
    for key in a:
        assert b[key] == pytest.approx(a[key])
    assert a["energy"].shape == (7,)
    assert a["spectrum_x"].shape == (2, 16 // 2 + 1)
    assert a["spectrum_y"].shape == (2, 12 // 2 + 1)


def test_compare_diagnostics_of_a_perfect_forecast_shows_no_error():
    rng = np.random.default_rng(1)
    target = rng.normal(size=(5, 2, 16, 12))

    scores = compare_diagnostics(target, target, DX, DY)

    assert scores["energy_rel_error"] == pytest.approx(0.0, abs=1e-12)
    assert scores["enstrophy_rel_error"] == pytest.approx(0.0, abs=1e-12)
    assert scores["divergence_ratio"] == pytest.approx(1.0)
    assert scores["spectrum_x_lsd"] == pytest.approx(0.0, abs=1e-9)
    assert scores["spectrum_y_lsd"] == pytest.approx(0.0, abs=1e-9)


def test_compare_diagnostics_scales_as_expected_for_a_doubled_field():
    """Doubling the field multiplies energy, enstrophy and spectral power by 4
    and the divergence by 2, so every score has a known value."""
    rng = np.random.default_rng(2)
    target = rng.normal(size=(5, 2, 16, 12))

    scores = compare_diagnostics(2.0 * target, target, DX, DY, chunk_t=2)

    assert scores["energy_rel_error"] == pytest.approx(3.0)
    assert scores["enstrophy_rel_error"] == pytest.approx(3.0)
    assert scores["divergence_ratio"] == pytest.approx(2.0)
    assert scores["spectrum_x_lsd"] == pytest.approx(np.log10(4.0))
    assert scores["spectrum_y_lsd"] == pytest.approx(np.log10(4.0))


def test_a_smoothed_forecast_loses_small_scale_power():
    """Replacing the field by its mean over x (what regressing toward the mean
    does) keeps the large scales but removes all x-structure: the x-spectrum
    distance is large while a perfect forecast scores zero."""
    rng = np.random.default_rng(3)
    target = rng.normal(size=(4, 2, 16, 12))
    smoothed = np.broadcast_to(target.mean(axis=2, keepdims=True), target.shape)

    scores = compare_diagnostics(smoothed, target, DX, DY)

    assert scores["spectrum_x_lsd"] > 1.0
    assert scores["energy_rel_error"] > 0.5
