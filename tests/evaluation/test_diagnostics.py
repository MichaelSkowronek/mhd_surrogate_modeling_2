import numpy as np
import pytest

from mhd_surrogate.evaluation.diagnostics import (
    compare_diagnostics,
    enstrophy,
    kinetic_energy,
    log_spectral_distance,
    mean_drift_scores,
    rms_divergence,
    series_summary,
    temporal_scores,
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
    assert a["mean_velocity"].shape == (7, 2)
    assert a["mean_velocity"] == pytest.approx(series.mean(axis=(2, 3)))
    assert a["spectrum_x"].shape == (2, 16 // 2 + 1)
    assert a["spectrum_y"].shape == (2, 12 // 2 + 1)


def test_compare_diagnostics_of_a_perfect_forecast_shows_no_error():
    rng = np.random.default_rng(1)
    target = rng.normal(size=(5, 2, 16, 12))

    scores = compare_diagnostics(target, target, DX, DY, nperseg=4)

    assert scores["energy_rel_error"] == pytest.approx(0.0, abs=1e-12)
    assert scores["enstrophy_rel_error"] == pytest.approx(0.0, abs=1e-12)
    assert scores["divergence_ratio"] == pytest.approx(1.0)
    assert scores["spectrum_x_lsd"] == pytest.approx(0.0, abs=1e-9)
    assert scores["spectrum_y_lsd"] == pytest.approx(0.0, abs=1e-9)
    assert scores["temporal_u_x_lsd"] == pytest.approx(0.0, abs=1e-9)
    assert scores["temporal_u_y_lsd"] == pytest.approx(0.0, abs=1e-9)
    assert scores["u_y_period_error"] == pytest.approx(0.0, abs=1e-9)
    assert scores["u_y_peak_power_ratio"] == pytest.approx(1.0)


def test_compare_diagnostics_scales_as_expected_for_a_doubled_field():
    """Doubling the field multiplies energy, enstrophy and spectral power by 4
    and the divergence by 2, so every score has a known value."""
    rng = np.random.default_rng(2)
    target = rng.normal(size=(5, 2, 16, 12))

    scores = compare_diagnostics(2.0 * target, target, DX, DY, chunk_t=2, nperseg=4)

    assert scores["energy_rel_error"] == pytest.approx(3.0)
    assert scores["enstrophy_rel_error"] == pytest.approx(3.0)
    assert scores["divergence_ratio"] == pytest.approx(2.0)
    assert scores["spectrum_x_lsd"] == pytest.approx(np.log10(4.0))
    assert scores["spectrum_y_lsd"] == pytest.approx(np.log10(4.0))
    assert scores["temporal_u_y_lsd"] == pytest.approx(np.log10(4.0))
    assert scores["u_y_period_error"] == pytest.approx(0.0, abs=1e-9)
    assert scores["u_y_peak_power_ratio"] == pytest.approx(4.0)


def test_log_spectral_distance_of_a_flat_prediction_is_bounded_not_floor_dominated():
    true = np.array([[0.0, 1.0, 10.0, 100.0]])
    flat = np.zeros_like(true)

    distance = log_spectral_distance(flat, true)

    # The floor is relative to the true spectrum's peak (1e-10 of it), so the
    # distance is a few decades, not set by an arbitrary absolute epsilon.
    assert 5.0 < distance < 12.0


def test_temporal_scores_of_a_constant_forecast_have_no_period():
    """A flat forecast has no oscillation: zero peak power, and its 'period'
    is undefined (nan) rather than numerical noise."""
    true = oscillating_means(36.0)
    flat = np.zeros_like(true)

    scores = temporal_scores(flat, true)

    assert np.isnan(scores["u_y_period_error"])
    assert scores["u_y_peak_power_ratio"] == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(scores["temporal_u_y_lsd"]) and scores["temporal_u_y_lsd"] > 5.0


def test_a_smoothed_forecast_loses_small_scale_power():
    """Replacing the field by its mean over x (what regressing toward the mean
    does) keeps the large scales but removes all x-structure: the x-spectrum
    distance is large while a perfect forecast scores zero."""
    rng = np.random.default_rng(3)
    target = rng.normal(size=(4, 2, 16, 12))
    smoothed = np.broadcast_to(target.mean(axis=2, keepdims=True), target.shape)

    scores = compare_diagnostics(smoothed, target, DX, DY, nperseg=4)

    assert scores["spectrum_x_lsd"] > 1.0
    assert scores["energy_rel_error"] > 0.5


def oscillating_means(period, amplitude=1.0, n=800, seed=0):
    """(T, 2) domain-averaged velocity: u_y a sine of the given period plus noise
    (same noise for any period/amplitude), u_x noise only."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    noise = rng.normal(scale=0.3, size=(n, 2))
    series = noise.copy()
    series[:, 1] += amplitude * np.sin(2 * np.pi * t / period)
    return series


def test_temporal_scores_of_a_perfect_forecast_show_no_error():
    true = oscillating_means(36.0)

    scores = temporal_scores(true, true)

    assert scores["temporal_u_x_lsd"] == pytest.approx(0.0, abs=1e-9)
    assert scores["temporal_u_y_lsd"] == pytest.approx(0.0, abs=1e-9)
    assert scores["u_y_period_error"] == pytest.approx(0.0, abs=1e-9)
    assert scores["u_y_peak_power_ratio"] == pytest.approx(1.0)


def test_temporal_scores_detect_a_damped_oscillation():
    """A forecast with half the amplitude has a quarter of the power at every
    frequency, at the right period."""
    true = oscillating_means(36.0)

    scores = temporal_scores(0.5 * true, true)

    assert scores["u_y_peak_power_ratio"] == pytest.approx(0.25)
    assert scores["u_y_period_error"] == pytest.approx(0.0, abs=1e-9)
    assert scores["temporal_u_y_lsd"] == pytest.approx(np.log10(4.0))


def test_temporal_scores_detect_a_mistimed_oscillation():
    """A period of 50 instead of 36 is a ~39% period error, resolved even
    though the Welch bins are coarse (period 200/k)."""
    true = oscillating_means(36.0)
    mistimed = oscillating_means(50.0)

    scores = temporal_scores(mistimed, true)

    assert scores["u_y_period_error"] == pytest.approx(abs(50.0 / 36.0 - 1.0), abs=0.06)
    # Broadband noise dominates the distance, so a shifted peak moves it only
    # a little: the period error is the score that catches this.
    assert scores["temporal_u_y_lsd"] > 0.05
    # u_x is the same noise in both, so its temporal spectrum is untouched.
    assert scores["temporal_u_x_lsd"] == pytest.approx(0.0, abs=1e-9)


def test_a_drifting_forecast_has_no_period_and_its_drift_is_measured():
    """A forecast whose domain-mean u_y drifts slowly away has its spectral
    peak in the lowest bin: a drift, not a period."""
    t = np.arange(600, dtype=float)
    oscillation = 0.01 * np.sin(2 * np.pi * t / 25.6)
    u_x = 1.0 + 0.01 * np.cos(2 * np.pi * t / 40.0)
    true = np.stack([u_x, oscillation], axis=1)
    drifting = np.stack([u_x, oscillation + 0.2 * t / 600], axis=1)

    scores = temporal_scores(drifting, true, nperseg=200)

    assert np.isnan(scores["u_y_period_error"])
    assert scores["u_y_mean_offset"] > 5
    assert scores["u_y_mean_std_ratio"] > 5


def test_mean_drift_scores_are_in_units_of_the_truths_variability():
    rng = np.random.default_rng(0)
    true = rng.normal(0.0, 2.0, 100_000)

    on_mean = mean_drift_scores(rng.normal(0.0, 2.0, 100_000), true)
    offset = mean_drift_scores(true + 1.0, true)
    damped = mean_drift_scores(0.5 * true, true)

    assert on_mean["u_y_mean_offset"] == pytest.approx(0.0, abs=0.02)
    assert on_mean["u_y_mean_std_ratio"] == pytest.approx(1.0, rel=0.02)
    assert offset["u_y_mean_offset"] == pytest.approx(1.0 / true.std())  # ~0.5
    assert offset["u_y_mean_std_ratio"] == pytest.approx(1.0)
    assert damped["u_y_mean_std_ratio"] == pytest.approx(0.5)
