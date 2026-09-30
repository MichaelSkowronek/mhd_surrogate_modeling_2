import numpy as np
import pytest
from check_stationarity import analyze_half, half_windows, per_timestep_summary, relative_change


def test_per_timestep_summary_basic_values():
    # 2 time steps, 2 channels, 2x2 spatial grid.
    t0 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]])
    t1 = np.array([[[5.0, 5.0], [5.0, 5.0]], [[1.0, 1.0], [1.0, 1.0]]])
    arr = np.stack([t0, t1])  # (2, 2, 2, 2)

    means, stds, energy = per_timestep_summary(arr, chunk_t=1, n_steps=2)

    assert means[0, 0] == pytest.approx(2.5)
    assert means[1, 0] == pytest.approx(5.0)
    assert means[0, 1] == pytest.approx(0.0)
    assert means[1, 1] == pytest.approx(1.0)
    assert stds[1, 1] == pytest.approx(0.0)
    assert energy[0, 0] == pytest.approx(0.5 * np.mean(np.array([1.0, 2.0, 3.0, 4.0]) ** 2))


def test_per_timestep_summary_respects_n_steps_and_chunking():
    arr = np.arange(4 * 1 * 2 * 2, dtype=np.float64).reshape(4, 1, 2, 2)

    full, _, _ = per_timestep_summary(arr, chunk_t=1, n_steps=4)
    chunked, _, _ = per_timestep_summary(arr, chunk_t=3, n_steps=4)
    bounded, _, _ = per_timestep_summary(arr, chunk_t=2, n_steps=2)

    np.testing.assert_allclose(full, chunked)
    assert bounded.shape == (2, 1)
    np.testing.assert_allclose(bounded, full[:2])


def test_half_windows_splits_even_n_steps_evenly():
    first, second = half_windows(10)
    assert (first.start, first.stop) == (0, 5)
    assert (second.start, second.stop) == (5, 10)


def test_half_windows_gives_the_extra_step_to_the_second_half():
    first, second = half_windows(11)
    assert first.stop - first.start == 5
    assert second.stop - second.start == 6


def test_relative_change_sign_and_magnitude():
    assert relative_change(100.0, 110.0) == pytest.approx(0.1)
    assert relative_change(100.0, 90.0) == pytest.approx(-0.1)
    assert relative_change(-50.0, -75.0) == pytest.approx(-0.5)


def test_analyze_half_recovers_a_known_period_and_aggregate_stats():
    n, t0 = 200, 20  # t0 divides n exactly: lands on a periodogram bin
    t = np.arange(n)
    means = np.zeros((n, 1))
    means[:, 0] = 3.0 + np.sin(2 * np.pi * t / t0)
    stds = np.full((n, 1), 0.5)
    energy = np.full((n, 1), 1.5)

    result = analyze_half(means, stds, energy, n_peaks=1)

    assert set(result) == {"u_x"}
    assert result["u_x"]["std"] == pytest.approx(0.5)
    assert result["u_x"]["energy"] == pytest.approx(1.5)
    assert result["u_x"]["top_periods"][0]["period"] == pytest.approx(t0, rel=0.02)
