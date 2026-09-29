import numpy as np
import pytest
from check_spatial_mean_spectrum import spatial_series


def test_spatial_series_basic_values():
    # 2 time steps, 2 channels, 2x2 spatial grid.
    t0 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]])
    t1 = np.array([[[5.0, 5.0], [5.0, 5.0]], [[1.0, 1.0], [1.0, 1.0]]])
    arr = np.stack([t0, t1])  # (2, 2, 2, 2)

    means, energy = spatial_series(arr, chunk_t=1)

    assert means.shape == (2, 2)
    assert means[0, 0] == 2.5  # mean of [1, 2, 3, 4]
    assert means[1, 0] == 5.0
    assert means[0, 1] == 0.0
    assert means[1, 1] == 1.0

    assert energy[0, 0] == pytest.approx(0.5 * np.mean(np.array([1.0, 2.0, 3.0, 4.0]) ** 2))
    assert energy[1, 0] == pytest.approx(0.5 * 5.0**2)
    assert energy[0, 1] == 0.0
    assert energy[1, 1] == pytest.approx(0.5 * 1.0**2)


def test_spatial_series_chunking_does_not_change_result():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((17, 2, 3, 4))

    whole_means, whole_energy = spatial_series(arr, chunk_t=100)
    chunked_means, chunked_energy = spatial_series(arr, chunk_t=5)

    np.testing.assert_allclose(whole_means, chunked_means)
    np.testing.assert_allclose(whole_energy, chunked_energy)


def test_spatial_series_respects_n_steps_bound():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((20, 2, 3, 4))

    bounded_means, bounded_energy = spatial_series(arr, chunk_t=4, n_steps=8)
    whole_means, whole_energy = spatial_series(arr, chunk_t=4)

    np.testing.assert_allclose(bounded_means, whole_means[:8])
    np.testing.assert_allclose(bounded_energy, whole_energy[:8])
