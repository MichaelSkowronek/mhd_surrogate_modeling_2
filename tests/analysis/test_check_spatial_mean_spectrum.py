import numpy as np
from check_spatial_mean_spectrum import spatial_mean_series


def test_spatial_mean_series_basic_values():
    # 2 time steps, 2 channels, 2x2 spatial grid.
    t0 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]])
    t1 = np.array([[[5.0, 5.0], [5.0, 5.0]], [[1.0, 1.0], [1.0, 1.0]]])
    arr = np.stack([t0, t1])  # (2, 2, 2, 2)

    means = spatial_mean_series(arr, chunk_t=1)

    assert means.shape == (2, 2)
    assert means[0, 0] == 2.5  # mean of [1, 2, 3, 4]
    assert means[1, 0] == 5.0
    assert means[0, 1] == 0.0
    assert means[1, 1] == 1.0


def test_spatial_mean_series_chunking_does_not_change_result():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((17, 2, 3, 4))

    whole = spatial_mean_series(arr, chunk_t=100)
    chunked = spatial_mean_series(arr, chunk_t=5)

    np.testing.assert_allclose(whole, chunked)


def test_spatial_mean_series_respects_n_steps_bound():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((20, 2, 3, 4))

    bounded = spatial_mean_series(arr, chunk_t=4, n_steps=8)
    whole = spatial_mean_series(arr, chunk_t=4)

    np.testing.assert_allclose(bounded, whole[:8])
