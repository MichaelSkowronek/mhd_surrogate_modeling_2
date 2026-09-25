import numpy as np
import pytest
from check_split import channel_labels, per_timestep_stats


def test_per_timestep_stats_basic_values():
    # 2 time steps, 2 channels, 2x2 spatial grid.
    t0 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]])
    t1 = np.array([[[5.0, 5.0], [5.0, 5.0]], [[1.0, 1.0], [1.0, 1.0]]])
    arr = np.stack([t0, t1])  # (2, 2, 2, 2)

    means, stds, mins, maxs, energy, correlation = per_timestep_stats(arr, chunk_t=1)

    assert means[0, 0] == pytest.approx(2.5)
    assert means[1, 0] == pytest.approx(5.0)
    assert means[0, 1] == pytest.approx(0.0)
    assert means[1, 1] == pytest.approx(1.0)
    assert mins[0, 0] == 1.0
    assert maxs[0, 0] == 4.0
    assert energy[0, 0] == pytest.approx(0.5 * np.mean(np.array([1.0, 2.0, 3.0, 4.0]) ** 2))
    assert correlation.shape == (2,)


def test_per_timestep_stats_correlation_extremes():
    ux = np.array([[1.0, 2.0], [3.0, 4.0]])

    perfectly_correlated = np.stack([ux, ux])[None]
    _, _, _, _, _, corr_pos = per_timestep_stats(perfectly_correlated, chunk_t=1)
    assert corr_pos[0] == pytest.approx(1.0)

    anti_correlated = np.stack([ux, -ux])[None]
    _, _, _, _, _, corr_neg = per_timestep_stats(anti_correlated, chunk_t=1)
    assert corr_neg[0] == pytest.approx(-1.0)


def test_per_timestep_stats_no_correlation_for_single_channel():
    arr = np.zeros((3, 1, 2, 2))
    *_, correlation = per_timestep_stats(arr, chunk_t=2)
    assert correlation is None


def test_per_timestep_stats_chunking_does_not_change_result():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((17, 2, 3, 4))

    whole = per_timestep_stats(arr, chunk_t=100)
    chunked = per_timestep_stats(arr, chunk_t=5)

    for whole_part, chunked_part in zip(whole, chunked):
        np.testing.assert_allclose(whole_part, chunked_part)


def test_channel_labels_uses_known_names_only_for_two_channels():
    assert channel_labels(2) == ["u_x", "u_y"]
    assert channel_labels(1) == ["channel=0"]
    assert channel_labels(3) == ["channel=0", "channel=1", "channel=2"]
