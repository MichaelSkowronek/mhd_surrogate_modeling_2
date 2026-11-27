import numpy as np
import pytest
import zarr

from mhd_surrogate.data.climatology import (
    Climatology,
    block_std,
    climatology_from_series,
    domain_mean_series,
    store_domain_means,
    window_statistics,
)


def frames_with_means(means: np.ndarray, shape=(4, 3)) -> np.ndarray:
    """(T, C, Nx, Ny) frames whose domain means are `means` (T, C): each
    channel a zero-mean spatial pattern plus its mean."""
    pattern = np.arange(np.prod(shape), dtype=float).reshape(shape)
    pattern -= pattern.mean()
    return means[:, :, None, None] + 0.1 * pattern[None, None]


def test_the_domain_mean_series_is_each_frames_spatial_mean_whatever_the_chunks():
    means = np.random.default_rng(0).normal(size=(11, 2))
    frames = frames_with_means(means)

    for chunk_t in (1, 4, 32):
        np.testing.assert_allclose(domain_mean_series(frames, chunk_t), means, atol=1e-12)


def test_block_std_is_the_std_of_the_full_blocks_means():
    series = np.repeat([[0.0], [2.0], [4.0]], 5, axis=0)  # block means 0, 2, 4
    series = np.concatenate([series, [[100.0]]])  # a partial block, left out

    assert block_std(series, 5) == pytest.approx([np.std([0.0, 2.0, 4.0])])
    with pytest.raises(ValueError, match="fewer than 2 blocks"):
        block_std(series[:9], 5)


def test_window_statistics_cover_every_window_from_the_start_every_stride():
    series = np.arange(20, dtype=float)[:, None]  # a ramp

    means, stds, slow = window_statistics(series, 10, 5, 5)

    assert means[:, 0].tolist() == [4.5, 9.5, 14.5]  # windows from 0, 5, 10
    np.testing.assert_allclose(stds[:, 0], np.arange(10).std())
    np.testing.assert_allclose(slow[:, 0], np.std([2.0, 7.0]))  # its two block means
    with pytest.raises(ValueError, match="no window of 21"):
        window_statistics(series, 21, 5, 5)


def climatology_of(series_by_name, window=10, stride=5, block=5):
    return climatology_from_series(series_by_name, ["u_x", "u_y"], window, stride, block)


def test_the_climatology_pools_the_datasets_windows():
    rng = np.random.default_rng(1)
    series = {"a": rng.normal(1.0, 0.1, size=(30, 2)), "b": rng.normal(1.2, 0.2, size=(20, 2))}

    climatology = climatology_of(series)

    stats = [window_statistics(s, 10, 5, 5) for s in series.values()]
    means, stds, slow = (np.concatenate(parts) for parts in zip(*stats))
    assert climatology.n_windows == 5 + 3
    np.testing.assert_allclose(climatology.level, means.mean(axis=0))
    np.testing.assert_allclose(climatology.unit, np.median(stds, axis=0))
    np.testing.assert_allclose(climatology.slow_std, np.median(slow, axis=0))
    offsets = np.abs(means - means.mean(axis=0)) / np.median(stds, axis=0)
    np.testing.assert_allclose(climatology.max_window_offset, offsets.max(axis=0))
    ratios = stds / np.median(stds, axis=0)
    assert climatology.std_ratio_range[0] == pytest.approx((ratios[:, 0].min(), ratios[:, 0].max()))
    assert climatology.source_datasets == ["a", "b"]
    # b's windows sit higher: its level pulls the pooled one up.
    assert 1.0 < climatology.level[0] < 1.2


def test_the_climatology_reads_the_datasets_in_order_from_the_store(tmp_path):
    store = zarr.open_group(store=str(tmp_path / "data.zarr"), mode="w")
    rng = np.random.default_rng(2)
    expected = {}
    for name in ("b", "a"):
        means = rng.normal(size=(12, 2))
        store.create_array(name, data=frames_with_means(means))
        expected[name] = means

    series = store_domain_means(tmp_path / "data.zarr", ["b", "a"], chunk_t=5, backend="sequential")

    assert list(series) == ["b", "a"]
    for name, means in expected.items():
        np.testing.assert_allclose(series[name], means, atol=1e-12)


def valid_fields(**overrides):
    fields = dict(
        channel_names=["u_x", "u_y"],
        window_steps=10,
        stride=5,
        block_steps=5,
        level=[1.0, 0.0],
        unit=[0.1, 0.2],
        slow_std=[0.05, 0.05],
        max_window_offset=[0.3, 0.1],
        std_ratio_range=[(0.9, 1.1), (0.95, 1.05)],
        n_windows=4,
        source_datasets=["a", "b"],
    )
    return {**fields, **overrides}


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"channel_names": []}, "channel_names"),
        ({"level": [1.0]}, "level must have one entry per channel"),
        ({"unit": [0.1, 0.0]}, "unit and slow_std"),
        ({"slow_std": [float("nan"), 0.1]}, "unit and slow_std"),
        ({"source_datasets": []}, "source_datasets"),
        ({"window_steps": 0}, "greater than 0"),
    ],
)
def test_a_climatology_refuses_inconsistent_fields(overrides, match):
    with pytest.raises(ValueError, match=match):
        Climatology(**valid_fields(**overrides))


def test_a_climatology_must_come_from_exactly_the_training_datasets():
    climatology = Climatology(**valid_fields())

    climatology.check_source(["b", "a"], "t")
    with pytest.raises(ValueError, match="the test dataset 'a'"):
        climatology.check_source(["a", "b"], "a")
    with pytest.raises(ValueError, match="training datasets are"):
        climatology.check_source(["a", "b", "c"], "t")


def test_a_climatology_round_trips_through_json(tmp_path):
    climatology = Climatology(**valid_fields())

    climatology.save(tmp_path / "sub" / "climatology.json")

    assert Climatology.load(tmp_path / "sub" / "climatology.json") == climatology
