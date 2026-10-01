import numpy as np
import pytest
from pydantic import ValidationError

from mhd_surrogate.data.normalization import (
    NormalizationStats,
    Normalizer,
    channel_moments,
    compute_normalization_stats,
    merge_moments,
)

CHANNELS = ["u_x", "u_y"]


def _field(rng, shape, mean, std):
    """(T, C, H, W) float32 field with per-channel known mean/std."""
    scale = np.array(std)[None, :, None, None]
    shift = np.array(mean)[None, :, None, None]
    return (rng.standard_normal(shape) * scale + shift).astype(np.float32)


def _stats(**overrides):
    kwargs = dict(
        channel_names=CHANNELS,
        mean=[1.0, 0.0],
        std=[0.8, 0.4],
        n_samples=100,
        source_datasets=["a", "b"],
    )
    kwargs.update(overrides)
    return NormalizationStats(**kwargs)


def test_channel_moments_match_numpy_for_any_chunking():
    rng = np.random.default_rng(0)
    arr = _field(rng, (23, 2, 5, 6), mean=[1.0, 0.0], std=[0.8, 0.4])
    ref = arr.astype(np.float64)

    for chunk_t in (1, 4, 23, 100):
        n, mean, m2 = channel_moments(arr, chunk_t)
        assert n == 23 * 5 * 6
        np.testing.assert_allclose(mean, ref.mean(axis=(0, 2, 3)), rtol=1e-12)
        np.testing.assert_allclose(m2 / n, ref.var(axis=(0, 2, 3)), rtol=1e-10)


def test_channel_moments_rejects_bad_chunk():
    with pytest.raises(ValueError, match="chunk_t"):
        channel_moments(np.zeros((4, 2, 3, 3)), chunk_t=0)


def test_merge_moments_with_empty_is_identity():
    m = (10, np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    empty = (0, np.zeros(2), np.zeros(2))
    assert merge_moments(empty, empty)[0] == 0
    n, mean, m2 = merge_moments(empty, m)
    assert n == 10
    np.testing.assert_array_equal(mean, m[1])
    np.testing.assert_array_equal(m2, m[2])


def test_compute_stats_recovers_known_mean_and_std_pooled_across_datasets():
    rng = np.random.default_rng(1)
    # Different lengths per dataset, as in the real store; same distribution.
    arrays = {
        "a": _field(rng, (40, 2, 16, 16), mean=[1.0, 0.0], std=[0.8, 0.42]),
        "b": _field(rng, (25, 2, 16, 16), mean=[1.0, 0.0], std=[0.8, 0.42]),
    }
    stats = compute_normalization_stats(arrays, CHANNELS, chunk_t=7)

    pooled = np.concatenate([a.astype(np.float64) for a in arrays.values()])
    np.testing.assert_allclose(stats.mean, pooled.mean(axis=(0, 2, 3)), rtol=1e-10)
    np.testing.assert_allclose(stats.std, pooled.std(axis=(0, 2, 3)), rtol=1e-10)
    np.testing.assert_allclose(stats.mean, [1.0, 0.0], atol=0.02)
    np.testing.assert_allclose(stats.std, [0.8, 0.42], rtol=0.02)
    assert stats.n_samples == 65 * 16 * 16
    assert stats.source_datasets == ["a", "b"]


def test_pooled_std_includes_between_dataset_mean_shift():
    # Two constant-valued datasets with different means: the pooled std is
    # all between-dataset spread, which a mean-of-stds would miss (-> 0).
    a = np.full((4, 1, 3, 3), 1.0, dtype=np.float32)
    b = np.full((4, 1, 3, 3), 3.0, dtype=np.float32)
    stats = compute_normalization_stats({"a": a, "b": b}, ["u"])
    assert stats.mean == [2.0]
    np.testing.assert_allclose(stats.std, [1.0])


def test_compute_stats_channel_count_mismatch():
    with pytest.raises(ValueError, match="channels"):
        compute_normalization_stats({"a": np.zeros((4, 3, 2, 2))}, CHANNELS)


def test_compute_stats_constant_field_is_rejected():
    # std == 0 can't be normalized by; the contract refuses it.
    with pytest.raises(ValidationError, match="std"):
        compute_normalization_stats({"a": np.ones((4, 2, 3, 3))}, CHANNELS)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"channel_names": []}, "must not be empty"),
        ({"channel_names": ["u_x", "u_x"]}, "duplicate channel_names"),
        ({"mean": [1.0]}, "one entry per channel"),
        ({"std": [0.8]}, "one entry per channel"),
        ({"mean": [float("nan"), 0.0]}, "finite"),
        ({"std": [0.8, 0.0]}, "> 0"),
        ({"std": [0.8, -1.0]}, "> 0"),
        ({"std": [float("inf"), 0.4]}, "> 0"),
        ({"source_datasets": []}, "must not be empty"),
        ({"source_datasets": ["a", "a"]}, "duplicate source_datasets"),
        ({"n_samples": 0}, "n_samples"),
    ],
)
def test_stats_validation_rejects(overrides, match):
    with pytest.raises(ValidationError, match=match):
        _stats(**overrides)


def test_save_load_roundtrip(tmp_path):
    stats = _stats()
    path = tmp_path / "nested" / "stats.json"
    stats.save(path)
    assert NormalizationStats.load(path) == stats


def test_load_rejects_corrupted_file(tmp_path):
    path = tmp_path / "stats.json"
    _stats().save(path)
    path.write_text(path.read_text().replace("0.4", "-0.4"))
    with pytest.raises(ValidationError, match="> 0"):
        NormalizationStats.load(path)


def test_check_compatible_accepts_matching_setup():
    _stats().check_compatible(CHANNELS, ["b", "a"], test_dataset="t")


@pytest.mark.parametrize(
    "channels, train, test, match",
    [
        (CHANNELS, ["a", "b"], "b", "test dataset"),
        (["u_y", "u_x"], ["a", "b"], "t", "channel mismatch"),
        (CHANNELS, ["a", "c"], "t", "training datasets"),
        (CHANNELS, ["a"], "t", "training datasets"),
    ],
)
def test_check_compatible_rejects(channels, train, test, match):
    with pytest.raises(ValueError, match=match):
        _stats().check_compatible(channels, train, test)


def test_compute_stats_with_no_datasets_is_rejected():
    with pytest.raises(ValidationError, match="n_samples"):
        compute_normalization_stats({}, CHANNELS)


def test_effective_std_per_channel_and_shared():
    stats = _stats(std=[0.8, 0.6])
    assert stats.effective_std("per_channel") == [0.8, 0.6]
    # RMS of (0.8, 0.6) = sqrt((0.64 + 0.36) / 2)
    np.testing.assert_allclose(stats.effective_std("shared"), [np.sqrt(0.5)] * 2)


def test_effective_std_rejects_unknown_mode():
    with pytest.raises(ValueError, match="std_mode"):
        _stats().effective_std("per_pixel")


def test_normalizer_per_channel_gives_zero_mean_unit_std():
    rng = np.random.default_rng(2)
    arr = _field(rng, (30, 2, 12, 12), mean=[1.0, 0.0], std=[0.8, 0.42])
    stats = compute_normalization_stats({"a": arr}, CHANNELS)

    out = Normalizer.from_stats(stats, "per_channel")(arr)

    assert out.dtype == np.float32
    np.testing.assert_allclose(out.mean(axis=(0, 2, 3)), 0.0, atol=1e-5)
    np.testing.assert_allclose(out.std(axis=(0, 2, 3)), 1.0, atol=1e-5)


def test_normalizer_shared_keeps_relative_channel_amplitudes():
    rng = np.random.default_rng(3)
    arr = _field(rng, (30, 2, 12, 12), mean=[1.0, 0.0], std=[0.8, 0.42])
    stats = compute_normalization_stats({"a": arr}, CHANNELS)

    out = Normalizer.from_stats(stats, "shared")(arr)

    np.testing.assert_allclose(out.mean(axis=(0, 2, 3)), 0.0, atol=1e-5)
    std = out.std(axis=(0, 2, 3))
    np.testing.assert_allclose(std[1] / std[0], 0.42 / 0.8, rtol=0.02)
    # The shared scalar is the RMS of the channel stds, so the mean variance is 1.
    np.testing.assert_allclose(np.sqrt((std**2).mean()), 1.0, rtol=1e-4)


@pytest.mark.parametrize("std_mode", ["per_channel", "shared"])
def test_normalizer_inverse_roundtrip_on_leading_dims(std_mode):
    rng = np.random.default_rng(4)
    # (window, C, H, W) as WindowedDataset passes it
    x = _field(rng, (4, 2, 5, 5), mean=[1.0, 0.0], std=[0.8, 0.42])
    normalizer = Normalizer.from_stats(_stats(), std_mode)

    np.testing.assert_allclose(normalizer.inverse(normalizer(x)), x, atol=1e-6)
