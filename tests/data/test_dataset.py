import numpy as np
import pytest
import zarr

from mhd_surrogate.data.dataset import WindowedDataset, load_full_dataset
from mhd_surrogate.data.normalization import NormalizationStats, Normalizer


def test_windowed_dataset_length_and_sample_content():
    arr = np.arange(20 * 2 * 3 * 4).reshape(20, 2, 3, 4).astype(np.float32)
    ds = WindowedDataset(arr, start=0, end=20, window=4, horizon=1, stride=1)

    assert len(ds) == 16  # (20 - 5) // 1 + 1

    x, y = ds[0]
    assert x.shape == (4, 2, 3, 4)
    assert y.shape == (1, 2, 3, 4)
    np.testing.assert_array_equal(np.asarray(x), arr[0:4])
    np.testing.assert_array_equal(np.asarray(y), arr[4:5])

    x_last, y_last = ds[len(ds) - 1]
    np.testing.assert_array_equal(np.asarray(x_last), arr[15:19])
    np.testing.assert_array_equal(np.asarray(y_last), arr[19:20])


def test_windowed_dataset_respects_region_start():
    arr = np.arange(20 * 2 * 3 * 4).reshape(20, 2, 3, 4).astype(np.float32)
    ds = WindowedDataset(arr, start=10, end=20, window=4, horizon=1, stride=1)

    assert len(ds) == 6  # (10 - 5) // 1 + 1
    x, _ = ds[0]
    np.testing.assert_array_equal(np.asarray(x), arr[10:14])


def test_windowed_dataset_stride():
    arr = np.zeros((20, 2, 3, 4), dtype=np.float32)
    ds = WindowedDataset(arr, start=0, end=20, window=4, horizon=1, stride=3)
    assert len(ds) == (20 - 5) // 3 + 1


def test_windowed_dataset_index_errors():
    arr = np.zeros((20, 2, 3, 4), dtype=np.float32)
    ds = WindowedDataset(arr, start=0, end=20, window=4, horizon=1, stride=1)
    with pytest.raises(IndexError):
        ds[len(ds)]
    with pytest.raises(IndexError):
        ds[-1]  # deliberately not Python-sequence-like, see WindowedDataset docs


def test_windowed_dataset_rejects_region_too_short_for_window_and_horizon():
    arr = np.zeros((3, 2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        WindowedDataset(arr, start=0, end=3, window=4, horizon=1, stride=1)


def test_load_full_dataset_covers_the_whole_array(tmp_path):
    store_path = tmp_path / "store.zarr"
    root = zarr.open_group(store=str(store_path), mode="w")
    data = np.arange(10 * 2 * 3 * 4).reshape(10, 2, 3, 4).astype(np.float32)
    root.create_array("ds0", data=data)

    ds = load_full_dataset(store_path, "ds0", window=2, horizon=1, stride=1)

    assert len(ds) == 8  # (10 - 3) // 1 + 1
    x, y = ds[0]
    np.testing.assert_array_equal(np.asarray(x), data[0:2])
    np.testing.assert_array_equal(np.asarray(y), data[2:3])
    x_last, y_last = ds[len(ds) - 1]
    np.testing.assert_array_equal(np.asarray(x_last), data[7:9])
    np.testing.assert_array_equal(np.asarray(y_last), data[9:10])


def _normalizer():
    stats = NormalizationStats(
        channel_names=["u_x", "u_y"],
        mean=[1.0, 0.0],
        std=[0.5, 2.0],
        n_samples=10,
        source_datasets=["a"],
    )
    return Normalizer.from_stats(stats, "per_channel")


def test_windowed_dataset_normalizes_input_and_target():
    arr = np.arange(20 * 2 * 3 * 4).reshape(20, 2, 3, 4).astype(np.float32)
    normalizer = _normalizer()
    ds = WindowedDataset(arr, 0, 20, window=4, horizon=1, stride=1, normalizer=normalizer)

    x, y = ds[2]

    np.testing.assert_allclose(np.asarray(x), normalizer(arr[2:6]), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(y), normalizer(arr[6:7]), rtol=1e-6)
    assert x.dtype == np.float32


def test_load_full_dataset_passes_normalizer_through(tmp_path):
    store_path = tmp_path / "store.zarr"
    root = zarr.open_group(store=str(store_path), mode="w")
    data = np.random.default_rng(0).standard_normal((10, 2, 3, 3)).astype(np.float32)
    root.create_array("ds0", data=data)
    normalizer = _normalizer()

    ds = load_full_dataset(store_path, "ds0", 2, 1, 1, normalizer=normalizer)

    x, _ = ds[0]
    np.testing.assert_allclose(np.asarray(x), normalizer(data[0:2]), rtol=1e-6)
