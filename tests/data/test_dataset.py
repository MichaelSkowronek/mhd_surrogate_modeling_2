import json

import numpy as np
import pytest
import zarr

from mhd_surrogate.data.dataset import WindowedDataset, load_dataset


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


def test_load_dataset_reads_manifest_and_zarr_store(tmp_path):
    store_path = tmp_path / "store.zarr"
    root = zarr.open_group(store=str(store_path), mode="w")
    data = np.arange(10 * 2 * 3 * 4).reshape(10, 2, 3, 4).astype(np.float32)
    root.create_array("ds0", data=data)

    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "config": {"zarr_store": str(store_path)},
        "splits": {"ds0": {"n_steps": 10, "train": [0, 7], "test": [7, 10]}},
    }
    manifest_path.write_text(json.dumps(manifest))

    train_ds = load_dataset(manifest_path, "ds0", "train", window=2, horizon=1, stride=1)
    test_ds = load_dataset(manifest_path, "ds0", "test", window=2, horizon=1, stride=1)

    assert len(train_ds) == 5  # (7 - 3) // 1 + 1
    assert len(test_ds) == 1  # (3 - 3) // 1 + 1

    x, y = test_ds[0]
    np.testing.assert_array_equal(np.asarray(x), data[7:9])
    np.testing.assert_array_equal(np.asarray(y), data[9:10])
