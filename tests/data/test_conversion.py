import numpy as np
import pytest
import zarr

from mhd_surrogate.data.conversion import convert_array, convert_to_zarr


@pytest.fixture
def raw_dir(tmp_path):
    rng = np.random.default_rng(0)
    d = tmp_path / "raw"
    d.mkdir()
    data = {}
    for name, t in [("sim_0", 10), ("sim_1", 7), ("sim_2", 3)]:
        data[name] = rng.standard_normal((t, 2, 4, 5)).astype(np.float32)
        np.save(d / f"{name}.npy", data[name])
    return d, data


def test_convert_array_writes_data_and_attrs(tmp_path, raw_dir):
    d, data = raw_dir
    store = tmp_path / "out.zarr"
    zarr.open_group(store=str(store), mode="w")

    name = convert_array(str(d / "sim_1.npy"), str(store), chunk_t=4, overwrite=False)

    assert name == "sim_1"
    arr = zarr.open_group(store=str(store), mode="r")["sim_1"]
    np.testing.assert_array_equal(arr[:], data["sim_1"])
    assert arr.chunks == (4, 2, 4, 5)
    assert arr.attrs["source_file"] == "sim_1.npy"
    assert arr.attrs["n_steps"] == 7


def test_convert_array_chunk_is_capped_by_series_length(tmp_path, raw_dir):
    d, _ = raw_dir
    store = tmp_path / "out.zarr"
    zarr.open_group(store=str(store), mode="w")

    convert_array(str(d / "sim_2.npy"), str(store), chunk_t=32, overwrite=False)

    assert zarr.open_group(store=str(store), mode="r")["sim_2"].chunks[0] == 3


@pytest.mark.parametrize("backend", ["sequential", "processes", "ray"])
def test_convert_to_zarr_on_every_backend(tmp_path, raw_dir, backend):
    d, data = raw_dir
    out = tmp_path / "out.zarr"

    names = convert_to_zarr(
        sorted(d.glob("*.npy")),
        out,
        chunk_t=4,
        overwrite=True,
        description="test store",
        channel_names=["u_x", "u_y"],
        excluded_files=["bad.npy"],
        backend=backend,
        workers=2,
    )

    assert names == ["sim_0", "sim_1", "sim_2"]
    root = zarr.open_group(store=str(out), mode="r")
    assert root.attrs["channel_names"] == ["u_x", "u_y"]
    assert root.attrs["excluded_files"] == ["bad.npy"]
    assert root.attrs["description"] == "test store"
    for name, expected in data.items():
        np.testing.assert_array_equal(root[name][:], expected)
