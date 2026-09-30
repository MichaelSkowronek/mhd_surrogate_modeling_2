import json

import numpy as np
import pytest

from mhd_surrogate.analysis.summary import filter_datasets, write_summary


def test_filter_datasets_returns_everything_when_only_is_none():
    datasets = ["a", "b", "c"]
    assert filter_datasets(datasets, None) == datasets


def test_filter_datasets_returns_everything_when_only_is_empty():
    datasets = ["a", "b"]
    assert filter_datasets(datasets, []) == datasets


def test_filter_datasets_preserves_original_order_not_request_order():
    datasets = ["a", "b", "c"]
    assert filter_datasets(datasets, ["c", "a"]) == ["a", "c"]


def test_filter_datasets_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="unknown dataset"):
        filter_datasets(["a"], ["a", "nonexistent"])


def test_write_summary_round_trips_numpy_scalars_and_arrays(tmp_path):
    data = {
        "a_float64": np.float64(1.5),
        "a_bool": np.bool_(True),
        "an_array": np.array([1.0, 2.0, 3.0]),
        "nested": {"lag1": np.float32(0.9), "zero": None},
    }

    path = write_summary("ds0", "check_x", data, out_dir=tmp_path)

    assert path == tmp_path / "ds0__check_x.json"
    loaded = json.loads(path.read_text())
    assert loaded["dataset"] == "ds0"
    assert loaded["script"] == "check_x"
    assert loaded["a_float64"] == pytest.approx(1.5)
    assert loaded["a_bool"] is True
    assert loaded["an_array"] == [1.0, 2.0, 3.0]
    assert loaded["nested"] == {"lag1": pytest.approx(0.9, abs=1e-6), "zero": None}


def test_write_summary_round_trips_a_complex_number(tmp_path):
    path = write_summary("ds0", "check_x", {"amplitude": 1.5 - 2.0j}, out_dir=tmp_path)

    loaded = json.loads(path.read_text())
    assert loaded["amplitude"] == {"real": pytest.approx(1.5), "imag": pytest.approx(-2.0)}


def test_write_summary_creates_output_directory(tmp_path):
    out_dir = tmp_path / "nested" / "summaries"
    path = write_summary("ds0", "check_x", {}, out_dir=out_dir)
    assert path.exists()


def test_write_summary_rejects_a_genuinely_unsupported_type(tmp_path):
    with pytest.raises(TypeError, match="not JSON serializable"):
        write_summary("ds0", "check_x", {"bad": {1, 2, 3}}, out_dir=tmp_path)
