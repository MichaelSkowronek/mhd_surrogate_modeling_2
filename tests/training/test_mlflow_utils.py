from mhd_surrogate.training.mlflow_utils import flatten_for_mlflow


def test_flatten_for_mlflow_flattens_nested_dict():
    data = {"data": {"dataset": "re16k_t400_0", "zarr_store": "x.zarr"}, "seed": 42}
    assert flatten_for_mlflow(data) == {
        "data.dataset": "re16k_t400_0",
        "data.zarr_store": "x.zarr",
        "seed": 42,
    }


def test_flatten_for_mlflow_handles_multiple_nesting_levels():
    data = {"a": {"b": {"c": 1}}}
    assert flatten_for_mlflow(data) == {"a.b.c": 1}


def test_flatten_for_mlflow_flat_dict_is_unchanged():
    data = {"x": 1, "y": "two"}
    assert flatten_for_mlflow(data) == data


def test_flatten_for_mlflow_empty_dict():
    assert flatten_for_mlflow({}) == {}
