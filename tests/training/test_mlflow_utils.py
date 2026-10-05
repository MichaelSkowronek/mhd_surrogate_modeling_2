import mlflow
from mhd_surrogate.training import mlflow_utils
from mhd_surrogate.training.mlflow_utils import (
    finite_metrics,
    flatten_for_mlflow,
    log_metric_series,
)


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


def test_finite_metrics_drops_nan_and_inf_and_names_them():
    finite, undefined = finite_metrics({"a": 1.0, "b": float("nan"), "c": float("inf"), "d": 0})

    assert finite == {"a": 1.0, "d": 0.0}
    assert undefined == ["b", "c"]


def test_log_metric_series_logs_each_value_at_its_step_across_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(mlflow_utils, "MAX_BATCH", 3)  # force several batches
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlruns.db'}")
    mlflow.set_experiment("exp")
    values = [0.5, 0.25, 1.0, 2.0, 4.0, 8.0, 3.0]

    with mlflow.start_run() as run:
        log_metric_series("val.rmse", values, start_step=1)

    history = mlflow.MlflowClient().get_metric_history(run.info.run_id, "val.rmse")
    assert sorted((m.step, m.value) for m in history) == list(enumerate(values, start=1))


def test_log_metric_series_links_the_history_to_a_logged_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlruns.db'}")
    mlflow.set_experiment("exp")

    with mlflow.start_run() as run:
        model = mlflow.create_external_model(name="m", source_run_id=run.info.run_id)
        log_metric_series("val.rmse", [0.5, 0.7], start_step=1, model_id=model.model_id)

    logged = mlflow.get_logged_model(model.model_id)
    assert sorted((m.step, m.value) for m in logged.metrics if m.key == "val.rmse") == [
        (1, 0.5),
        (2, 0.7),
    ]
