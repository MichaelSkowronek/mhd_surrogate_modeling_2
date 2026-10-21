import shutil

import numpy as np
import pytest

import mlflow
from mhd_surrogate.models.baselines import MeanField
from mhd_surrogate.training import mlflow_utils
from mhd_surrogate.training.export import checkpoint_digest
from mhd_surrogate.training.mlflow_model import log_surrogate
from mhd_surrogate.training.mlflow_utils import (
    finite_metrics,
    flatten_for_mlflow,
    log_checkpoint_once,
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


FRAME = (2, 5, 4)


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    # A sqlite store writes artifacts to ./mlruns; keep them in tmp_path.
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlruns.db'}")
    mlflow.set_experiment("exp")


def mean_field_checkpoint(directory, seed=0):
    model = MeanField()
    model.fit({"a": np.random.default_rng(seed).normal(size=(6, *FRAME)).astype(np.float32)})
    model.save(directory)
    return model


class Logger:
    """`log_model` for log_checkpoint_once, counting its calls."""

    def __init__(self, model, checkpoint):
        self.model, self.checkpoint, self.calls = model, checkpoint, 0

    def __call__(self):
        self.calls += 1
        return log_surrogate(self.model, self.checkpoint, FRAME)


def test_a_checkpoint_is_logged_once_and_reused_by_a_run_with_an_identical_one(tracking, tmp_path):
    model = mean_field_checkpoint(tmp_path / "first")
    shutil.copytree(tmp_path / "first", tmp_path / "copy")
    first, again = Logger(model, tmp_path / "first"), Logger(model, tmp_path / "copy")

    with mlflow.start_run() as run_1:
        model_id = log_checkpoint_once(tmp_path / "first", first)
    with mlflow.start_run() as run_2:
        reused = log_checkpoint_once(tmp_path / "copy", again)

    assert (first.calls, again.calls) == (1, 0)
    assert reused == model_id
    digest = checkpoint_digest(tmp_path / "first")
    assert mlflow.get_logged_model(model_id).tags["checkpoint_sha256"] == digest
    client = mlflow.MlflowClient()
    for run in (run_1, run_2):
        assert client.get_run(run.info.run_id).data.tags["checkpoint_sha256"] == digest
    second = client.get_run(run_2.info.run_id)
    assert second.data.tags["reused_logged_model"] == model_id
    assert [m.model_id for m in second.outputs.model_outputs] == [model_id]
    assert "reused_logged_model" not in client.get_run(run_1.info.run_id).data.tags


def test_a_different_checkpoint_a_deleted_model_or_another_experiment_logs_anew(tracking, tmp_path):
    a = mean_field_checkpoint(tmp_path / "a", seed=0)
    b = mean_field_checkpoint(tmp_path / "b", seed=1)
    with mlflow.start_run():
        first = log_checkpoint_once(tmp_path / "a", Logger(a, tmp_path / "a"))
    with mlflow.start_run():
        other = log_checkpoint_once(tmp_path / "b", Logger(b, tmp_path / "b"))
    mlflow.MlflowClient().delete_logged_model(first)
    with mlflow.start_run():
        after_delete = log_checkpoint_once(tmp_path / "a", Logger(a, tmp_path / "a"))
    mlflow.set_experiment("another")
    with mlflow.start_run():
        elsewhere = log_checkpoint_once(tmp_path / "a", Logger(a, tmp_path / "a"))

    assert len({first, other, after_delete, elsewhere}) == 4
