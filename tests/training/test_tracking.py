import logging

import pytest

import mlflow
from mhd_surrogate.training.tracking import (
    DivergenceError,
    resumed_run_id,
    save_run_record,
    tracked_run,
)
from mlflow import MlflowClient


@pytest.fixture
def log_file(tmp_path):
    """A root-logger file handler, as Hydra's job_logging config installs."""
    path = tmp_path / "run.log"
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    yield path
    root.removeHandler(handler)
    handler.close()


@pytest.fixture
def uri(tmp_path):
    return f"sqlite:///{tmp_path / 'mlruns.db'}"


def _run(uri, run_id):
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    return client.get_run(run_id), client


def test_logs_flattened_params_and_config_artifact(uri):
    config = {"data": {"val": "a"}, "seed": 7}
    with tracked_run(uri, "exp", config) as run:
        pass

    finished, client = _run(uri, run.info.run_id)
    assert finished.data.params == {"data.val": "a", "seed": "7"}
    assert finished.info.status == "FINISHED"
    assert "config.json" in [a.path for a in client.list_artifacts(run.info.run_id)]


def test_uploads_log_file_on_success(uri, log_file):
    with tracked_run(uri, "exp", {}, log_file=log_file) as run:
        logging.getLogger("mhd_surrogate.test").info("epoch 1 done")

    _, client = _run(uri, run.info.run_id)
    assert "run.log" in [a.path for a in client.list_artifacts(run.info.run_id)]


def test_crash_still_uploads_log_with_traceback_and_fails_run(uri, log_file):
    with pytest.raises(ValueError, match="boom"):
        with tracked_run(uri, "exp", {}, log_file=log_file):
            raise ValueError("boom")

    finished = MlflowClient(uri).search_runs([mlflow.get_experiment_by_name("exp").experiment_id])[
        0
    ]
    assert finished.info.status == "FAILED"
    assert "diverged" not in finished.data.tags
    assert "run.log" in [a.path for a in MlflowClient(uri).list_artifacts(finished.info.run_id)]
    assert "ValueError: boom" in log_file.read_text()


def test_divergence_sets_tag_and_fails_run(uri):
    with pytest.raises(DivergenceError):
        with tracked_run(uri, "exp", {}) as run:
            raise DivergenceError("loss is NaN at step 3")

    finished, _ = _run(uri, run.info.run_id)
    assert finished.data.tags["diverged"] == "true"
    assert finished.info.status == "FAILED"


def test_missing_log_file_warns_instead_of_raising(uri, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        with tracked_run(uri, "exp", {}, log_file=tmp_path / "nope.log"):
            pass
    assert "not found" in caplog.text


def test_run_name_is_set_when_given(uri):
    with tracked_run(uri, "exp", {}, run_name="mean_field") as run:
        pass

    finished, _ = _run(uri, run.info.run_id)
    assert finished.info.run_name == "mean_field"


def test_run_id_continues_that_run_and_tags_it_resumed(uri):
    config = {"seed": 7}
    with pytest.raises(RuntimeError):
        with tracked_run(uri, "exp", config, run_name="unet") as first:
            mlflow.log_metric("loss", 1.0, step=1)
            raise RuntimeError("killed")

    with tracked_run(uri, "exp", config, run_name="ignored", run_id=first.info.run_id) as run:
        mlflow.log_metric("loss", 0.5, step=2)

    finished, client = _run(uri, first.info.run_id)
    assert run.info.run_id == first.info.run_id
    assert finished.info.status == "FINISHED"
    assert finished.info.run_name == "unet"
    assert finished.data.tags["resumed"] == "true"
    assert [m.value for m in client.get_metric_history(run.info.run_id, "loss")] == [1.0, 0.5]


def test_run_record_gives_back_the_run_for_the_same_config(tmp_path):
    config = {"model": {"name": "unet", "window": 4}, "seed": 1}
    save_run_record(tmp_path / "state", "abc123", config)

    assert resumed_run_id(tmp_path / "state", config) == "abc123"


def test_run_record_refuses_a_different_config_and_names_what_changed(tmp_path):
    save_run_record(tmp_path, "abc123", {"model": {"window": 4}, "seed": 1})

    with pytest.raises(ValueError, match="model differ"):
        resumed_run_id(tmp_path, {"model": {"window": 8}, "seed": 1})


def test_resuming_without_a_run_record_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="no resumable training run"):
        resumed_run_id(tmp_path, {})
