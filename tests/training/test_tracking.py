import logging

import pytest

import mlflow
from mhd_surrogate.training.tracking import DivergenceError, tracked_run
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
