import logging
from types import SimpleNamespace

import pandas as pd
import pytest

import mlflow
from mhd_surrogate.training import run as run_module
from mhd_surrogate.training.tuning import (
    TRIAL_LOG,
    apply_overrides,
    close_stopped_runs,
    default_point,
    recording_asha,
    run_trial,
    search_space,
    setup_trial_logging,
    summary_rows,
    trial_name,
)
from mlflow import MlflowClient

# -- search space and overrides -------------------------------------------------


def test_search_space_builds_each_domain_type():
    from ray.tune.search.sample import Categorical, Float, Integer

    space = search_space(
        {
            "a.lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
            "a.x": {"type": "uniform", "low": 0.0, "high": 1.0},
            "a.n": {"type": "randint", "low": 1, "high": 5},
            "a.c": {"type": "choice", "values": [16, 32]},
        }
    )

    assert isinstance(space["a.lr"], Float) and space["a.lr"].lower == 1e-4
    assert isinstance(space["a.x"], Float)
    assert isinstance(space["a.n"], Integer) and space["a.n"].upper == 5
    assert isinstance(space["a.c"], Categorical) and space["a.c"].categories == [16, 32]


def test_search_space_rejects_an_unknown_type():
    with pytest.raises(ValueError, match="unknown type 'normal'"):
        search_space({"a": {"type": "normal"}})


SPACE = {
    "model.training.learning_rate": {"type": "loguniform", "low": 1e-4, "high": 3e-3},
    "model.depth": {"type": "randint", "low": 3, "high": 5},
    "model.window": {"type": "choice", "values": [1, 2, 4]},
}


def test_default_point_takes_the_configs_values_of_the_searched_keys():
    config = {"model": {"window": 4, "depth": 4, "training": {"learning_rate": 1e-3}}}

    point = default_point(config, SPACE)

    assert point == {"model.training.learning_rate": 1e-3, "model.depth": 4, "model.window": 4}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("window", 8),  # not one of the choices
        ("depth", 5),  # randint's high is exclusive
        ("learning_rate", 1e-2),  # above the float range
    ],
)
def test_default_point_rejects_a_default_outside_the_space(key, value):
    config = {"model": {"window": 4, "depth": 4, "training": {"learning_rate": 1e-3}}}
    section = config["model"]["training"] if key == "learning_rate" else config["model"]
    section[key] = value

    with pytest.raises(ValueError, match="outside the search space"):
        default_point(config, SPACE)


def test_default_point_accepts_the_float_ranges_upper_end():
    config = {"model": {"window": 4, "depth": 3, "training": {"learning_rate": 3e-3}}}

    assert default_point(config, SPACE)["model.training.learning_rate"] == 3e-3


@pytest.mark.parametrize(
    ("config", "key"), [({"model": {"depth": 4}}, "model.window"), ({"model": 1}, "model.depth")]
)
def test_default_point_rejects_a_key_the_config_lacks(config, key):
    with pytest.raises(KeyError, match="not a key"):
        default_point(config, {key: {"type": "choice", "values": [1]}})


def test_apply_overrides_sets_dotted_keys_on_a_copy():
    config = {"model": {"window": 4, "training": {"learning_rate": 1e-3}}, "seed": 1}

    out = apply_overrides(config, {"model.training.learning_rate": 3e-4, "seed": 2})

    assert out == {"model": {"window": 4, "training": {"learning_rate": 3e-4}}, "seed": 2}
    assert config["model"]["training"]["learning_rate"] == 1e-3


@pytest.mark.parametrize("key", ["model.windw", "modl.window", "model.window.x"])
def test_apply_overrides_rejects_keys_that_do_not_exist(key):
    with pytest.raises(KeyError):
        apply_overrides({"model": {"window": 4}}, {key: 1})


def test_trial_name_shows_the_sampled_values_compactly():
    name = trial_name("unet", {"model.window": 4, "model.training.learning_rate": 0.000123456})

    assert name == "unet model.window=4 model.training.learning_rate=0.000123"


def test_trial_logging_writes_project_debug_records_to_the_file(tmp_path):
    handler = setup_trial_logging(tmp_path / "trial" / TRIAL_LOG)
    try:
        logging.getLogger("mhd_surrogate.test").debug("epoch 1 done")
        handler.flush()
    finally:
        logging.getLogger("mhd_surrogate").removeHandler(handler)
        handler.close()

    assert "epoch 1 done" in (tmp_path / "trial" / TRIAL_LOG).read_text()


# -- the trial ------------------------------------------------------------------


def test_run_trial_trains_the_overridden_config_and_reports_each_validation(tmp_path, monkeypatch):
    from ray import tune

    monkeypatch.chdir(tmp_path)  # run_trial changes directory; restore it after
    reports, calls = [], []
    monkeypatch.setattr(tune, "get_context", lambda: SimpleNamespace(get_trial_id=lambda: "t1"))
    monkeypatch.setattr(tune, "report", reports.append)

    def fake_run_training(cfg, output_dir, log_file, run_name, tags, on_validation):
        calls.append((cfg, output_dir, log_file, run_name, tags))
        on_validation({"selection_score": 2.5}, "run-1")

    monkeypatch.setattr(run_module, "run_training", fake_run_training)
    base = {"model": {"name": "unet", "window": 4}}

    try:
        run_trial(
            {"model.window": 2},
            base_config=base,
            root=str(tmp_path),
            output_dir=str(tmp_path / "trials"),
            tags={"sweep": "s"},
        )
    finally:
        for handler in list(logging.getLogger("mhd_surrogate").handlers):
            if isinstance(handler, logging.FileHandler):
                logging.getLogger("mhd_surrogate").removeHandler(handler)
                handler.close()

    ((cfg, output_dir, log_file, run_name, tags),) = calls
    assert cfg.model.window == 2
    assert output_dir == tmp_path / "trials" / "t1"
    assert log_file == output_dir / TRIAL_LOG
    assert run_name == "unet model.window=2"
    assert tags == {"sweep": "s", "trial_id": "t1"}
    assert reports == [
        {"selection_score": 2.5, "mlflow_run_id": "run-1", "trial_dir": str(output_dir)}
    ]


# -- after the search -----------------------------------------------------------


@pytest.fixture
def uri(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("exp")
    return uri


def open_run(status=None):
    run = mlflow.start_run()
    mlflow.end_run(status) if status else mlflow.end_run()
    return run.info.run_id


def result(metrics, error=None, config=None, scores=None):
    history = None if scores is None else pd.DataFrame({"selection_score": scores})
    return SimpleNamespace(metrics=metrics, error=error, config=config, metrics_dataframe=history)


def test_recording_asha_records_the_trials_it_stops():
    from ray.tune.schedulers import TrialScheduler

    asha = recording_asha(
        metric="score", mode="max", time_attr="it", max_t=10, grace_period=1, reduction_factor=2
    )
    good, bad = SimpleNamespace(trial_id="good"), SimpleNamespace(trial_id="bad")
    for trial in (good, bad):
        asha.on_trial_add(None, trial)

    first = asha.on_trial_result(None, good, {"it": 1, "score": 1.0})
    second = asha.on_trial_result(None, bad, {"it": 1, "score": 0.0})

    assert (first, second) == (TrialScheduler.CONTINUE, TrialScheduler.STOP)
    assert asha.stopped == {"bad"}


def test_close_stopped_runs_kills_only_unfinished_runs_of_trials_without_error(uri, tmp_path):
    finished = open_run()
    pruned = open_run("RUNNING")  # ASHA ended it mid-training
    out_of_time = open_run("RUNNING")  # running when the time budget ran out
    errored = open_run("FAILED")
    (tmp_path / "t2").mkdir()
    (tmp_path / "t2" / TRIAL_LOG).write_text("epoch 2\n")

    def trial(trial_id, run_id, **kwargs):
        dir_ = str(tmp_path / trial_id)
        return result({"trial_id": trial_id, "mlflow_run_id": run_id, "trial_dir": dir_}, **kwargs)

    results = [
        trial("t1", finished),
        trial("t2", pruned),
        trial("t3", out_of_time),
        trial("t4", errored, error="boom"),
        result({}),  # failed before its first validation
    ]

    closed = close_stopped_runs(results, uri, pruned_trials={"t2"})

    client = MlflowClient(uri)
    assert closed == {"pruned": [pruned], "time_budget": [out_of_time]}
    run = client.get_run(pruned)
    assert run.info.status == "KILLED"
    assert run.data.tags["pruned"] == "true" and "time_budget" not in run.data.tags
    assert TRIAL_LOG in [a.path for a in client.list_artifacts(pruned)]
    run = client.get_run(out_of_time)
    assert run.info.status == "KILLED"
    assert run.data.tags["time_budget"] == "true" and "pruned" not in run.data.tags
    assert client.get_run(finished).info.status == "FINISHED"
    assert client.get_run(errored).info.status == "FAILED"


def test_summary_rows_are_best_first_with_unscored_trials_last():
    results = [
        result({"training_iteration": 1, "selection_score": 0.5}, config={"w": 1}, scores=[0.5]),
        result({}, error="boom", config={"w": 2}),
        result(
            {"training_iteration": 3, "selection_score": 1.0},
            config={"w": 4},
            scores=[0.2, 2.0, 1.0],
        ),
    ]

    rows = summary_rows(results, ["w"])

    assert [r["w"] for r in rows] == [4, 1, 2]
    assert rows[0] == {
        "w": 4,
        "validations": 3,
        "best_score": 2.0,
        "last_score": 1.0,
        "error": False,
    }
    assert rows[2]["best_score"] is None and rows[2]["error"] is True
