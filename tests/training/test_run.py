"""run_training end to end on a tiny synthetic zarr store: the same code path
as `scripts/training/train.py` and every Ray Tune trial."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr
from hydra import compose, initialize_config_dir

import mhd_surrogate.utils.hydra_resolvers  # noqa: F401
import mlflow
from mhd_surrogate.data.normalization import compute_normalization_stats
from mhd_surrogate.training import run as run_module
from mhd_surrogate.training.run import run_training
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[2]
SHAPE = (2, 8, 6)
TRAIN = ["a", "b"]


def wave(n, phase):
    t = np.arange(n)[:, None, None]
    x = np.arange(SHAPE[1])[None, :, None]
    y = np.sin(np.pi * np.arange(SHAPE[2]) / (SHAPE[2] - 1))[None, None, :]
    u = np.sin(0.5 * (x - t) + phase) * y
    return np.stack([1.0 + u, 0.5 * np.cos(0.5 * (x - t) + phase) * y], axis=1).astype(np.float32)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A repo-like working directory: zarr store, stats, grid config and a
    tracking store, with the data version stubbed (no dvc.lock pins here)."""
    monkeypatch.chdir(tmp_path)
    store = zarr.open_group(store=str(tmp_path / "data.zarr"), mode="w")
    store.attrs["channel_names"] = ["u_x", "u_y"]
    series = {"a": wave(210, 0.0), "b": wave(25, 1.0), "v": wave(210, 2.0)}
    for name, frames in series.items():
        store.create_array(name, data=frames)
    stats = compute_normalization_stats({k: series[k] for k in TRAIN}, ["u_x", "u_y"])
    stats.save(tmp_path / "stats.json")
    (tmp_path / "configs" / "analysis").mkdir(parents=True)
    shutil.copy(ROOT / "configs" / "analysis" / "grid.yaml", tmp_path / "configs" / "analysis")
    (tmp_path / "dvc.lock").write_text("schema: '2.0'\n")
    monkeypatch.setattr(run_module, "data_provenance", lambda: {"raw": "stub"})
    return tmp_path


def config(workspace, *overrides):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(
            "config",
            overrides=[
                f"data.zarr_store={workspace / 'data.zarr'}",
                f"data.train_datasets={TRAIN}",
                "data.val_dataset=v",
                "data.test_dataset=never_opened",
                "data.context_steps=4",
                f"normalization.stats_path={workspace / 'stats.json'}",
                f"mlflow.tracking_uri=sqlite:///{workspace / 'mlruns.db'}",
                "mlflow.experiment_name=test",
                "evaluation.train_datasets=[a]",
                "evaluation.report_leads=[1,2]",
                "evaluation.tie_break_lead=2",
                "jax.compilation_cache_dir=null",
                *overrides,
            ],
        )


UNET = [
    "model=unet",
    "model.window=2",
    "model.base_channels=4",
    "model.depth=1",
    "model.compute_dtype=float32",
    "model.training.max_epochs=2",
    "model.training.batch_size=4",
    "model.training.samples_per_epoch=16",
    "model.training.prefetch=0",
]


def only_run(workspace):
    client = MlflowClient(f"sqlite:///{workspace / 'mlruns.db'}")
    (run,) = client.search_runs([mlflow.get_experiment_by_name("test").experiment_id])
    return run, client


def test_a_closed_form_model_is_fitted_scored_and_tracked(workspace):
    scores = run_training(
        config(workspace, "model=persistence"), workspace / "out", None, "persistence"
    )

    assert set(scores) == {"val", "train.a"}
    assert scores["val"]["rmse_lead_1"] > 0
    run, client = only_run(workspace)
    assert run.info.run_name == "persistence"
    assert run.data.params["data_version.raw"] == "stub"
    assert "val.skill_horizon" in run.data.metrics
    assert not (workspace / "out" / "training_state").exists()
    assert (workspace / "out" / "model" / "model.json").exists()


def test_an_iterative_model_reports_every_validation_and_records_its_run(workspace):
    reported = []

    scores = run_training(
        config(workspace, *UNET),
        workspace / "out",
        None,
        "unet",
        tags={"sweep": "s1"},
        on_validation=lambda scores, run_id: reported.append((scores, run_id)),
    )

    run, client = only_run(workspace)
    assert len(reported) == 2
    assert {run_id for _, run_id in reported} == {run.info.run_id}
    assert "selection_score" in reported[0][0]
    assert run.data.tags["sweep"] == "s1"
    assert "val_monitor.selection_score" in run.data.metrics
    record = json.loads((workspace / "out" / "training_state" / "mlflow_run.json").read_text())
    assert record["run_id"] == run.info.run_id
    assert "resume" not in record["config"]
    assert np.isfinite(scores["val"]["rmse_lead_1"])


def test_resume_continues_the_recorded_run(workspace):
    run_training(config(workspace, *UNET), workspace / "out", None, "unet")

    run_training(
        config(workspace, *UNET, f"resume={workspace / 'out'}"), workspace / "out", None, "x"
    )

    run, _ = only_run(workspace)
    assert run.data.tags["resumed"] == "true"
    assert run.info.run_name == "unet"


def test_resume_is_refused_for_a_closed_form_model(workspace):
    from mhd_surrogate.training.tracking import save_run_record

    cfg = config(workspace, "model=persistence", f"resume={workspace / 'out'}")
    resolved = mhd_surrogate_config(cfg)
    save_run_record(workspace / "out" / "training_state", _start_run(workspace), resolved)

    with pytest.raises(ValueError, match="isn't trained iteratively"):
        run_training(cfg, workspace / "out", None, "persistence")


def test_export_dir_gets_the_checkpoint_and_metrics(workspace):
    run_training(
        config(workspace, "model=persistence", f"export.dir={workspace / 'export'}"),
        workspace / "out",
        None,
        "persistence",
    )

    metrics = json.loads((workspace / "export" / "metrics.json").read_text())
    assert set(metrics) == {"val", "train.a"}
    assert (workspace / "export" / "model" / "model.json").exists()


def mhd_surrogate_config(cfg):
    from omegaconf import OmegaConf

    resolved = OmegaConf.to_container(cfg, resolve=True)
    resolved.pop("resume")
    return resolved


def _start_run(workspace):
    mlflow.set_tracking_uri(f"sqlite:///{workspace / 'mlruns.db'}")
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        pass
    return run.info.run_id
