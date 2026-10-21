"""run_codec_training end to end on a tiny synthetic zarr store: the code
path of `scripts/training/train_codec.py`."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr
from hydra import compose, initialize_config_dir

import mhd_surrogate.utils.hydra_resolvers  # noqa: F401
import mlflow
from mhd_surrogate.data.normalization import compute_normalization_stats
from mhd_surrogate.training import startup
from mhd_surrogate.training.codec_run import run_codec_training
from mhd_surrogate.training.export import checkpoint_digest
from mhd_surrogate.training.mlflow_codec import load_logged_codec
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
    monkeypatch.setattr(startup, "data_provenance", lambda: {"raw": "stub"})
    return tmp_path


def config(workspace, *overrides):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(
            "train_codec",
            overrides=[
                f"data.zarr_store={workspace / 'data.zarr'}",
                f"data.train_datasets={TRAIN}",
                "data.val_dataset=v",
                "data.test_dataset=never_opened",
                f"normalization.stats_path={workspace / 'stats.json'}",
                f"mlflow.tracking_uri=sqlite:///{workspace / 'mlruns.db'}",
                "evaluation.train_datasets=[a]",
                "jax.compilation_cache_dir=null",
                *overrides,
            ],
        )


def test_a_codec_is_fitted_scored_logged_and_loadable_by_its_model_id(workspace):
    """The waves' fluctuations span two modes, so one mode loses some."""
    scores = run_codec_training(
        config(workspace, "codec.rank=1"), workspace / "out", None, "pod codec.rank=1"
    )

    assert set(scores) == {"val", "train.a"}
    assert 0 < scores["val"]["reconstruction_rmse"] < 1
    client = MlflowClient(f"sqlite:///{workspace / 'mlruns.db'}")
    (run,) = client.search_runs([mlflow.get_experiment_by_name("mhd-codecs").experiment_id])
    assert run.info.run_name == "pod codec.rank=1"
    assert run.data.params["codec.rank"] == "1"
    assert run.data.params["data_version.raw"] == "stub"
    assert "model.name" not in run.data.params
    assert run.data.metrics["latent_size"] == 1
    assert run.data.metrics["val.reconstruction_rmse"] == pytest.approx(
        scores["val"]["reconstruction_rmse"]
    )
    checkpoint = workspace / "out" / "codec"
    assert run.data.tags["checkpoint_sha256"] == checkpoint_digest(checkpoint)

    (model,) = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id], output_format="list"
    )
    codec = load_logged_codec(f"models:/{model.model_id}")
    assert codec.latent_shape == (1,)


def test_scoring_a_dataset_that_isnt_a_training_dataset_as_train_is_refused(workspace):
    with pytest.raises(ValueError, match="not in data.train_datasets"):
        run_codec_training(
            config(workspace, "codec.rank=2", "evaluation.train_datasets=[v]"),
            workspace / "out",
            None,
            "pod",
        )
