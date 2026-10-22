"""Fixtures for the serving tests: an experiment store and a registry
(two sqlite stores), and small fitted models logged in the first as a
training run logs them."""

from pathlib import Path

import numpy as np
import pytest

import mlflow
from mhd_surrogate.models.baselines import MeanField
from mhd_surrogate.serving.promotion import promote
from mhd_surrogate.training.mlflow_model import log_surrogate
from mhd_surrogate.training.mlflow_utils import log_checkpoint_once

FRAME = (2, 5, 4)
SOURCE_EXPERIMENT = "exp"
EXPERIMENT = "promotions"
NAME = "surrogate"


@pytest.fixture
def stores(tmp_path, monkeypatch):
    """(source, target) tracking URIs: two sqlite stores, their artifacts
    (./mlruns) kept in tmp_path."""
    monkeypatch.chdir(tmp_path)
    return f"sqlite:///{tmp_path / 'source.db'}", f"sqlite:///{tmp_path / 'target.db'}"


@pytest.fixture
def logged_checkpoint(stores, tmp_path):
    """`logged_checkpoint(seed)`: a mean field fitted on random frames, its
    checkpoint logged once in the source store's experiment; returns the
    model, the checkpoint and the logged model's id."""

    def log(seed: int = 0) -> tuple[MeanField, Path, str]:
        model = MeanField()
        frames = np.random.default_rng(seed).normal(size=(6, *FRAME)).astype(np.float32)
        model.fit({"a": frames})
        checkpoint = tmp_path / f"checkpoint-{seed}"
        model.save(checkpoint)
        mlflow.set_tracking_uri(stores[0])
        mlflow.set_experiment(SOURCE_EXPERIMENT)
        with mlflow.start_run():
            model_id = log_checkpoint_once(
                checkpoint, lambda: log_surrogate(model, checkpoint, FRAME)
            )
        return model, checkpoint, model_id

    return log


@pytest.fixture
def promoted(stores, logged_checkpoint):
    """`promoted(seed, **kwargs)`: a logged checkpoint promoted to the
    target store's registered model `NAME` (alias `champion`); returns the
    model and the `Promotion`."""

    def run(seed: int = 0, **kwargs):
        model, checkpoint, _ = logged_checkpoint(seed)
        source, target = stores
        promotion = promote(
            checkpoint, source, SOURCE_EXPERIMENT, target, NAME, "champion", EXPERIMENT, **kwargs
        )
        return model, promotion

    return run
