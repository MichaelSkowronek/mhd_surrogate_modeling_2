import numpy as np
import pytest

import mlflow
from mhd_surrogate.models.baselines import MeanField, Persistence
from mhd_surrogate.training.mlflow_model import log_surrogate, pinned_requirements

FRAME = (2, 5, 4)


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    # A sqlite store writes artifacts to ./mlruns; keep them in tmp_path.
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlruns.db'}")
    mlflow.set_experiment("exp")


def fitted_mean_field():
    model = MeanField()
    rng = np.random.default_rng(0)
    model.fit({"a": rng.normal(size=(6, *FRAME)).astype(np.float32)})
    return model


def test_logged_model_is_named_after_the_model_and_predicts_like_it(tracking, tmp_path):
    model = fitted_mean_field()
    model.save(tmp_path / "ckpt")

    with mlflow.start_run():
        info = log_surrogate(model, tmp_path / "ckpt", FRAME, params={"chunk_t": 64})

    assert info.name == "mean_field"
    logged = mlflow.get_logged_model(info.model_id)
    assert logged.name == "mean_field"
    assert logged.params == {"chunk_t": "64"}

    loaded = mlflow.pyfunc.load_model(info.model_uri)
    prediction = loaded.predict(np.empty((0, *FRAME), dtype=np.float32), params={"n_steps": 3})
    assert prediction.shape == (3, *FRAME)
    assert (prediction == model.mean).all()


def test_logged_model_uses_only_the_last_window_frames_of_the_input(tracking, tmp_path):
    model = Persistence()
    model.save(tmp_path / "ckpt")
    context = np.arange(3 * np.prod(FRAME), dtype=np.float32).reshape(3, *FRAME)

    with mlflow.start_run():
        info = log_surrogate(model, tmp_path / "ckpt", FRAME)

    prediction = mlflow.pyfunc.load_model(info.model_uri).predict(context, params={"n_steps": 2})
    assert (prediction == context[-1]).all()
    assert prediction.shape == (2, *FRAME)


def test_requirements_are_pinned_to_the_installed_versions():
    pins = pinned_requirements()
    assert [p.split("==")[0] for p in pins] == ["numpy", "jax", "mlflow"]
    assert all("==" in p and p.split("==")[1] for p in pins)


def test_a_model_adds_its_own_requirements_once():
    pins = pinned_requirements(("equinox", "jax"))

    assert [p.split("==")[0] for p in pins] == ["numpy", "jax", "mlflow", "equinox"]
