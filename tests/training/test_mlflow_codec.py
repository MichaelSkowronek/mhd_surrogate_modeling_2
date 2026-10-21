import numpy as np
import pytest

import mlflow
from mhd_surrogate.models.pod_codec import PODCodec
from mhd_surrogate.training.mlflow_codec import load_logged_codec, log_codec

FRAME = (2, 5, 4)


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    # A sqlite store writes artifacts to ./mlruns; keep them in tmp_path.
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlruns.db'}")
    mlflow.set_experiment("exp")


def frames(n, seed):
    return np.random.default_rng(seed).normal(size=(n, *FRAME)).astype(np.float32)


def test_logged_codec_reconstructs_and_loads_back_as_the_codec(tracking, tmp_path):
    codec = PODCodec(rank=3)
    codec.fit({"a": frames(8, 0)})
    codec.save(tmp_path / "ckpt")
    unseen = frames(4, 1)

    with mlflow.start_run():
        info = log_codec(codec, tmp_path / "ckpt", FRAME, params={"chunk_t": 64})

    logged = mlflow.get_logged_model(info.model_id)
    assert logged.name == "pod"
    assert logged.params == {"chunk_t": "64", "n_parameters": "120", "latent_shape": "[3]"}
    reconstruction = mlflow.pyfunc.load_model(info.model_uri).predict(unseen)
    np.testing.assert_allclose(reconstruction, codec.decode(codec.encode(unseen)), rtol=1e-6)
    loaded = load_logged_codec(f"models:/{info.model_id}")
    assert (loaded.encode(unseen) == codec.encode(unseen)).all()
