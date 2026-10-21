import shutil

import numpy as np
import pytest

import mlflow
from mhd_surrogate.models.latent import LatentSurrogate, resolve_codec
from mhd_surrogate.models.latent_linear import LatentLinear
from mhd_surrogate.models.pod_codec import PODCodec
from mhd_surrogate.models.registry import load_model
from mhd_surrogate.training.mlflow_codec import log_codec

FRAME = (2, 5, 4)


def noisy_datasets(n_datasets=2, length=30, seed=1):
    """An AR(1) process per pixel: many distinct singular values."""
    rng = np.random.default_rng(seed)
    datasets = {}
    for i in range(n_datasets):
        x = np.empty((length, *FRAME))
        x[0] = rng.normal(size=FRAME)
        for step in range(1, length):
            x[step] = 0.9 * x[step - 1] + rng.normal(size=FRAME)
        datasets[f"d{i}"] = x.astype(np.float32)
    return datasets


@pytest.fixture
def codec_dir(tmp_path):
    codec = PODCodec(rank=6)
    codec.fit(noisy_datasets())
    codec.save(tmp_path / "codec")
    return tmp_path / "codec"


def fitted(codec_reference, window=2):
    model = LatentLinear(str(codec_reference), window=window)
    model.fit(noisy_datasets())
    return model


def test_a_logged_codec_is_resolved_by_its_model_uri(tmp_path, monkeypatch, codec_dir):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlruns.db'}")
    mlflow.set_experiment("exp")
    with mlflow.start_run():
        info = log_codec(PODCodec.load(codec_dir), codec_dir, FRAME)
    frames = noisy_datasets(seed=2)["d0"]

    model = fitted(f"models:/{info.model_id}")

    assert (
        resolve_codec(f"models:/{info.model_id}").encode(frames) == model.codec.encode(frames)
    ).all()
    assert model.codec_reference == f"models:/{info.model_id}"


def test_streamed_blocks_are_the_one_forecast(codec_dir):
    model = fitted(codec_dir, window=3)
    context = noisy_datasets(seed=2)["d0"][:5]

    blocks = list(model.predict_blocks(context, 11, 4))

    assert [len(b) for b in blocks] == [4, 4, 3]
    # Up to float rounding: the window carried over is the float32 latents.
    np.testing.assert_allclose(
        np.concatenate(blocks), model.predict(context, 11), rtol=1e-5, atol=1e-6
    )


def test_the_checkpoint_carries_the_codec(tmp_path, codec_dir):
    model = fitted(codec_dir)
    model.save(tmp_path / "model")
    shutil.rmtree(codec_dir)  # the reference no longer resolves

    loaded = load_model(tmp_path / "model")

    context = noisy_datasets(seed=2)["d1"][:4]
    assert isinstance(loaded, LatentLinear)
    assert (loaded.predict(context, 6) == model.predict(context, 6)).all()
    assert loaded.codec_reference == str(codec_dir) and loaded.window == 2
    assert loaded.fit_info == model.fit_info and loaded.datasets == ["d0", "d1"]
    assert loaded.parameter_count() == model.parameter_count()


def test_use_before_fit_and_bad_windows_are_rejected(codec_dir):
    with pytest.raises(RuntimeError, match="not fitted"):
        LatentLinear(str(codec_dir)).predict(np.zeros((1, *FRAME)), 3)
    with pytest.raises(RuntimeError, match="not fitted"):
        LatentLinear(str(codec_dir)).save("unused")
    with pytest.raises(ValueError, match="window must be >= 1"):
        LatentLinear(str(codec_dir), window=0)
    with pytest.raises(ValueError, match="need window=2"):
        fitted(codec_dir).predict(np.zeros((1, *FRAME)), 3)


def test_a_subclass_must_supply_the_latent_dynamics():
    model = LatentSurrogate("unused")
    for call in (
        lambda: model.fit_latents({}),
        lambda: model.forecast_latents(np.zeros((1, 2)), 1),
        model.latent_parameter_count,
        model.config,
        model.arrays,
        lambda: model.set_arrays({}),
    ):
        with pytest.raises(NotImplementedError):
            call()
