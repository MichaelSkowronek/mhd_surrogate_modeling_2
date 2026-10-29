import shutil

import numpy as np
import pytest

import mlflow
from mhd_surrogate.models.base import FitHooks, fit_model
from mhd_surrogate.models.latent import (
    LatentSurrogate,
    StochasticLatentSurrogate,
    resolve_codec,
)
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


class Recorder(LatentSurrogate):
    """An iterative latent model that records what its fit got."""

    name = "recorder"
    iterative = True

    def fit_latents(self, latents, hooks=None):
        self.received = (sorted(latents), hooks)


def test_an_iterative_latent_model_gets_the_fit_hooks(codec_dir):
    model = Recorder(str(codec_dir))
    hooks = FitHooks(neighbors=2)

    fit_model(model, noisy_datasets(), hooks)

    assert model.received == (["d0", "d1"], hooks)
    assert model.datasets == ["d0", "d1"]


class Drift(StochasticLatentSurrogate):
    """A stochastic latent model whose step k (of the forecast) adds
    seed + k to every entry of the newest latent: its randomness is a
    function of the seed and the step alone, as `sample_latents` requires."""

    name = "drift"

    def sample_latents(self, window, n_steps, seeds, first_step=0):
        steps = first_step + np.arange(n_steps, dtype=np.float32)
        return np.stack([window[-1] + np.cumsum(seed + steps)[:, None] for seed in seeds]).astype(
            np.float32
        )


def drift(codec_dir):
    model = Drift(str(codec_dir), window=2)
    model.codec = resolve_codec(str(codec_dir))
    return model


def test_a_stochastic_latent_models_members_and_blocks_are_its_samples(codec_dir):
    model = drift(codec_dir)
    context = noisy_datasets(seed=2)["d0"][:4]
    newest = model.codec.encode(context[-1:])[0]

    members = model.predict_members(context, 5, [0, 7])
    blocks = list(model.predict_blocks(context, 5, 2, seed=7))

    expected = newest + np.cumsum(7 + np.arange(5))[:, None]
    np.testing.assert_allclose(model.codec.encode(members[1]), expected, rtol=1e-4, atol=1e-3)
    np.testing.assert_array_equal(members[1], model.predict(context, 5, seed=7))
    assert members.shape == (2, 5, *FRAME)
    assert [len(b) for b in blocks] == [2, 2, 1]
    np.testing.assert_allclose(np.concatenate(blocks), members[1], rtol=1e-5, atol=1e-5)
    assert model.stochastic


def test_a_stochastic_latent_model_must_supply_its_sampling():
    with pytest.raises(NotImplementedError):
        StochasticLatentSurrogate("unused").sample_latents(np.zeros((1, 2)), 1, [0])
