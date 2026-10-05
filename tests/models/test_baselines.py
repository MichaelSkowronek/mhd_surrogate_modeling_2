import numpy as np
import pytest

from mhd_surrogate.models.baselines import MeanField, Persistence


def make_datasets():
    rng = np.random.default_rng(0)
    return {
        "a": rng.normal(size=(7, 2, 5, 4)).astype(np.float32),
        "b": rng.normal(loc=3.0, size=(4, 2, 5, 4)).astype(np.float32),
    }


@pytest.mark.parametrize("chunk_t", [1, 3, 64])
def test_mean_field_is_the_pooled_per_pixel_mean_for_any_chunk_size(chunk_t):
    datasets = make_datasets()
    expected = np.concatenate(list(datasets.values())).astype(np.float64).mean(axis=0)

    model = MeanField(chunk_t=chunk_t)
    model.fit(datasets)

    assert model.mean.shape == (2, 5, 4)
    assert model.mean == pytest.approx(expected, abs=1e-5)
    assert model.n_steps == 11
    assert model.datasets == ["a", "b"]


def test_mean_field_weighs_every_step_equally_not_every_dataset():
    datasets = {"a": np.zeros((3, 1, 1, 1)), "b": np.full((1, 1, 1, 1), 4.0)}

    model = MeanField()
    model.fit(datasets)

    assert model.mean.item() == pytest.approx(1.0)  # (0 + 0 + 0 + 4) / 4, not (0 + 4) / 2


def test_mean_field_predicts_its_mean_at_every_step_and_ignores_the_context():
    model = MeanField()
    model.fit(make_datasets())

    prediction = model.predict(np.empty((0, 2, 5, 4)), n_steps=6)

    assert prediction.shape == (6, 2, 5, 4)
    assert (prediction == model.mean).all()


def test_mean_field_rejects_use_before_fit_and_empty_data():
    with pytest.raises(RuntimeError):
        MeanField().predict(np.empty((0, 2, 5, 4)), 3)
    with pytest.raises(RuntimeError):
        MeanField().save("unused")
    with pytest.raises(ValueError):
        MeanField().fit({})


def test_mean_field_checkpoint_round_trips(tmp_path):
    model = MeanField(chunk_t=3)
    model.fit(make_datasets())

    model.save(tmp_path / "ckpt")
    loaded = MeanField.load(tmp_path / "ckpt")

    assert (loaded.mean == model.mean).all()
    assert loaded.chunk_t == 3 and loaded.n_steps == 11 and loaded.datasets == ["a", "b"]


def test_persistence_repeats_the_last_context_frame(tmp_path):
    context = np.arange(2 * 2 * 3 * 2, dtype=float).reshape(2, 2, 3, 2)
    model = Persistence()
    model.fit({})

    prediction = model.predict(context, n_steps=4)

    assert prediction.shape == (4, 2, 3, 2)
    assert (prediction == context[-1]).all()

    model.save(tmp_path / "ckpt")
    assert isinstance(Persistence.load(tmp_path / "ckpt"), Persistence)
