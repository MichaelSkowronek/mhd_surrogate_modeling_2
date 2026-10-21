import numpy as np
import pytest

from mhd_surrogate.models.dmd import DMD
from mhd_surrogate.models.latent_linear import LatentLinear, transition_matrix, window_pairs
from mhd_surrogate.models.pod_codec import PODCodec

FRAME = (2, 6, 4)
PERIODS = (8, 4)  # full periods in every dataset, so each has zero temporal mean


def rotation_datasets(n_datasets=3, length=16, seed=0):
    """Trajectories of a known linear system: two rotations (periods 8 and 4
    steps) embedded in the frame by a fixed random map, plus a mean field
    and different channel amplitudes."""
    rng = np.random.default_rng(seed)
    embed = rng.normal(size=(int(np.prod(FRAME)), 2 * len(PERIODS)))
    mean = rng.normal(size=FRAME)
    channel_scale = np.array([3.0, 0.5])[:, None, None]
    t = np.arange(length)
    datasets = {}
    for i in range(n_datasets):
        latent = np.concatenate(
            [
                a * np.stack([np.cos(2 * np.pi * t / p + ph), np.sin(2 * np.pi * t / p + ph)])
                for p, ph, a in zip(
                    PERIODS, rng.uniform(0, 2 * np.pi, 2), rng.uniform(0.5, 2, 2), strict=True
                )
            ]
        )
        fluct = (embed @ latent).T.reshape(length, *FRAME)
        datasets[f"d{i}"] = (mean + channel_scale * fluct).astype(np.float32)
    return datasets


def standing_wave_datasets(n_datasets=3, length=24, seed=0):
    """One spatial pattern whose amplitude oscillates, plus a mean field: a
    single frame doesn't say whether the amplitude rises or falls, two do."""
    rng = np.random.default_rng(seed)
    pattern, mean = rng.normal(size=FRAME), rng.normal(size=FRAME)
    t = np.arange(length)
    datasets = {}
    for i in range(n_datasets):
        amplitude = rng.uniform(0.5, 2) * np.cos(2 * np.pi * t / 8 + rng.uniform(0, 2 * np.pi))
        datasets[f"d{i}"] = (mean + amplitude[:, None, None, None] * pattern).astype(np.float32)
    return datasets


def pod_codec(datasets, rank, directory):
    codec = PODCodec(rank=rank)
    codec.fit(datasets)
    codec.save(directory)
    return str(directory)


def test_window_pairs_stack_the_window_newest_first():
    latents = np.arange(10.0).reshape(5, 2)

    features, targets = window_pairs(latents, window=2)

    assert features.tolist() == [[2, 3, 0, 1], [4, 5, 2, 3], [6, 7, 4, 5]]
    assert targets.tolist() == [[4, 5], [6, 7], [8, 9]]


def test_transition_matrix_advances_the_window_state():
    rng = np.random.default_rng(0)
    operator = rng.normal(size=(6, 2))  # window 3, r 2
    state = rng.normal(size=6)

    advanced = state @ transition_matrix(operator, window=3)

    assert advanced[:2] == pytest.approx(state @ operator)
    assert advanced[2:] == pytest.approx(state[:4])


def test_on_a_pod_codec_with_window_1_it_forecasts_a_linear_system_like_dmd(tmp_path):
    datasets = rotation_datasets()
    held_out = datasets.pop("d2")
    model = LatentLinear(pod_codec(datasets, 4, tmp_path / "codec"), window=1)
    model.fit(datasets)
    dmd = DMD(rank=4, stabilize=False)
    dmd.fit(datasets)

    prediction = model.predict(held_out[:5], n_steps=11)

    assert prediction.shape == (11, *FRAME)
    np.testing.assert_allclose(prediction, held_out[5:], atol=1e-3)
    np.testing.assert_allclose(prediction, dmd.predict(held_out[:5], 11), atol=1e-3)
    assert model.fit_info["spectral_radius"] == pytest.approx(1.0, abs=1e-4)
    assert model.fit_info["n_pairs"] == 2 * 15


def test_a_standing_wave_needs_a_window_of_two(tmp_path):
    datasets = standing_wave_datasets()
    held_out = datasets.pop("d2")
    codec = pod_codec(datasets, 1, tmp_path / "codec")
    one, two = LatentLinear(codec, window=1), LatentLinear(codec, window=2)
    one.fit(datasets)
    two.fit(datasets)

    truth = held_out[2:12]
    np.testing.assert_allclose(two.predict(held_out[:2], 10), truth, atol=1e-3)
    assert np.abs(one.predict(held_out[:2], 10) - truth).max() > 0.1


def test_ridge_shrinks_the_operator(tmp_path):
    datasets = rotation_datasets()
    codec = pod_codec(datasets, 4, tmp_path / "codec")
    plain, ridge = LatentLinear(codec), LatentLinear(codec, ridge=0.5)
    plain.fit(datasets)
    ridge.fit(datasets)

    assert np.linalg.norm(ridge.operator) < np.linalg.norm(plain.operator)
    assert ridge.fit_info["spectral_radius"] < 1


def test_parameters_are_the_operator_and_the_codec(tmp_path):
    datasets = rotation_datasets()
    model = LatentLinear(pod_codec(datasets, 4, tmp_path / "codec"), window=3)
    model.fit(datasets)

    # A (3 * 4 x 4) operator and the codec's 48 x 4 basis; the latent mean
    # is a centering, like the codec's mean field.
    assert model.parameter_count() == 12 * 4 + 48 * 4


def test_datasets_no_longer_than_the_window_are_rejected(tmp_path):
    datasets = rotation_datasets()
    model = LatentLinear(pod_codec(datasets, 4, tmp_path / "codec"), window=3)

    with pytest.raises(ValueError, match="more than window=3"):
        model.fit({"short": datasets["d0"][:3]})
