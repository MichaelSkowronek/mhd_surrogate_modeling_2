import numpy as np
import pytest

from mhd_surrogate.models.dmd import DMD
from mhd_surrogate.models.hankel_dmd import HankelDMD, delay_gram, delay_pair_indices
from mhd_surrogate.models.registry import build_model, load_model

FRAME = (2, 6, 4)
PERIOD = 8  # full periods in every dataset, so each has zero temporal mean


def standing_wave_datasets(n_datasets=3, length=24, seed=0):
    """A single fixed spatial pattern whose amplitude oscillates, A cos(wt + phi),
    plus a mean field. One frame shows only the current amplitude, not
    whether it's rising or falling, so a one-frame linear model can't
    forecast it; two frames determine it: a_{t+1} = 2 cos(w) a_t - a_{t-1}."""
    rng = np.random.default_rng(seed)
    pattern = rng.normal(size=FRAME)
    mean = rng.normal(size=FRAME)
    t = np.arange(length)
    datasets = {}
    for i in range(n_datasets):
        amplitude = rng.uniform(0.5, 2.0) * np.cos(
            2 * np.pi * t / PERIOD + rng.uniform(0, 2 * np.pi)
        )
        datasets[f"d{i}"] = (mean + amplitude[:, None, None, None] * pattern).astype(np.float32)
    return datasets


def noisy_datasets(n_datasets=2, length=30, seed=1):
    """Generic data (an AR(1) process per pixel): many distinct singular
    values, so truncation actually removes something."""
    rng = np.random.default_rng(seed)
    datasets = {}
    for i in range(n_datasets):
        x = np.empty((length, *FRAME))
        x[0] = rng.normal(size=FRAME)
        for step in range(1, length):
            x[step] = 0.9 * x[step - 1] + rng.normal(size=FRAME)
        datasets[f"d{i}"] = x.astype(np.float32)
    return datasets


def test_delay_pairs_need_their_history_in_the_same_dataset():
    before, after = delay_pair_indices([5, 4], delays=2)

    # Dataset 0 is rows 0-4, dataset 1 rows 5-8; a state needs one predecessor.
    assert before.tolist() == [1, 2, 3, 6, 7]
    assert after.tolist() == [2, 3, 4, 7, 8]
    assert delay_pair_indices([3, 2], delays=1)[0].tolist() == [0, 1, 3]


def test_delay_gram_matches_the_explicitly_stacked_delay_states():
    rng = np.random.default_rng(0)
    coefficients = rng.normal(size=(9, 3))
    rows, cols = delay_pair_indices([5, 4], delays=3)

    def stacked(newest):
        return np.stack([np.concatenate([coefficients[t - k] for k in range(3)]) for t in newest])

    gram = delay_gram(coefficients @ coefficients.T, rows, cols, delays=3)

    assert gram == pytest.approx(stacked(rows) @ stacked(cols).T)


def test_one_delay_is_plain_dmd():
    datasets = noisy_datasets()
    context = datasets["d1"][:3]
    plain = DMD(rank=5)
    plain.fit(datasets)
    hankel = HankelDMD(rank=5, delays=1, spatial_rank=12)
    hankel.fit(datasets)

    assert hankel.window == 1
    assert hankel.predict(context, 9) == pytest.approx(plain.predict(context, 9), abs=1e-4)
    assert sorted(np.angle(hankel.eigenvalues)) == pytest.approx(
        sorted(np.angle(plain.eigenvalues)), abs=1e-5
    )


def test_delays_forecast_an_oscillation_that_one_frame_cannot():
    datasets = standing_wave_datasets()
    held_out = datasets.pop("d2")
    hankel = HankelDMD(rank=2, delays=2, spatial_rank=4)
    hankel.fit(datasets)
    plain = DMD(rank=4)
    plain.fit(datasets)

    assert hankel.fit_info["spatial_rank"] == 1  # one pattern: one usable mode
    assert sorted(np.angle(hankel.eigenvalues)) == pytest.approx(
        [-2 * np.pi / PERIOD, 2 * np.pi / PERIOD], abs=1e-4
    )
    prediction = hankel.predict(held_out[:4], n_steps=20)
    assert prediction.shape == (20, *FRAME)
    assert prediction == pytest.approx(held_out[4:24], abs=1e-3)
    # A real one-frame operator can't rotate the amplitude's phase.
    assert np.abs(plain.predict(held_out[3:4], 20) - held_out[4:24]).max() > 0.1


def test_ranks_are_capped_at_the_usable_singular_values():
    model = HankelDMD(rank=10, delays=3, spatial_rank=10)
    model.fit(standing_wave_datasets())

    # One spatial mode, three delays of it: at most three delay modes.
    assert model.fit_info["spatial_rank"] == 1
    assert model.fit_info["rank"] <= 3
    assert model.hankel_basis.shape == (3, 1, model.fit_info["rank"])
    assert model.fit_info["n_pairs"] == 3 * (24 - 3)


def test_checkpoint_round_trips_through_the_registry(tmp_path):
    model = build_model({"name": "hankel_dmd", "rank": 4, "delays": 3, "spatial_rank": 6})
    model.fit(noisy_datasets())
    model.save(tmp_path)

    loaded = load_model(tmp_path)

    context = noisy_datasets(seed=2)["d0"][:5]
    assert isinstance(loaded, HankelDMD) and loaded.window == 3
    assert (loaded.predict(context, 5) == model.predict(context, 5)).all()
    assert loaded.fit_info == model.fit_info


def test_bad_arguments_and_use_before_fit_are_rejected():
    with pytest.raises(ValueError, match="delays"):
        HankelDMD(delays=0)
    with pytest.raises(RuntimeError):
        HankelDMD().predict(np.zeros((8, *FRAME)), 3)
    with pytest.raises(RuntimeError):
        HankelDMD().save("unused")
    with pytest.raises(ValueError, match="at least 4 frames"):
        HankelDMD(delays=3).fit({"a": np.zeros((3, *FRAME))})

    model = HankelDMD(rank=2, delays=2, spatial_rank=2)
    model.fit(standing_wave_datasets())
    with pytest.raises(ValueError, match="need delays=2"):
        model.predict(np.zeros((1, *FRAME)), 3)
