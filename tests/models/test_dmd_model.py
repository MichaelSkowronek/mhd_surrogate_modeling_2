import numpy as np
import pytest

from mhd_surrogate.models.dmd import DMD, pair_indices, stabilize_eigenvalues

FRAME = (2, 6, 4)
PERIODS = (8, 4)  # full periods in every dataset, so each has zero temporal mean


def linear_datasets(n_datasets=2, length=16, seed=0):
    """Trajectories of a known linear system: two rotations (periods 8 and 4
    steps), embedded in the frame by a fixed random map, with a mean field
    and different channel amplitudes added. Every dataset covers whole
    periods, so the pooled temporal mean is exactly the added mean field and
    the fluctuations follow the linear dynamics exactly."""
    rng = np.random.default_rng(seed)
    embed = rng.normal(size=(int(np.prod(FRAME)), 2 * len(PERIODS)))
    mean = rng.normal(size=FRAME)
    channel_scale = np.array([3.0, 0.5])[:, None, None]
    t = np.arange(length)
    datasets = {}
    for i in range(n_datasets):
        phases = rng.uniform(0, 2 * np.pi, size=len(PERIODS))
        amps = rng.uniform(0.5, 2.0, size=len(PERIODS))
        latent = np.concatenate(
            [
                np.stack([a * np.cos(2 * np.pi * t / p + ph), a * np.sin(2 * np.pi * t / p + ph)])
                for p, ph, a in zip(PERIODS, phases, amps, strict=True)
            ]
        )
        fluct = (embed @ latent).T.reshape(length, *FRAME)
        datasets[f"d{i}"] = (mean + channel_scale * fluct).astype(np.float32)
    return datasets


def test_pair_indices_never_pair_across_a_dataset_boundary():
    before, after = pair_indices([3, 2])

    assert before.tolist() == [0, 1, 3]
    assert after.tolist() == [1, 2, 4]


def test_stabilize_moves_only_growing_eigenvalues_onto_the_unit_circle():
    eigenvalues = np.array([1.1 * np.exp(0.3j), 0.9 * np.exp(-0.2j), 1.0 + 0j])

    stable = stabilize_eigenvalues(eigenvalues)

    assert np.abs(stable) == pytest.approx([1.0, 0.9, 1.0])
    assert np.angle(stable) == pytest.approx(np.angle(eigenvalues))


def test_dmd_recovers_the_frequencies_of_a_known_linear_system():
    model = DMD(rank=4)
    model.fit(linear_datasets())

    expected = np.exp(2j * np.pi * np.array([1 / 8, -1 / 8, 1 / 4, -1 / 4]))
    assert sorted(np.angle(model.eigenvalues)) == pytest.approx(
        sorted(np.angle(expected)), abs=1e-4
    )
    assert np.abs(model.eigenvalues) == pytest.approx(np.ones(4), abs=1e-4)
    assert model.fit_info["explained_variance"] == pytest.approx(1.0)


def test_dmd_forecasts_a_known_linear_system_exactly_in_raw_units():
    datasets = linear_datasets(n_datasets=3)
    held_out = datasets.pop("d2")
    model = DMD(rank=4)
    model.fit(datasets)

    prediction = model.predict(held_out[:5], n_steps=11)

    assert prediction.shape == (11, *FRAME)
    assert prediction == pytest.approx(held_out[5:16], abs=1e-3)


def test_rank_is_capped_at_the_number_of_usable_singular_values():
    model = DMD(rank=10)
    model.fit(linear_datasets())

    assert model.fit_info["rank"] == 4
    assert model.basis.shape == (int(np.prod(FRAME)), 4)


def test_result_does_not_depend_on_the_device_block_size():
    datasets = linear_datasets()
    small, large = DMD(rank=4, column_block=5, chunk_t=3), DMD(rank=4)
    small.fit(datasets)
    large.fit(datasets)

    context = datasets["d0"][:1]
    assert small.predict(context, 7) == pytest.approx(large.predict(context, 7), abs=1e-4)


def test_checkpoint_round_trips(tmp_path):
    model = DMD(rank=4, stabilize=False)
    model.fit(linear_datasets())
    model.save(tmp_path)

    loaded = DMD.load(tmp_path)

    context = linear_datasets()["d1"][:2]
    assert (loaded.predict(context, 5) == model.predict(context, 5)).all()
    assert loaded.stabilize is False and loaded.fit_info == model.fit_info


def test_use_before_fit_and_too_short_datasets_are_rejected():
    with pytest.raises(RuntimeError):
        DMD().predict(np.zeros((1, *FRAME)), 3)
    with pytest.raises(RuntimeError):
        DMD().save("unused")
    with pytest.raises(ValueError):
        DMD().fit({"a": np.zeros((1, *FRAME))})
