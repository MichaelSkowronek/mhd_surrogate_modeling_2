import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mhd_surrogate.models.base import FitHooks
from mhd_surrogate.models.neural import (
    AutoregressiveSurrogate,
    Frame,
    add_input_noise,
    coordinate_channels,
    crop,
    pad_to_multiple,
    padded_size,
    rollout_loss,
    step,
)
from mhd_surrogate.models.registry import build_model, load_model
from mhd_surrogate.models.unet import UNetSurrogate

TRAINING = dict(max_epochs=2, batch_size=2, log_every=1, prefetch=0, warmup_steps=0)


def wave_datasets(lengths=(14, 11), shape=(10, 6)):
    """A wave travelling along x in both channels, with no-slip walls (zero
    at the first and last y) and a constant inlet u_y (first x) -- the
    constant pixels the model must keep fixed."""
    nx, ny = shape
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    out = {}
    for i, n in enumerate(lengths):
        t = np.arange(n)[:, None, None]
        u_x = 2.0 + np.sin(0.6 * (x - t) + i) * np.sin(np.pi * y / (ny - 1))
        u_y = 0.5 * np.cos(0.6 * (x - t) + i) * np.sin(np.pi * y / (ny - 1))
        frames = np.stack([u_x, u_y], axis=1).astype(np.float32)
        frames[:, :, :, 0] = frames[:, :, :, -1] = 0.0
        frames[:, 1, 0, 1:-1] = 0.3
        out[f"d{i}"] = frames
    return out


def small_model(**kwargs):
    defaults = dict(window=2, base_channels=4, depth=1, compute_dtype="float32", training=TRAINING)
    return UNetSurrogate(**{**defaults, **kwargs})


# -- padding and coordinates ----------------------------------------------------


@pytest.mark.parametrize("n, multiple, expected", [(1151, 16, 1152), (127, 16, 128), (16, 16, 16)])
def test_padded_size(n, multiple, expected):
    assert padded_size(n, multiple) == expected


def test_pad_repeats_the_edge_and_crop_undoes_it():
    x = jnp.arange(2 * 5 * 3, dtype=jnp.float32).reshape(1, 2, 5, 3)

    padded = pad_to_multiple(x, 4)

    assert padded.shape == (1, 2, 8, 4)
    np.testing.assert_array_equal(
        padded[..., 5:, :3], jnp.broadcast_to(x[..., 4:5, :], (1, 2, 3, 3))
    )
    np.testing.assert_array_equal(padded[..., :5, 3], x[..., :, 2])
    np.testing.assert_array_equal(crop(padded, (5, 3)), x)


def test_coordinate_channels_span_minus_one_to_one_along_their_axis():
    coords = coordinate_channels((5, 3))

    assert coords.shape == (2, 5, 3)
    np.testing.assert_allclose(coords[0, :, 1], [-1.0, -0.5, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(coords[1, 2, :], [-1.0, 0.0, 1.0])
    # Each channel varies along its own axis only.
    np.testing.assert_array_equal(coords[0], jnp.broadcast_to(coords[0, :, :1], (5, 3)))
    np.testing.assert_array_equal(coords[1], jnp.broadcast_to(coords[1, :1, :], (5, 3)))


# -- normalization and constant pixels --------------------------------------------


def test_normalization_is_the_per_channel_mean_and_std_over_all_frames():
    datasets = wave_datasets()
    model = small_model()
    model._fit_normalization(datasets)

    pooled = np.concatenate(list(datasets.values())).astype(np.float64)
    np.testing.assert_allclose(model.mean, pooled.mean(axis=(0, 2, 3)), rtol=1e-6)
    np.testing.assert_allclose(model.std, pooled.std(axis=(0, 2, 3)), rtol=1e-5)
    assert model.frame_shape == (2, 10, 6)


def test_walls_and_the_constant_inlet_are_found_as_constant_pixels():
    model = small_model()
    model._fit_normalization(wave_datasets())

    expected = np.zeros((2, 10, 6), dtype=bool)
    expected[:, :, 0] = expected[:, :, -1] = True
    expected[1, 0, :] = True
    np.testing.assert_array_equal(model.constant, expected)
    assert model.constant_values[1, 0, 2] == pytest.approx(0.3)
    assert np.all(model.constant_values[:, :, 0] == 0.0)


class AddOne(eqx.Module):
    """A stand-in network whose update is +1 everywhere."""

    def __call__(self, x):
        return jnp.ones((2, *x.shape[1:]), dtype=x.dtype)


def frame(shape=(2, 4, 4), constant=None, values=None):
    constant = np.zeros(shape, dtype=bool) if constant is None else constant
    return Frame(
        coords=jnp.zeros((2, *shape[1:])),
        constant=jnp.asarray(constant),
        constant_values=jnp.zeros(shape) if values is None else jnp.asarray(values),
        loss_weight=jnp.asarray(~constant, dtype=jnp.float32),
    )


def test_step_adds_the_update_to_the_newest_frame_and_keeps_constant_pixels():
    constant = np.zeros((2, 4, 4), dtype=bool)
    constant[:, :, 0] = True
    window = jnp.stack([jnp.zeros((2, 4, 4)), jnp.full((2, 4, 4), 5.0)])

    out = step(
        AddOne(), frame(constant=constant, values=np.full((2, 4, 4), 7.0)), window, jnp.float32
    )

    assert np.all(out[:, :, 1:] == 6.0)
    assert np.all(out[:, :, 0] == 7.0)


def test_rollout_loss_feeds_back_its_own_predictions():
    """With update +1 from a zero frame, the rollout predicts 1 then 2."""
    sequence = jnp.stack([jnp.zeros((2, 4, 4)), jnp.zeros((2, 4, 4)), jnp.zeros((2, 4, 4))])

    loss = rollout_loss(AddOne(), frame(), sequence[None], window=1, dtype=jnp.float32)

    # Own predictions 1 and 2 against zeros: (1 + 4) / 2. Teacher forcing
    # would predict 1 from the true 0 at both steps: (1 + 1) / 2.
    assert float(loss) == pytest.approx(2.5)


def test_rollout_loss_ignores_constant_and_padding_pixels():
    constant = np.zeros((2, 4, 4), dtype=bool)
    constant[:, 0, :] = True
    sequence = jnp.zeros((1, 2, 2, 4, 4))

    loss = rollout_loss(AddOne(), frame(constant=constant), sequence, window=1, dtype=jnp.float32)

    # Constant pixels are reset to their value (0, matching the target);
    # the rest are off by one, and the mean is over those only.
    assert float(loss) == pytest.approx(1.0)


def test_input_noise_perturbs_only_the_counted_pixels_of_the_input_frames():
    constant = np.zeros((2, 8, 8), dtype=bool)
    constant[:, 0, :] = True
    batch = jnp.zeros((64, 3, 2, 8, 8))  # window 2, 1 target

    noisy = add_input_noise(batch, 2, 0.5, frame((2, 8, 8), constant), jax.random.key(0))

    inputs = np.asarray(noisy[:, :2])
    assert np.all(noisy[:, 2] == 0.0)  # the target stays clean
    assert np.all(inputs[:, :, :, 0, :] == 0.0)  # constant pixels too
    assert np.std(inputs[:, :, :, 1:, :]) == pytest.approx(0.5, rel=0.03)


def test_rollout_loss_with_input_noise_is_the_noise_variance_for_an_exact_model():
    """A model that adds nothing predicts its (noisy) input: off by the
    noise, so the loss is ~noise_std**2; without noise it is exact."""

    class Zero(eqx.Module):
        def __call__(self, x):
            return jnp.zeros((2, *x.shape[1:]))

    batch = jnp.zeros((256, 2, 2, 8, 8))
    key = jax.random.key(1)

    clean = rollout_loss(Zero(), frame((2, 8, 8)), batch, 1, jnp.float32, 0.0, key)
    noisy = rollout_loss(Zero(), frame((2, 8, 8)), batch, 1, jnp.float32, 0.2, key)

    assert float(clean) == 0.0
    assert float(noisy) == pytest.approx(0.04, rel=0.05)


def test_input_noise_trains_and_is_kept_in_the_checkpoint(tmp_path):
    model = small_model(input_noise_std=0.1)

    model.fit(wave_datasets())
    model.save(tmp_path)

    assert np.all(np.isfinite(model.predict(wave_datasets()["d0"][:2], 4)))
    assert load_model(tmp_path).input_noise_std == 0.1


# -- the model ------------------------------------------------------------------


def test_untrained_model_forecasts_persistence():
    model = small_model(training={**TRAINING, "max_epochs": 1, "learning_rate": 0.0})
    datasets = wave_datasets()
    model.fit(datasets)
    context = datasets["d0"][:2]

    forecast = model.predict(context, 3)

    assert forecast.shape == (3, 2, 10, 6)
    np.testing.assert_allclose(forecast, np.broadcast_to(context[-1], (3, 2, 10, 6)), atol=1e-5)


def test_fit_trains_validates_logs_and_keeps_the_boundary_exact(tmp_path):
    model = small_model(training={**TRAINING, "max_epochs": 3})
    datasets = wave_datasets()
    validated = []
    logged = []
    hooks = FitHooks(
        validate=lambda m: (
            validated.append(m.predict(datasets["d1"][:2], 4))
            or {"selection_score": -float(len(validated))}
        ),
        log_metrics=lambda metrics, step: logged.append(metrics),
        state_dir=tmp_path,
    )

    model.fit(datasets, hooks)

    assert len(validated) == 3
    assert model.fit_info["best_epoch"] == 1
    assert model.fit_info["parameters"] > 0
    assert any("loss" in m for m in logged)
    assert (tmp_path / "state.json").exists()
    forecast = model.predict(datasets["d1"][:2], 5)
    assert np.all(forecast[:, :, :, 0] == 0.0) and np.all(forecast[:, :, :, -1] == 0.0)
    np.testing.assert_allclose(forecast[:, 1, 0, 1:-1], 0.3, rtol=1e-6)


def test_training_lowers_the_one_step_error():
    datasets = wave_datasets()
    context, target = datasets["d0"][3:5], datasets["d0"][5]
    untrained = small_model(training={**TRAINING, "max_epochs": 1, "learning_rate": 0.0})
    untrained.fit(datasets)
    trained = small_model(training={**TRAINING, "max_epochs": 15, "learning_rate": 3e-3})
    trained.fit(datasets)

    def error(model):
        return float(np.sqrt(np.mean((model.predict(context, 1)[0] - target) ** 2)))

    assert error(trained) < 0.7 * error(untrained)


@pytest.mark.parametrize("compute_dtype, rollout_steps", [("bfloat16", 1), ("float32", 3)])
def test_bf16_compute_and_multi_step_rollouts_train(compute_dtype, rollout_steps):
    model = small_model(compute_dtype=compute_dtype, rollout_steps=rollout_steps)

    model.fit(wave_datasets())

    assert np.all(np.isfinite(model.predict(wave_datasets()["d0"][:2], 4)))
    assert model.fit_info["steps"] > 0


def test_save_and_load_reproduce_the_forecast(tmp_path):
    datasets = wave_datasets()
    model = small_model()
    model.fit(datasets)
    model.save(tmp_path)

    loaded = load_model(tmp_path)

    assert isinstance(loaded, UNetSurrogate)
    assert loaded.config() == model.config()
    assert loaded.fit_info == model.fit_info
    assert loaded.datasets == ["d0", "d1"]
    context = datasets["d1"][:2]
    np.testing.assert_array_equal(loaded.predict(context, 4), model.predict(context, 4))


def test_registry_builds_it_from_a_config():
    model = build_model({"name": "unet", "window": 3, "depth": 2, "training": {"batch_size": 4}})

    assert isinstance(model, UNetSurrogate)
    assert model.window == 3
    assert model.pad_multiple() == 4
    assert model.trainer_config.batch_size == 4
    assert model.iterative


def test_seed_is_the_trainer_seed_too():
    assert small_model(seed=5).trainer_config.seed == 5


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"window": 0}, "window"),
        ({"rollout_steps": 0}, "rollout_steps"),
        ({"input_noise_std": -0.1}, "input_noise_std"),
        ({"compute_dtype": "int8"}, "dtype"),
        ({"host_dtype": "float64"}, "dtype"),
    ],
)
def test_rejects_invalid_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        small_model(**kwargs)


def test_rejects_unknown_training_settings():
    with pytest.raises(TypeError):
        small_model(training={"epochs": 3})


def test_predict_and_save_need_a_fitted_model(tmp_path):
    with pytest.raises(RuntimeError, match="not fitted"):
        small_model().predict(np.zeros((2, 2, 10, 6)), 1)
    with pytest.raises(RuntimeError, match="not fitted"):
        small_model().save(tmp_path)


def test_the_base_class_needs_a_network():
    with pytest.raises(NotImplementedError):
        AutoregressiveSurrogate().build_network(1, 1, jax.random.key(0))
    assert AutoregressiveSurrogate().pad_multiple() == 1
