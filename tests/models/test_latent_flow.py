import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mhd_surrogate.evaluation.protocol import forecast_members, is_stochastic
from mhd_surrogate.evaluation.rollout import rollout_blocks
from mhd_surrogate.models.base import FitHooks
from mhd_surrogate.models.flow import step_noise
from mhd_surrogate.models.latent_flow import (
    TIME_FREQUENCIES,
    TIME_WIDTH,
    LatentFlow,
    LatentScales,
    VelocityMLP,
    fit_scales,
    latent_flow_loss,
)
from mhd_surrogate.models.pod_codec import PODCodec
from mhd_surrogate.models.registry import build_model, load_model

FRAME = (2, 5, 4)
RANK = 6
TRAINING = dict(max_epochs=2, batch_size=4, log_every=1, prefetch=0, warmup_steps=0)


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


@pytest.fixture(scope="module")
def codec_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("codec")
    codec = PODCodec(rank=RANK)
    codec.fit(noisy_datasets())
    codec.save(directory)
    return directory


def small_model(codec_dir, **kwargs):
    defaults = dict(window=2, width=16, depth=1, sampling_steps=2, training=TRAINING)
    return LatentFlow(str(codec_dir), **{**defaults, **kwargs})


@pytest.fixture(scope="module")
def fitted(codec_dir):
    model = small_model(codec_dir)
    model.fit(noisy_datasets())
    return model


def context():
    return noisy_datasets(seed=2)["d0"][:3]


# -- the network --------------------------------------------------------------------


def test_the_untrained_network_maps_a_point_time_and_context_to_a_zero_velocity():
    network = VelocityMLP(5, window=3, width=8, depth=2, key=jax.random.key(0))

    velocity = network(jnp.ones(5), jnp.asarray(0.3), jnp.ones((3, 5)))

    assert velocity.shape == (5,)
    assert np.all(velocity == 0.0)


def test_the_point_the_time_and_the_context_all_reach_the_velocity():
    network = VelocityMLP(5, window=3, width=8, depth=2, key=jax.random.key(0))
    # A random head: the init's is zero, and one of equal weights sums the
    # zero-mean LayerNorm output to zero whatever the input.
    head = jax.random.normal(jax.random.key(1), network.head.weight.shape)
    network = eqx.tree_at(lambda n: n.head.weight, network, head)
    x, t, condition = jnp.ones(5), jnp.asarray(0.3), jnp.ones((3, 5))

    velocity = network(x, t, condition)

    assert not np.allclose(network(2.0 * x, t, condition), velocity)
    assert not np.allclose(network(x, jnp.asarray(0.7), condition), velocity)
    assert not np.allclose(network(x, t, condition.at[0].set(2.0)), velocity)


def test_the_network_needs_a_block():
    with pytest.raises(ValueError, match="depth must be >= 1"):
        VelocityMLP(5, window=3, width=8, depth=0, key=jax.random.key(0))


def dot_precisions(jaxpr):
    """The precision of every matmul in `jaxpr` and the jaxprs inside it."""
    found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            found.append(eqn.params["precision"])
        for param in eqn.params.values():
            inner = getattr(param, "jaxpr", param)
            if hasattr(inner, "eqns"):
                found += dot_precisions(inner)
    return found


def test_every_matmul_of_the_training_step_is_float32_on_any_device():
    """The loss's gradient, the network's forward pass included, asks for
    full float32 in every matmul. At the GPU's default (TF32), a member
    sampled in a batch rounds differently from the same member alone."""
    network = VelocityMLP(5, window=3, width=8, depth=2, key=jax.random.key(0))
    scales = LatentScales(jnp.zeros(5), jnp.ones(5), jnp.ones(5))
    batch = jnp.ones((4, 4, 5))

    def loss(network):
        return latent_flow_loss(network, scales, batch, 3, jax.random.key(0))

    jaxpr, _, _ = eqx.filter_make_jaxpr(eqx.filter_grad(loss))(network)
    precisions = dot_precisions(jaxpr.jaxpr)

    highest = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    # More than the forward pass's 10 (input, 2 in the time MLP, 3 per block, head).
    assert len(precisions) > 10
    assert all(precision == highest for precision in precisions)


# -- units and the loss ------------------------------------------------------------


def test_scales_are_per_mode_with_one_step_changes_within_a_dataset():
    arrays = [
        np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        np.array([[10.0, 0.0], [12.0, 0.0]], dtype=np.float32),
    ]

    mean, std, step_rms = fit_scales(arrays)

    np.testing.assert_allclose(mean, [5.0, 1.0])
    np.testing.assert_allclose(std, np.concatenate(arrays).std(axis=0), rtol=1e-6)
    # Changes 1, 1 and 2 in the first mode (not the 2 -> 10 between
    # datasets), 0, 2 and 0 in the second.
    np.testing.assert_allclose(step_rms, [np.sqrt(2.0), np.sqrt(4.0 / 3.0)], rtol=1e-6)


def test_a_mode_that_never_changes_is_refused():
    arrays = [np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float32)]

    with pytest.raises(ValueError, match=r"latent modes \[1\]"):
        fit_scales(arrays)


SCALES = LatentScales(jnp.array([1.0, -2.0]), jnp.array([2.0, 0.5]), jnp.array([0.5, 4.0]))


def test_the_loss_is_flow_matching_on_the_change_in_units_of_its_rms():
    """A zero field scores the target's mean square plus 1: here the change
    is (1, -8) = (2, -2) step RMSs, so 4 + 1."""
    sequence = jnp.array([[0.0, 0.0], [1.0, 2.0], [2.0, -6.0]])  # window 2, then the target
    batch = jnp.broadcast_to(sequence, (20_000, 3, 2))

    loss = latent_flow_loss(lambda x, t, c: jnp.zeros_like(x), SCALES, batch, 2, jax.random.key(0))

    assert float(loss) == pytest.approx(5.0, rel=0.01)


def test_input_noise_is_in_step_rms_and_the_target_undoes_it():
    """On a context at the mean and no change, noise of n step RMSs makes
    the target minus n times the newest latent's draw: a zero field scores
    1 + n**2. A field that reads the newest latent's perturbation off the
    standardized context (its draw x n x step RMS / std) and returns minus
    it hits the target, leaving flow matching's own 1; reading the oldest
    latent's, an independent draw, it scores 1 + 2 n**2."""

    def zero(x, t, condition):
        return jnp.zeros_like(x)

    def undo(frame):
        return lambda x, t, condition: -condition[frame] * SCALES.std / SCALES.step_rms

    batch = jnp.broadcast_to(jnp.array([1.0, -2.0]), (20_000, 3, 2))

    def loss(network, noise_std):
        key = jax.random.key(1)
        return float(latent_flow_loss(network, SCALES, batch, 2, key, input_noise_std=noise_std))

    assert loss(zero, 0.0) == pytest.approx(1.0, rel=0.02)
    assert loss(zero, 0.5) == pytest.approx(1.25, rel=0.02)
    assert loss(undo(-1), 0.5) == pytest.approx(1.0, rel=0.02)
    assert loss(undo(0), 0.5) == pytest.approx(1.5, rel=0.02)


# -- sampling ---------------------------------------------------------------------


def test_an_untrained_model_steps_each_mode_by_its_noise_times_its_step_rms(codec_dir):
    """A zero velocity field leaves the noise where it is: each step's
    change is the step's noise (drawn from the seed and the step's index)
    in units of the mode's step RMS."""
    model = small_model(codec_dir, training={**TRAINING, "max_epochs": 1, "learning_rate": 0.0})
    model.fit(noisy_datasets())
    window = model.codec.encode(context()[-2:])

    latents = model.sample_latents(window, 4, [3], first_step=5)[0]

    previous = np.concatenate([window[-1:], latents[:-1]])
    noise = np.stack([step_noise(jax.random.key(3), 5 + k, (RANK,)) for k in range(4)])
    np.testing.assert_allclose(latents - previous, model.step_rms * noise, rtol=1e-5, atol=1e-6)


def test_a_seed_gives_its_own_sample_every_time(fitted):
    first = fitted.predict(context(), 5, seed=1)

    np.testing.assert_array_equal(fitted.predict(context(), 5, seed=1), first)
    assert not np.allclose(fitted.predict(context(), 5, seed=2), first)
    assert first.shape == (5, *FRAME)


def test_a_member_is_its_seed_alone_or_in_a_batch(fitted):
    members = fitted.predict_members(context(), 6, [3, 0, 5])

    assert members.shape == (3, 6, *FRAME)
    for member, seed in zip(members, [3, 0, 5], strict=True):
        np.testing.assert_allclose(
            member, fitted.predict(context(), 6, seed=seed), rtol=1e-5, atol=1e-6
        )


def test_streamed_blocks_are_the_one_forecast(fitted):
    window = fitted.codec.encode(context()[-2:])
    whole = fitted.sample_latents(window, 11, [4])[0]

    blocks = list(fitted.predict_blocks(context(), 11, 4, seed=4))

    assert [len(b) for b in blocks] == [4, 4, 3]
    # The carried state is the latents themselves: exact, not up to rounding.
    first = fitted.sample_latents(window, 4, [4])[0]
    second = fitted.sample_latents(np.concatenate([window, first])[-2:], 4, [4], first_step=4)[0]
    np.testing.assert_array_equal(np.concatenate([first, second]), whole[:8])
    np.testing.assert_allclose(
        np.concatenate(blocks), fitted.predict(context(), 11, seed=4), rtol=1e-6, atol=1e-6
    )


def test_the_protocol_streams_and_batches_its_members(fitted):
    series = noisy_datasets(seed=2)["d0"][:9]

    members = list(forecast_members(fitted, series, 3, [1, 2], batch_size=2))
    streamed = np.concatenate(list(rollout_blocks(fitted, series[:3], 6, 4, seed=2)))

    assert is_stochastic(fitted)
    np.testing.assert_allclose(members[1], fitted.predict(series[1:3], 6, seed=2), rtol=1e-5)
    np.testing.assert_allclose(streamed, members[1], rtol=1e-5, atol=1e-6)


# -- fitting ------------------------------------------------------------------------


def test_fit_trains_validates_logs_and_saves_resumable_state(tmp_path, codec_dir):
    model = small_model(codec_dir, training={**TRAINING, "max_epochs": 3})
    validated = []
    logged = []
    hooks = FitHooks(
        validate=lambda m: (
            validated.append(m.predict(context(), 4, seed=0))
            or {"selection_score": -float(len(validated))}
        ),
        log_metrics=lambda metrics, step: logged.append(metrics),
        state_dir=tmp_path,
    )

    model.fit(noisy_datasets(), hooks)

    assert len(validated) == 3 and validated[0].shape == (4, *FRAME)
    assert model.fit_info["best_epoch"] == 1
    assert any("loss" in m for m in logged)
    assert (tmp_path / "state.json").exists()
    assert model.datasets == ["d0", "d1"]


def test_training_lowers_the_loss(codec_dir):
    datasets = noisy_datasets()
    untrained = small_model(codec_dir, training={**TRAINING, "learning_rate": 0.0})
    untrained.fit(datasets)
    trained = small_model(codec_dir, training={**TRAINING, "max_epochs": 30, "learning_rate": 3e-3})
    trained.fit(datasets)
    # Every training window: the changes there have unit mean square.
    latents = [untrained.codec.encode(frames) for frames in datasets.values()]
    batch = jnp.stack([z[s : s + 3] for z in latents for s in range(len(z) - 2)])

    def loss(model):
        key = jax.random.key(0)
        return float(latent_flow_loss(model.network, model._scales(), batch, 2, key))

    assert loss(untrained) == pytest.approx(2.0, rel=0.15)
    assert loss(trained) < 0.85 * loss(untrained)


def test_save_and_load_reproduce_the_forecast(tmp_path, fitted):
    fitted.save(tmp_path / "model")

    loaded = load_model(tmp_path / "model")

    assert isinstance(loaded, LatentFlow)
    assert loaded.config() == fitted.config() and loaded.window == 2
    assert loaded.fit_info == fitted.fit_info and loaded.datasets == ["d0", "d1"]
    np.testing.assert_array_equal(
        loaded.predict(context(), 5, seed=3), fitted.predict(context(), 5, seed=3)
    )


def test_parameter_count_is_the_networks_weights_plus_the_codecs(fitted):
    width, size, inputs = 16, RANK, 3 * RANK  # the window of 2 and the point x_t
    time = (2 * TIME_FREQUENCIES + 1) * TIME_WIDTH + (TIME_WIDTH + 1) * TIME_WIDTH
    block = (TIME_WIDTH + 1) * 2 * width + 2 * (width + 1) * width

    expected = (inputs + 1) * width + time + block + 2 * width + (width + 1) * size

    assert fitted.latent_parameter_count() == expected
    assert fitted.parameter_count() == expected + fitted.codec.parameter_count()


# -- configuration ----------------------------------------------------------------


def test_registry_builds_it_from_a_config():
    model = build_model(
        {"name": "latent_flow", "codec": "unused", "window": 3, "training": {"batch_size": 4}}
    )

    assert isinstance(model, LatentFlow)
    assert model.window == 3 and model.trainer_config.batch_size == 4
    assert model.iterative and is_stochastic(model)


def test_seed_is_the_trainer_seed_too(codec_dir):
    assert small_model(codec_dir, seed=5).trainer_config.seed == 5


def test_requirements_add_the_network_libraries_to_the_codecs(fitted):
    assert {"equinox", "optax"} <= set(fitted.requirements)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"window": 0}, "window"),
        ({"width": 0}, "width"),
        ({"depth": 0}, "depth"),
        ({"sampling_steps": 0}, "sampling_steps"),
        ({"input_noise_std": -0.1}, "input_noise_std"),
    ],
)
def test_rejects_invalid_settings(codec_dir, kwargs, match):
    with pytest.raises(ValueError, match=match):
        small_model(codec_dir, **kwargs)


def test_fit_needs_datasets_longer_than_the_window(codec_dir):
    with pytest.raises(ValueError, match="more than window=2"):
        small_model(codec_dir).fit({"d0": noisy_datasets()["d0"][:2]})


def test_use_before_fit_is_rejected(codec_dir):
    with pytest.raises(RuntimeError, match="not fitted"):
        small_model(codec_dir).predict(context(), 3, seed=0)
    with pytest.raises(RuntimeError, match="not fitted"):
        small_model(codec_dir).sample_latents(np.zeros((2, RANK)), 3, [0])
