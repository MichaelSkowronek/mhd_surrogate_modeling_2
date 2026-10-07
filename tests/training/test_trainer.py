import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mhd_surrogate.models.base import FitHooks
from mhd_surrogate.training.tracking import DivergenceError
from mhd_surrogate.training.trainer import (
    STATE_FILE,
    TrainerConfig,
    WindowSampler,
    make_optimizer,
    prefetch,
    train,
)


class Gain(eqx.Module):
    """x_{t+1} = gain * x_t: one parameter to learn."""

    gain: jax.Array

    def __call__(self, x):
        return self.gain * x


def decay_series(lengths=(12, 9), factor=0.5):
    """Datasets of x_t = x_0 * factor**t, so the right gain is `factor`."""
    rng = np.random.default_rng(0)
    return [
        (rng.uniform(1, 2, size=(1, 3)) * factor ** np.arange(n)[:, None]).astype(np.float32)
        for n in lengths
    ]


def one_step_loss(network, batch, key):
    return jnp.mean((network(batch[:, 0]) - batch[:, 1]) ** 2)


def noisy_loss(network, batch, key):
    """The one-step loss with the input perturbed by noise from `key`."""
    noise = 0.3 * jax.random.normal(key, batch[:, 0].shape)
    return jnp.mean((network(batch[:, 0] + noise) - batch[:, 1]) ** 2)


def config(**kwargs):
    defaults = dict(
        max_epochs=4,
        batch_size=2,
        learning_rate=0.1,
        warmup_steps=0,
        log_every=1,
        prefetch=0,
        weight_decay=0.0,
    )
    return TrainerConfig(**{**defaults, **kwargs})


class Recorder:
    def __init__(self):
        self.records = []

    def __call__(self, metrics, step):
        self.records.append((step, dict(metrics)))

    def keys(self):
        return set().union(*(m for _, m in self.records))


# -- configuration --------------------------------------------------------------


@pytest.mark.parametrize("field", ["max_epochs", "batch_size", "patience", "eval_every"])
def test_config_rejects_non_positive_counts(field):
    with pytest.raises(ValueError, match=field):
        TrainerConfig(**{field: 0})


def test_learning_rate_warms_up_linearly_then_decays_to_the_final_fraction():
    _, schedule = make_optimizer(
        TrainerConfig(learning_rate=1.0, warmup_steps=10, final_lr_fraction=0.1), 100
    )

    assert float(schedule(0)) == 0.0
    assert float(schedule(5)) == pytest.approx(0.5)
    assert float(schedule(10)) == pytest.approx(1.0)
    assert float(schedule(100)) == pytest.approx(0.1)
    assert float(schedule(55)) < float(schedule(20))


# -- sampling -------------------------------------------------------------------


def test_sampler_windows_never_cross_a_dataset_boundary():
    arrays = [np.zeros((5, 1)), np.ones((3, 1))]
    sampler = WindowSampler(arrays, span=3)

    # 3 windows in the first dataset, 1 in the second.
    assert len(sampler) == 4
    for batch in sampler.epoch(np.random.default_rng(0), batch_size=1):
        assert np.all(batch == batch[0, 0])


def test_sampler_epoch_uses_every_window_once_and_drops_the_partial_batch():
    arrays = [np.arange(10.0)[:, None], 100 + np.arange(7.0)[:, None]]
    sampler = WindowSampler(arrays, span=2)  # 9 + 6 = 15 windows

    batches = list(sampler.epoch(np.random.default_rng(0), batch_size=4))

    assert len(batches) == 3 == sampler.steps_per_epoch(4, None)
    starts = [float(window[0, 0]) for batch in batches for window in batch]
    assert len(set(starts)) == 12
    for batch in batches:
        np.testing.assert_array_equal(batch[:, 1] - batch[:, 0], 1.0)


def test_sampler_cycles_through_fresh_permutations_for_a_longer_epoch():
    sampler = WindowSampler([np.arange(4.0)[:, None]], span=1)

    starts = [float(b[0, 0, 0]) for b in sampler.epoch(np.random.default_rng(0), 1, 10)]

    assert len(starts) == 10
    assert sorted(starts[:4]) == sorted(starts[4:8]) == [0.0, 1.0, 2.0, 3.0]


def test_sampler_is_reproducible_from_the_generator_state():
    sampler = WindowSampler(decay_series(), span=2)
    a = list(sampler.epoch(np.random.default_rng(7), 2))
    b = list(sampler.epoch(np.random.default_rng(7), 2))

    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_sampler_rejects_data_too_short_for_a_window():
    with pytest.raises(ValueError, match="3 frames"):
        WindowSampler([np.zeros((2, 1))], span=3)


def test_sampler_rejects_an_epoch_without_a_full_batch():
    sampler = WindowSampler([np.zeros((5, 1))], span=1)

    with pytest.raises(ValueError, match="no full batch"):
        next(sampler.epoch(np.random.default_rng(0), batch_size=8))


@pytest.mark.parametrize("size", [0, 1, 3])
def test_prefetch_yields_every_batch_in_order(size):
    batches = [np.full((2,), i) for i in range(5)]

    out = [np.asarray(x) for x in prefetch(iter(batches), size)]

    assert [x[0] for x in out] == [0, 1, 2, 3, 4]


def test_prefetch_reraises_an_error_from_the_producer():
    def failing():
        yield np.zeros(1)
        raise RuntimeError("bad batch")

    with pytest.raises(RuntimeError, match="bad batch"):
        list(prefetch(failing(), 2))


# -- training -------------------------------------------------------------------


def run(tmp_path=None, validate=None, network=None, loss=one_step_loss, **kwargs):
    log = Recorder()
    hooks = FitHooks(log_metrics=log, state_dir=tmp_path)
    result = train(
        network or Gain(jnp.array(1.0)),
        loss,
        WindowSampler(decay_series(), span=2),
        config(**kwargs),
        hooks,
        validate,
    )
    return result, log


def test_learns_the_decay_factor():
    result, log = run(max_epochs=30)

    assert float(result.network.gain) == pytest.approx(0.5, abs=0.02)
    losses = [m["epoch_loss"] for _, m in log.records if "epoch_loss" in m]
    assert losses[-1] < losses[0] / 10


def test_logs_step_and_epoch_metrics_at_increasing_steps():
    result, log = run(max_epochs=2, log_every=2)

    assert {"loss", "grad_norm", "lr", "samples_per_second", "epoch", "epoch_loss"} <= log.keys()
    steps = [step for step, _ in log.records]
    assert steps == sorted(steps)
    # 9 + 6 windows of 2 frames -> 18 // 2 = 9 steps per epoch.
    assert result.info["steps"] == 18
    assert result.info["epochs"] == 2


class ScriptedValidation:
    """Returns the scores in `script`, one per call, and remembers the gain
    it was shown each time."""

    def __init__(self, script):
        self.script = list(script)
        self.gains = []

    def __call__(self, network):
        self.gains.append(float(network.gain))
        return {"selection_score": self.script[len(self.gains) - 1], "skill_horizon": 1.0}


def test_early_stopping_returns_the_best_network_after_patience_runs_out():
    validate = ScriptedValidation([1.0, 2.0, 2.0, 1.5, 9.0])

    result, log = run(validate=validate, max_epochs=5, patience=2)

    # Epoch 2 is best; 3 (a tie is no improvement) and 4 don't improve: stop.
    assert len(validate.gains) == 4
    assert result.info["stopped_early"] == 1.0
    assert result.info["best_epoch"] == 2
    assert result.info["best_selection_score"] == 2.0
    assert float(result.network.gain) == validate.gains[1]
    assert "val_monitor.selection_score" in log.keys()


def test_validates_every_eval_every_epochs_and_after_the_last():
    validate = ScriptedValidation([1.0, 2.0, 3.0])

    result, _ = run(validate=validate, max_epochs=5, eval_every=2)

    assert len(validate.gains) == 3  # epochs 2, 4 and 5
    assert result.info["best_epoch"] == 5


def test_without_validation_the_last_network_is_returned():
    result, _ = run(max_epochs=3)

    assert result.info["best_epoch"] == 0
    assert "best_selection_score" not in result.info


@pytest.mark.parametrize("bad", [jnp.nan, 1e6])
def test_a_non_finite_or_blown_up_loss_is_divergence(bad):
    def loss(network, batch, key):
        return one_step_loss(network, batch, key) * 0 + bad

    with pytest.raises(DivergenceError, match="diverged"):
        train(
            Gain(jnp.array(1.0)),
            loss,
            WindowSampler(decay_series(), 2),
            config(log_every=100),
            FitHooks(),
        )


class Interrupt(Exception):
    pass


def test_resume_continues_exactly_where_an_interrupted_run_stopped(tmp_path):
    uninterrupted, _ = run(tmp_path / "a", validate=ScriptedValidation([1, 2, 3, 4]))

    def crash_at_epoch_3(network, calls=[]):  # noqa: B006 -- counts calls
        calls.append(1)
        if len(calls) == 3:
            raise Interrupt
        return {"selection_score": float(len(calls))}

    with pytest.raises(Interrupt):
        run(tmp_path / "b", validate=crash_at_epoch_3)
    state = json.loads((tmp_path / "b" / STATE_FILE).read_text())
    assert state["epoch"] == 2

    resumed, log = run(tmp_path / "b", validate=ScriptedValidation([3, 4]))

    assert resumed.info["resumed_at_epoch"] == 2
    assert float(resumed.network.gain) == float(uninterrupted.network.gain)
    assert min(step for step, _ in log.records) > state["step"]


def test_a_random_loss_gets_a_new_key_every_step_derived_from_the_seed():
    first, _ = run(loss=noisy_loss, max_epochs=2)
    again, _ = run(loss=noisy_loss, max_epochs=2)
    other_seed, _ = run(loss=noisy_loss, max_epochs=2, seed=1)

    assert float(first.network.gain) == float(again.network.gain)
    assert float(first.network.gain) != float(other_seed.network.gain)


def test_a_resumed_random_loss_draws_the_keys_an_uninterrupted_run_would(tmp_path):
    uninterrupted, _ = run(
        tmp_path / "a", loss=noisy_loss, max_epochs=3, validate=ScriptedValidation([1, 2, 3])
    )

    def crash_at_epoch_2(network, calls=[]):  # noqa: B006 -- counts calls
        calls.append(1)
        if len(calls) == 2:
            raise Interrupt
        return {"selection_score": 1.0}

    with pytest.raises(Interrupt):
        run(tmp_path / "b", loss=noisy_loss, max_epochs=3, validate=crash_at_epoch_2)
    resumed, _ = run(
        tmp_path / "b", loss=noisy_loss, max_epochs=3, validate=ScriptedValidation([2, 3])
    )

    assert resumed.info["resumed_at_epoch"] == 1
    assert float(resumed.network.gain) == float(uninterrupted.network.gain)


def test_resume_keeps_the_best_network_from_before_the_interruption(tmp_path):
    run(tmp_path, validate=ScriptedValidation([5.0, 1.0]), max_epochs=2)
    first = Gain(jnp.array(1.0))

    resumed, _ = run(tmp_path, validate=ScriptedValidation([]), max_epochs=2, network=first)

    # Already finished: nothing is trained, the epoch-1 best comes back.
    assert resumed.info["best_epoch"] == 1
    assert float(resumed.network.gain) != 1.0


def test_state_dir_keeps_only_the_current_weights(tmp_path):
    run(tmp_path, validate=ScriptedValidation([1.0, 3.0, 2.0]), max_epochs=3)

    state = json.loads((tmp_path / STATE_FILE).read_text())
    files = sorted(p.name for p in tmp_path.glob("*.eqx"))
    assert files == ["best-epoch-0002.eqx", "latest-epoch-0003.eqx"]
    assert {state["best_file"], state["latest_file"]} == set(files)


def test_resume_refuses_a_different_configuration(tmp_path):
    run(tmp_path, max_epochs=1)

    with pytest.raises(ValueError, match="different configuration"):
        run(tmp_path, max_epochs=1, learning_rate=0.5)
