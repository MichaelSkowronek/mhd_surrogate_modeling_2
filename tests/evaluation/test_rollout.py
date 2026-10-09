import numpy as np
import pytest

from mhd_surrogate.evaluation.rollout import rollout_block_means, rollout_stability
from mhd_surrogate.evaluation.stability import block_means, energy_and_enstrophy

DX, DY = 0.5, 0.25


class Doubling:
    """Each step doubles the newest frame; reads a window of `window`."""

    def __init__(self, window):
        self.window = window
        self.calls = []

    def predict(self, context, n_steps):
        self.calls.append((len(context), n_steps))
        last = context[-1]
        return np.stack([last * 2.0 ** (k + 1) for k in range(n_steps)])


def test_rollout_block_means_continue_from_the_models_own_last_frames():
    rng = np.random.default_rng(1)
    context = rng.normal(size=(5, 2, 6, 5))
    model = Doubling(window=2)

    blocks = rollout_block_means(model, context, n_steps=7, block_steps=3, dx=DX, dy=DY)

    # One predict per block, each from the last `window` frames, the last
    # block shorter; the same as one 7-step rollout.
    assert model.calls == [(2, 3), (2, 3), (2, 1)]
    whole = Doubling(window=2).predict(context[-2:], 7)
    expected = energy_and_enstrophy(whole, DX, DY)
    assert blocks["energy"] == pytest.approx(block_means(expected["energy"], 3))
    assert blocks["enstrophy"] == pytest.approx(block_means(expected["enstrophy"], 3))


def test_rollout_block_means_of_a_model_without_a_window_pass_it_no_frames():
    class Constant:
        window = 0

        def predict(self, context, n_steps):
            assert len(context) == 0
            return np.ones((n_steps, 2, 4, 4))

    blocks = rollout_block_means(Constant(), np.zeros((3, 2, 4, 4)), 5, 2, DX, DY)

    assert blocks["energy"] == pytest.approx([1.0, 1.0, 1.0])
    assert blocks["enstrophy"] == pytest.approx([0.0, 0.0, 0.0])


def test_rollout_block_means_sample_each_block_of_a_stochastic_model_with_its_own_seed():
    class Sampler:
        window = 1
        stochastic = True

        def __init__(self):
            self.seeds = []

        def predict(self, context, n_steps, seed=0):
            self.seeds.append(seed)
            return np.ones((n_steps, 2, 4, 4))

    model = Sampler()

    rollout_block_means(model, np.zeros((3, 2, 4, 4)), 5, 2, DX, DY)

    assert model.seeds == [0, 1, 2]


class Growing:
    """Each step scales the newest frame by `factor`, so energy and
    enstrophy grow by `factor**2` per step."""

    window = 1

    def __init__(self, factor):
        self.factor = factor

    def predict(self, context, n_steps):
        return np.stack([context[-1] * self.factor ** (k + 1) for k in range(n_steps)])


def test_rollout_stability_rates_the_rollout_against_the_truths_largest_block():
    rng = np.random.default_rng(2)
    frame = rng.normal(size=(2, 6, 5))
    context, targets = np.stack([frame] * 3), np.stack([frame] * 6)

    # Energy ratios sqrt(2), 2, 2*sqrt(2), 4 per 1-step block: over 2x from
    # step 2 (2x itself is still bounded).
    rollout = rollout_stability(Growing(2**0.25), context, targets, 4, 1, 2.0, DX, DY)

    assert rollout.ratios["energy"] == pytest.approx(2.0 ** (np.arange(1, 5) / 2))
    assert rollout.ratios["enstrophy"] == pytest.approx(2.0 ** (np.arange(1, 5) / 2))
    assert rollout.stable_steps == 2
    assert rollout.scores() == pytest.approx(
        {"stable_steps": 2.0, "energy_peak_ratio": 4.0, "enstrophy_peak_ratio": 4.0}
    )
    assert len(rollout.reference["energy"]) == 6  # the truth's own blocks


def test_a_bounded_rollout_is_stable_for_all_its_steps():
    frame = np.random.default_rng(3).normal(size=(2, 6, 5))
    context, targets = np.stack([frame] * 3), np.stack([frame] * 4)

    rollout = rollout_stability(Growing(1.0), context, targets, 10, 3, 2.0, DX, DY)

    assert rollout.stable_steps == 10
    assert rollout.scores()["energy_peak_ratio"] == pytest.approx(1.0)
