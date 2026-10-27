import numpy as np
import pytest

from mhd_surrogate.evaluation.rollout import (
    block_seed,
    rollout_block_means,
    rollout_blocks,
    rollout_stability,
)
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


class Sampler:
    """A stochastic model that records the seed of every call."""

    window = 1
    stochastic = True

    def __init__(self):
        self.seeds = []

    def predict(self, context, n_steps, seed=0):
        self.seeds.append(seed)
        return np.ones((n_steps, 2, 4, 4))


def test_a_restarted_stochastic_model_samples_each_block_with_its_own_seed():
    first, second = Sampler(), Sampler()

    list(rollout_blocks(first, np.zeros((3, 2, 4, 4)), 5, 2, seed=3))
    list(rollout_blocks(second, np.zeros((3, 2, 4, 4)), 5, 2, seed=4))

    # The first block is the member's own sample, as its `predict` starts.
    assert first.seeds[0] == 3 and second.seeds[0] == 4
    assert first.seeds == [block_seed(3, k) for k in range(3)]
    assert len(set(first.seeds + second.seeds)) == 6


def test_a_stochastic_rollout_by_default_is_the_member_for_seed_0():
    model = Sampler()

    rollout_block_means(model, np.zeros((3, 2, 4, 4)), 5, 2, DX, DY)

    assert model.seeds == [block_seed(0, k) for k in range(3)]


def test_a_stochastic_rollout_is_the_member_for_the_seed_asked():
    model = Sampler()

    rollout_stability(
        model, np.zeros((3, 2, 4, 4)), np.ones((6, 2, 4, 4)), 5, 2, 2.0, DX, DY, seed=3
    )

    assert model.seeds == [block_seed(3, k) for k in range(3)]


class Clock:
    """A model with a hidden state: frame k of a forecast is k + 1, whatever
    the context, so restarting it at a block starts the count over."""

    window = 1

    def predict(self, context, n_steps):
        return np.arange(1.0, n_steps + 1)[:, None, None, None] * np.ones((n_steps, 2, 3, 3))


class StreamingClock(Clock):
    """A `Clock` that streams its forecast, carrying the count."""

    def __init__(self):
        self.calls = []

    def predict_blocks(self, context, n_steps, block_steps):
        self.calls.append((len(context), n_steps, block_steps))
        whole = self.predict(context, n_steps)
        for start in range(0, n_steps, block_steps):
            yield whole[start : start + block_steps]


def test_rollout_blocks_stream_a_model_that_can_and_restart_one_that_cannot():
    context = np.zeros((4, 2, 3, 3))
    model = StreamingClock()

    streamed = list(rollout_blocks(model, context, 7, 3))
    restarted = list(rollout_blocks(Clock(), context, 7, 3))

    assert model.calls == [(1, 7, 3)]
    assert [len(b) for b in streamed] == [3, 3, 1]
    np.testing.assert_array_equal(np.concatenate(streamed), Clock().predict(context, 7))
    # The hidden state is lost at every restart.
    assert [b[:, 0, 0, 0].tolist() for b in restarted] == [[1, 2, 3], [1, 2, 3], [1]]


def test_rollout_blocks_pass_a_seed_to_a_streaming_model_only_if_it_is_stochastic():
    class StreamingSampler(Sampler):
        def predict_blocks(self, context, n_steps, block_steps, seed=0):
            self.seeds.append(seed)
            yield self.predict(context, n_steps, seed)

    model = StreamingSampler()

    list(rollout_blocks(model, np.zeros((3, 2, 4, 4)), 4, 4, seed=5))
    # A deterministic model's predict_blocks takes no seed: this would raise.
    list(rollout_blocks(StreamingClock(), np.zeros((3, 2, 3, 3)), 4, 4, seed=5))

    # Once from predict_blocks, once from the predict it calls.
    assert model.seeds == [5, 5]


@pytest.mark.parametrize(
    "blocks",
    [
        [np.ones((2, 2, 3, 3)), np.ones((2, 2, 3, 3))],  # too long a last block
        [np.ones((2, 2, 3, 3))],  # too few blocks
        [np.ones((2, 2, 3, 3)), np.ones((1, 2, 3, 3)), np.ones((1, 2, 3, 3))],  # too many
        [np.ones((2, 2, 3, 4)), np.ones((1, 2, 3, 3))],  # wrong frame shape
    ],
)
def test_rollout_blocks_check_what_a_streaming_model_yields(blocks):
    class Wrong(Clock):
        def predict_blocks(self, context, n_steps, block_steps):
            yield from blocks

    with pytest.raises(ValueError):
        list(rollout_blocks(Wrong(), np.zeros((2, 2, 3, 3)), 3, 2))


def test_rollout_blocks_need_a_positive_block_size():
    with pytest.raises(ValueError, match="block_steps"):
        list(rollout_blocks(Clock(), np.zeros((2, 2, 3, 3)), 3, 0))


def test_block_seed_is_the_members_own_for_the_first_block_and_fixed_after():
    assert block_seed(7, 0) == 7
    assert block_seed(7, 1) == block_seed(7, 1) != block_seed(7, 2)
    assert block_seed(7, 1) != block_seed(8, 1)


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
