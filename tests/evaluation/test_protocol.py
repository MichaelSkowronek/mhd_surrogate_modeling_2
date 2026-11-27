import time

import numpy as np
import pytest

from mhd_surrogate.evaluation.protocol import (
    MemberBlocks,
    StartBlocks,
    check_readable,
    check_window,
    ensemble_size,
    forecast,
    forecast_members,
    predict_member,
    rollout_seed,
    skill_starts,
    split_context,
    start_seeds,
)
from mhd_surrogate.models.base import DeviceBlock


class Persistence:
    """Repeats the last context frame; reads a window of 1."""

    window = 1

    def predict(self, context, n_steps):
        return np.repeat(context[-1:], n_steps, axis=0)


class Recorder:
    """Records what the protocol hands it and returns zeros."""

    def __init__(self, window):
        self.window = window
        self.seen = None

    def predict(self, context, n_steps):
        self.seen = context
        return np.zeros((n_steps, 2, 3, 4))


def make_series(n=10):
    """Step t is filled with the value t, so slices identify their steps."""
    return np.arange(n, dtype=float).reshape(n, 1, 1, 1) * np.ones((n, 2, 3, 4))


def test_split_context_separates_first_steps_from_scored_targets():
    context, targets = split_context(make_series(10), 4)

    assert context.shape[0] == 4 and targets.shape[0] == 6
    assert context[:, 0, 0, 0].tolist() == [0, 1, 2, 3]
    assert targets[:, 0, 0, 0].tolist() == [4, 5, 6, 7, 8, 9]


@pytest.mark.parametrize("n, context_steps", [(4, 4), (3, 4), (10, 0)])
def test_split_context_rejects_series_without_a_scored_step_or_bad_context(n, context_steps):
    with pytest.raises(ValueError):
        split_context(make_series(n), context_steps)


def test_check_window_allows_up_to_the_context_and_rejects_more():
    check_window(0, 4)
    check_window(4, 4)
    with pytest.raises(ValueError):
        check_window(5, 4)
    with pytest.raises(ValueError):
        check_window(-1, 4)


@pytest.mark.parametrize("window", [0, 1, 3, 4])
def test_forecast_shows_the_model_only_the_last_window_context_steps(window):
    model = Recorder(window)

    forecast(model, make_series(10), context_steps=4)

    assert model.seen[:, 0, 0, 0].tolist() == list(range(4 - window, 4))


def test_forecast_returns_the_prediction_and_the_same_targets_for_any_model():
    series = make_series(10)

    prediction, targets = forecast(Persistence(), series, context_steps=4)

    assert targets[:, 0, 0, 0].tolist() == [4, 5, 6, 7, 8, 9]
    assert (prediction == 3.0).all()  # the last context frame, repeated


def test_forecast_rejects_a_window_longer_than_the_context():
    with pytest.raises(ValueError, match="window"):
        forecast(Recorder(5), make_series(10), context_steps=4)


def test_forecast_rejects_a_prediction_of_the_wrong_shape():
    class Wrong:
        window = 0

        def predict(self, context, n_steps):
            return np.zeros((n_steps + 1, 2, 3, 4))

    with pytest.raises(ValueError, match="shape"):
        forecast(Wrong(), make_series(10), context_steps=4)


DATA_CONFIG = {"train_datasets": ["a", "b"], "val_dataset": "v", "test_dataset": "t"}


def test_check_readable_allows_validation_and_training_datasets():
    check_readable("v", DATA_CONFIG)
    check_readable("b", DATA_CONFIG)


def test_check_readable_refuses_the_test_dataset_and_unknown_names():
    with pytest.raises(ValueError, match="test dataset"):
        check_readable("t", DATA_CONFIG)
    with pytest.raises(ValueError, match="not a validation or training"):
        check_readable("x", DATA_CONFIG)


class Sampler:
    """A stochastic model: its sample for `seed` is the last context frame
    plus `seed`."""

    window = 1
    stochastic = True

    def __init__(self):
        self.seeds = []

    def predict(self, context, n_steps, seed=0):
        self.seeds.append(seed)
        return np.broadcast_to(context[-1] + seed, (n_steps, *context.shape[1:]))


def test_predict_member_passes_the_seed_to_a_stochastic_model_only():
    context = np.zeros((1, 2, 3, 3))
    sampler = Sampler()

    assert np.all(predict_member(sampler, context, 2, seed=4) == 4.0)
    assert sampler.seeds == [4]
    # Persistence takes no seed: passing one would fail.
    assert np.all(predict_member(Persistence(), context, 2, seed=4) == 0.0)


def test_forecast_is_the_seed_0_member_unless_another_seed_is_asked_for():
    series = np.zeros((6, 2, 3, 3))

    assert np.all(forecast(Sampler(), series, 3)[0] == 0.0)
    assert np.all(forecast(Sampler(), series, 3, seed=2)[0] == 2.0)


class ReadCounter:
    """A series (like a zarr array) that records which slices are read."""

    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.reads = []

    def __getitem__(self, index):
        self.reads.append(index)
        return self.array[index]


def test_forecast_members_yields_one_member_per_seed_reading_only_the_context():
    series = ReadCounter(np.zeros((6, 2, 3, 3)))
    sampler = Sampler()

    members = list(forecast_members(sampler, series, 3, range(1, 4)))

    assert [float(m[0, 0, 0, 0]) for m in members] == [1.0, 2.0, 3.0]
    assert all(m.shape == (3, 2, 3, 3) for m in members)
    assert series.reads == [slice(2, 3)]


class BatchSampler(Sampler):
    """A stochastic model that can batch: each member's noise comes from its
    own seed, so a member is the same alone or in any batch."""

    def __init__(self):
        super().__init__()
        self.batches = []

    def predict(self, context, n_steps, seed=0):
        self.seeds.append(seed)
        rng = np.random.default_rng(seed)
        return context[-1] + rng.normal(size=(n_steps, *context.shape[1:]))

    def predict_members(self, context, n_steps, seeds):
        self.batches.append(list(seeds))
        return np.stack(
            [
                context[-1] + np.random.default_rng(s).normal(size=(n_steps, *context.shape[1:]))
                for s in seeds
            ]
        )


@pytest.mark.parametrize("batch_size, calls", [(2, [[1, 2], [3, 4], [5]]), (8, [[1, 2, 3, 4, 5]])])
def test_forecast_members_batches_a_model_that_can_and_gets_the_same_members(batch_size, calls):
    series = np.zeros((6, 2, 3, 3))
    one_by_one = list(forecast_members(BatchSampler(), series, 3, range(1, 6)))
    model = BatchSampler()

    batched = list(forecast_members(model, series, 3, range(1, 6), batch_size=batch_size))

    assert model.batches == calls
    assert model.seeds == []  # predict was never called
    assert len(batched) == 5
    for alone, in_batch in zip(one_by_one, batched):
        np.testing.assert_array_equal(alone, in_batch)


def test_forecast_members_falls_back_to_one_call_per_member_without_predict_members():
    sampler = Sampler()

    members = list(forecast_members(sampler, np.zeros((6, 2, 3, 3)), 3, [1, 2, 3], batch_size=4))

    assert sampler.seeds == [1, 2, 3]
    assert len(members) == 3


def test_forecast_members_checks_a_batch_and_its_size():
    class Short(BatchSampler):
        def predict_members(self, context, n_steps, seeds):
            return super().predict_members(context, n_steps, seeds)[:-1]

    series = np.zeros((6, 2, 3, 3))
    with pytest.raises(ValueError, match="shape"):
        list(forecast_members(Short(), series, 3, [1, 2], batch_size=2))
    with pytest.raises(ValueError, match="batch_size"):
        list(forecast_members(BatchSampler(), series, 3, [1], batch_size=0))


def test_forecast_members_rejects_a_series_without_scored_steps():
    with pytest.raises(ValueError, match="context_steps"):
        list(forecast_members(Sampler(), np.zeros((3, 2, 3, 3)), 3, [0]))


def test_ensemble_size_is_one_for_a_deterministic_model():
    assert ensemble_size(Sampler(), 8) == 8
    assert ensemble_size(Persistence(), 8) == 1
    with pytest.raises(ValueError, match="n_members"):
        ensemble_size(Sampler(), 0)


def blocks_as_arrays(stream):
    """Every block of a `MemberBlocks` stream, (first, [each member's frames])."""
    return [(b.first, [np.asarray(f) for f in b]) for b in stream]


def test_member_blocks_stream_the_members_forecast_members_makes():
    series = np.random.default_rng(0).normal(size=(12, 2, 3, 3)).astype(np.float32)
    members = list(forecast_members(BatchSampler(), series, 4, [0, 1, 2]))

    stream = MemberBlocks(BatchSampler(), series, 4, [0, 1, 2], block_steps=3)
    blocks = blocks_as_arrays(stream)

    assert len(stream) == 3 and stream.n_steps == 8
    assert [first for first, _ in blocks] == [0, 3, 6]
    for k, member in enumerate(members):
        np.testing.assert_allclose(np.concatenate([b[k] for _, b in blocks]), member, rtol=1e-6)


@pytest.mark.parametrize(
    "batch_size, time_first, calls",
    [(2, False, [[0, 1], [2, 3], [4]]), (2, True, [[1, 2], [3, 4]]), (1, False, [])],
)
def test_member_blocks_batch_the_members_where_the_model_can(batch_size, time_first, calls):
    model = BatchSampler()

    stream = MemberBlocks(
        model, np.zeros((6, 2, 3, 3)), 3, range(5), 2, batch_size, time_first=time_first
    )
    blocks = blocks_as_arrays(stream)

    assert model.batches == calls
    assert model.seeds == ([0] if time_first else []) + ([0, 1, 2, 3, 4] if not calls else [])
    assert all(len(members) == 5 for _, members in blocks)


def test_member_blocks_time_the_first_member_alone():
    class Slow(Sampler):
        def predict(self, context, n_steps, seed=0):
            if seed == 0:
                time.sleep(0.05)
            return super().predict(context, n_steps, seed)

    stream = MemberBlocks(Slow(), np.zeros((6, 2, 3, 3)), 3, range(3), time_first=True)
    blocks_as_arrays(stream)

    assert 0.05 <= stream.first_member_seconds < 1.0
    assert MemberBlocks(Slow(), np.zeros((6, 2, 3, 3)), 3, range(3)).first_member_seconds == 0.0


class DeviceSampler(Sampler):
    """A stochastic model that forecasts on the device: its states are the
    member's offsets, decoded into frames when read."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def predict_device_blocks(self, context, n_steps, block_steps, seeds):
        self.calls.append(list(seeds))
        for start in range(0, n_steps, block_steps):
            n = min(block_steps, n_steps - start)
            states = np.broadcast_to(np.array(seeds, np.float32)[:, None], (len(seeds), n))
            yield DeviceBlock(states, lambda s, last=context[-1]: last + s[:, None, None, None])


def test_member_blocks_take_a_device_models_blocks_seeds_and_all():
    model = DeviceSampler()

    blocks = blocks_as_arrays(MemberBlocks(model, np.zeros((7, 2, 3, 3)), 3, range(3), 3, 2))

    assert model.calls == [[0, 1], [2]] and model.seeds == []
    assert [first for first, _ in blocks] == [0, 3]
    assert [m.shape for m in blocks[1][1]] == [(1, 2, 3, 3)] * 3
    assert [float(m[0, 0, 0, 0]) for m in blocks[0][1]] == [0.0, 1.0, 2.0]


def test_member_blocks_check_what_a_model_gives():
    class Short(DeviceSampler):
        def predict_device_blocks(self, context, n_steps, block_steps, seeds):
            yield from list(super().predict_device_blocks(context, n_steps, block_steps, seeds))[:1]

    class Misshapen(DeviceSampler):
        def predict_device_blocks(self, context, n_steps, block_steps, seeds):
            for block in super().predict_device_blocks(context, n_steps, block_steps, seeds):
                yield DeviceBlock(block.states, lambda s: np.zeros((len(s), 1, 3, 3)))

    class Long(DeviceSampler):
        def predict_device_blocks(self, context, n_steps, block_steps, seeds):
            yield from super().predict_device_blocks(context, n_steps + 1, block_steps, seeds)

    series = np.zeros((7, 2, 3, 3))
    with pytest.raises(ValueError, match="more blocks than the forecast's 2"):
        blocks_as_arrays(MemberBlocks(Long(), series, 3, [0], 2))
    with pytest.raises(ValueError, match="block from lead 3"):
        blocks_as_arrays(MemberBlocks(Short(), series, 3, [0], 2))
    with pytest.raises(ValueError, match="shape"):
        blocks_as_arrays(MemberBlocks(Misshapen(), series, 3, [0], 2))
    with pytest.raises(ValueError, match="shape"):
        blocks_as_arrays(MemberBlocks(Recorder(window=1), series, 3, [0]))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"seeds": []}, "no seeds"),
        ({"block_steps": 0}, "block_steps"),
        ({"batch_size": 0}, "batch_size"),
        ({"context_steps": 7}, "context_steps"),
    ],
)
def test_member_blocks_reject_what_cannot_be_streamed(kwargs, match):
    arguments = {"seeds": [0], "context_steps": 3, **kwargs}
    with pytest.raises(ValueError, match=match):
        MemberBlocks(Sampler(), np.zeros((7, 2, 3, 3)), **arguments)


def test_skill_starts_are_every_stride_frames_from_the_context_that_fit_the_leads():
    # The validation dataset: 917 frames, the last start's leads end with it.
    assert skill_starts(917, 80, 20, 237) == list(range(80, 681, 20))
    assert len(skill_starts(1248, 80, 20, 237)) == 47  # re16k_t400_0
    assert skill_starts(917, 80, 40, 237) == list(range(80, 681, 40))
    assert skill_starts(10, 3, 2, 7) == [3]


@pytest.mark.parametrize(
    "args, match",
    [
        ((9, 3, 2, 7), "no start for 7 leads: the first, frame 3, needs 10"),
        ((10, 3, 0, 7), "stride"),
        ((10, 3, 2, 0), "leads"),
        ((10, 0, 2, 7), "context_steps"),
    ],
)
def test_skill_starts_reject_a_rule_the_series_cannot_meet(args, match):
    with pytest.raises(ValueError, match=match):
        skill_starts(*args)


def test_start_seeds_give_each_start_its_own_and_the_validations_other_ones():
    # Start 0 (the context's end) has seeds 0, 1, ... when scored.
    assert start_seeds(3, 2) == [[0, 1], [1000, 1001], [2000, 2001]]
    assert start_seeds(2, 4, monitor=True) == [[500, 501, 502, 503], [1500, 1501, 1502, 1503]]
    scored = {s for start in start_seeds(24, 500) for s in start}
    monitored = {s for start in start_seeds(24, 500, monitor=True) for s in start}
    assert len(scored) == len(monitored) == 24 * 500
    assert not scored & monitored


@pytest.mark.parametrize(
    "args, match",
    [
        ((0, 2, False), "n_starts"),
        ((3, 0, False), "the scoring has seeds for 1 to 500"),
        ((3, 501, False), "501 members per start: the scoring"),
        ((3, 501, True), "a validation has seeds for 1 to 500"),
    ],
)
def test_start_seeds_refuse_members_beyond_their_range(args, match):
    with pytest.raises(ValueError, match=match):
        start_seeds(*args)


def test_a_validations_rollout_seed_is_new_at_every_validation_and_never_a_members():
    # Start k's validation range is 1000 k + 500 on; its 4 members take the first 4.
    assert rollout_seed(0, 4, 0) == 504
    assert rollout_seed(16, 4, 3) == 16507
    members = {s for start in start_seeds(17, 4, monitor=True) for s in start}
    scored = {s for start in start_seeds(17, 8) for s in start}
    seeds = {rollout_seed(k, 4, v) for k in range(17) for v in range(30)}
    assert len(seeds) == 17 * 30
    assert not seeds & members and not seeds & scored
    # Past the range's room the seeds come round again.
    assert rollout_seed(0, 4, 496) == rollout_seed(0, 4, 0)


@pytest.mark.parametrize(
    "args, match",
    [
        ((-1, 4, 0), "must be >= 0"),
        ((0, 4, -1), "must be >= 0"),
        ((0, 500, 0), "no validation seeds"),
    ],
)
def test_rollout_seeds_refuse_impossible_arguments(args, match):
    with pytest.raises(ValueError, match=match):
        rollout_seed(*args)


def starts_as_arrays(stream):
    """Every block of a `StartBlocks` stream: (first, {start index: [each
    member's frames]})."""
    return [
        (b.first, {k: [np.asarray(f) for f in members] for k, members in b.ensembles.items()})
        for b in stream
    ]


def joined(blocks, k):
    """Start `k`'s members, each its blocks joined."""
    pieces = [members[k] for _, members in blocks if k in members]
    return [np.concatenate([piece[m] for piece in pieces]) for m in range(len(pieces[0]))]


def test_start_blocks_stream_each_starts_members_as_its_own_forecast_makes_them():
    series = np.random.default_rng(0).normal(size=(14, 2, 3, 3)).astype(np.float32)

    seeds = [[0, 1, 2], [10, 11, 12]]  # each start its own

    stream = StartBlocks(BatchSampler(), series, [4, 6], 5, seeds, block_steps=2)
    blocks = starts_as_arrays(stream)

    assert stream.n_steps == 5 and len(stream) == 6  # 3 blocks for each start in turn
    assert [(first, list(members)) for first, members in blocks] == [
        (0, [0]),
        (2, [0]),
        (4, [0]),
        (0, [1]),
        (2, [1]),
        (4, [1]),
    ]
    for k, start in enumerate([4, 6]):
        expected = list(forecast_members(BatchSampler(), series[: start + 5], start, seeds[k]))
        for member, frames in zip(expected, joined(blocks, k), strict=True):
            np.testing.assert_allclose(frames, member, rtol=1e-6)


def test_start_blocks_batch_a_starts_seeds_where_the_model_can():
    model = BatchSampler()

    stream = StartBlocks(model, np.zeros((9, 2, 3, 3)), [3, 5], 4, [range(3), range(10, 13)], 2, 2)
    sizes = [len(ensemble) for block in stream for ensemble in block.ensembles.values()]

    assert model.batches == [[0, 1], [2], [10, 11], [12]]
    assert sizes == [3] * 4  # 2 starts x 2 blocks, of all 3 members


class BatchDevice(Sampler):
    """A stochastic model that forecasts several contexts per call on the
    device, each with its own seeds: a trajectory is its context's last
    frame plus its seed."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def predict_device_batch(self, contexts, n_steps, block_steps, seeds):
        self.calls.append((contexts[:, -1, 0, 0, 0].tolist(), [list(own) for own in seeds]))
        last = np.stack(
            [
                np.asarray(context[-1], np.float32) + seed
                for context, own in zip(contexts, seeds, strict=True)
                for seed in own
            ]
        )
        for start in range(0, n_steps, block_steps):
            n = min(block_steps, n_steps - start)
            yield DeviceBlock(np.repeat(last[:, None], n, axis=1))


def test_start_blocks_put_several_starts_in_a_call_for_a_model_that_can():
    series = make_series(12)  # frame t is t
    model = BatchDevice()

    seeds = [[0, 1], [10, 11], [20, 21]]

    stream = StartBlocks(model, series, [3, 5, 7], 2, seeds, block_steps=2, batch_size=4)
    blocks = starts_as_arrays(stream)

    # Trajectories (start, seed) in order, 4 per call: starts 3 and 5, then
    # 7, each start's context once with its seeds.
    assert model.calls == [([2, 4], [[0, 1], [10, 11]]), ([6], [[20, 21]])]
    assert [list(members) for _, members in blocks] == [[0, 1], [2]]
    first = blocks[0][1]
    assert [float(m[0, 0, 0, 0]) for m in first[1]] == [14.0, 15.0]  # frame 4, seeds 10 and 11
    assert model.seeds == []
    # Alone (batch_size 1) the same model forecasts a start's members per call.
    alone = starts_as_arrays(StartBlocks(Sampler(), series, [3, 5, 7], 2, seeds, 2))
    for k in range(3):
        np.testing.assert_allclose(joined(blocks, k), joined(alone, k))


def test_start_blocks_hold_no_more_trajectories_per_call_than_the_model_can():
    model = BatchDevice()
    model.max_device_batch = 3  # what a call of it fits on the device

    stream = StartBlocks(model, make_series(12), [3, 5, 7], 2, [[0, 1], [10, 11], [20, 21]], 2, 8)
    blocks = starts_as_arrays(stream)

    assert model.calls == [([2, 4], [[0, 1], [10]]), ([4, 6], [[11], [20, 21]])]
    assert all(len(members) == 2 for _, ensembles in blocks for members in ensembles.values())


def test_start_blocks_give_a_deterministic_batch_no_seeds():
    class Deterministic:
        window = 1

        def __init__(self):
            self.calls = []

        def predict_device_batch(self, contexts, n_steps, block_steps):
            self.calls.append(len(contexts))
            yield DeviceBlock(np.repeat(np.asarray(contexts[:, -1:], np.float32), n_steps, 1))

    model = Deterministic()

    blocks = starts_as_arrays(
        StartBlocks(model, make_series(12), [3, 5, 7], 2, start_seeds(3, 1), 4, 8)
    )

    assert model.calls == [3]  # all three starts in one call, no seeds passed
    assert [float(blocks[0][1][k][0][0, 0, 0, 0]) for k in range(3)] == [2.0, 4.0, 6.0]


def test_start_blocks_stream_together_the_starts_that_share_a_call():
    model = BatchDevice()

    seeds = [[0, 1], [10, 11], [20, 21]]

    stream = StartBlocks(model, make_series(12), [3, 5, 7], 3, seeds, 2, batch_size=3)
    blocks = starts_as_arrays(stream)

    # Calls (3, 3, 5) and (5, 7, 7) share start 5: one wave of 2 blocks.
    assert model.calls == [([2, 4], [[0, 1], [10]]), ([4, 6], [[11], [20, 21]])]
    assert len(stream) == 2
    assert [first for first, _ in blocks] == [0, 2]
    assert all(sorted(members) == [0, 1, 2] for _, members in blocks)
    assert all(len(members[1]) == 2 for _, members in blocks)


def test_start_blocks_check_what_a_batch_gives():
    class Short(BatchDevice):
        def predict_device_batch(self, contexts, n_steps, block_steps, seeds):
            yield from list(super().predict_device_batch(contexts, n_steps, block_steps, seeds))[:1]

    class Long(BatchDevice):
        def predict_device_batch(self, contexts, n_steps, block_steps, seeds):
            yield from super().predict_device_batch(contexts, n_steps + 1, block_steps, seeds)

    series = make_series(12)
    with pytest.raises(ValueError, match="block from lead 3"):
        starts_as_arrays(StartBlocks(Short(), series, [3, 5], 4, [[0], [1]], 2, 2))
    with pytest.raises(ValueError, match="more blocks than the forecast's 2"):
        starts_as_arrays(StartBlocks(Long(), series, [3, 5], 4, [[0], [1]], 2, 2))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"starts": []}, "no starts"),
        ({"seeds": [[]]}, "no seeds"),
        ({"seeds": [[0], [1]]}, "2 lists of seeds for 1 starts"),
        ({"n_steps": 0}, "n_steps"),
        ({"starts": [0, 3], "seeds": [[0], [1]]}, r"starts \[0\] leave no context"),
        ({"block_steps": 0}, "block_steps"),
        ({"batch_size": 0}, "batch_size"),
    ],
)
def test_start_blocks_reject_what_cannot_be_streamed(kwargs, match):
    arguments = {"starts": [3], "n_steps": 2, "seeds": [[0]], **kwargs}
    with pytest.raises(ValueError, match=match):
        StartBlocks(Sampler(), np.zeros((7, 2, 3, 3)), **arguments)
