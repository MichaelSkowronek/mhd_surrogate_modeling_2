import numpy as np
import pytest

from mhd_surrogate.evaluation.protocol import (
    check_readable,
    check_window,
    ensemble_size,
    forecast,
    forecast_members,
    predict_member,
    split_context,
)


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


def test_forecast_members_rejects_a_series_without_scored_steps():
    with pytest.raises(ValueError, match="context_steps"):
        list(forecast_members(Sampler(), np.zeros((3, 2, 3, 3)), 3, [0]))


def test_ensemble_size_is_one_for_a_deterministic_model():
    assert ensemble_size(Sampler(), 8) == 8
    assert ensemble_size(Persistence(), 8) == 1
    with pytest.raises(ValueError, match="n_members"):
        ensemble_size(Sampler(), 0)
