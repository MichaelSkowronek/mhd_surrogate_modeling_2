import numpy as np
import pytest

from mhd_surrogate.evaluation.protocol import check_window, forecast, split_context


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
