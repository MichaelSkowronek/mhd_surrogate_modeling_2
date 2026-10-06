import time

import numpy as np
import pytest

from mhd_surrogate.evaluation.evaluate import evaluate, selection_scores, train_eval_datasets
from mhd_surrogate.evaluation.metrics import selection_score
from mhd_surrogate.models.baselines import Persistence

DX, DY = 0.5, 0.25


class Oracle:
    """Predicts the true continuation (it is handed the whole series)."""

    window = 0

    def __init__(self, series, context_steps):
        self.future = series[context_steps:]

    def predict(self, context, n_steps):
        return self.future[:n_steps]


class Slow(Persistence):
    """Persistence that takes `seconds` to predict."""

    def __init__(self, seconds):
        self.seconds = seconds

    def predict(self, context, n_steps):
        time.sleep(self.seconds)
        return super().predict(context, n_steps)


class SlowToRead:
    """A series (like a zarr array) whose every read takes `seconds`."""

    def __init__(self, array, seconds):
        self.array = array
        self.shape = array.shape
        self.seconds = seconds

    def __getitem__(self, index):
        time.sleep(self.seconds)
        return self.array[index]


def make_series(n=30):
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, 2, 8, 6))


def run(model, series, **kwargs):
    defaults = dict(
        context_steps=10,
        dx=DX,
        dy=DY,
        scale=np.array([1.0, 1.0]),
        skill_threshold=0.5,
        report_leads=[1, 5, 100],
        chunk_t=8,
        nperseg=8,
    )
    return evaluate(model, series, **{**defaults, **kwargs})


def test_a_perfect_forecast_has_zero_error_and_full_skill():
    series = make_series()

    result = run(Oracle(series, 10), series)

    assert result.rmse.shape == (20,)
    assert result.rmse == pytest.approx(np.zeros(20))
    assert result.scores["rmse_mean"] == pytest.approx(0.0)
    assert result.scores["skill_horizon"] == 20
    assert result.scores["energy_rel_error"] == pytest.approx(0.0, abs=1e-12)


def test_scores_include_reported_leads_and_skip_leads_past_the_forecast():
    series = make_series()

    result = run(Persistence(), series)

    assert result.scores["rmse_lead_1"] == pytest.approx(result.rmse[0])
    assert result.scores["rmse_lead_5"] == pytest.approx(result.rmse[4])
    assert "rmse_lead_100" not in result.scores
    assert result.scores["rmse_mean"] == pytest.approx(result.rmse.mean())
    # Independent noise: persistence is off by ~sqrt(2) std at every lead.
    assert result.scores["skill_horizon"] == 0


def test_scale_divides_the_error_per_channel():
    series = make_series()

    unscaled = run(Persistence(), series)
    scaled = run(Persistence(), series, scale=np.array([2.0, 2.0]))

    assert scaled.rmse == pytest.approx(unscaled.rmse / 2.0)


def test_predict_seconds_times_the_forecast_and_divides_per_frame():
    series = make_series()

    result = run(Slow(0.05), series)

    assert result.predict_seconds >= 0.05
    assert result.seconds_per_frame == pytest.approx(result.predict_seconds / 20)


def test_predict_seconds_leaves_out_reading_the_series():
    series = SlowToRead(make_series(), seconds=0.2)

    result = run(Persistence(), series)

    assert result.predict_seconds < 0.2


def test_timings_are_not_in_the_reproducible_scores():
    result = run(Persistence(), make_series())

    assert not any("seconds" in key for key in result.scores)


def test_train_eval_datasets_accepts_a_subset_of_the_train_datasets():
    assert train_eval_datasets(["b"], ["a", "b", "c"]) == ["b"]
    assert train_eval_datasets([], ["a"]) == []


def test_train_eval_datasets_rejects_anything_that_is_not_a_train_dataset():
    with pytest.raises(ValueError, match="test_set"):
        train_eval_datasets(["a", "test_set"], ["a", "b"])


class Offset:
    """Predicts the true continuation plus `offset(lead)` everywhere."""

    window = 0

    def __init__(self, series, context_steps, offset):
        self.future = series[context_steps:]
        self.offset = offset

    def predict(self, context, n_steps):
        leads = np.arange(1, n_steps + 1, dtype=float)
        return self.future[:n_steps] + self.offset(leads)[:, None, None, None]


def test_selection_scores_match_the_skill_horizon_and_the_tie_break_lead():
    series = make_series()
    # Error 0.1 * lead: within the 0.5 threshold up to lead 5.
    model = Offset(series, 10, lambda lead: 0.1 * lead)

    scores = selection_scores(model, series, 10, np.array([1.0, 1.0]), 0.5, tie_break_lead=10)

    assert scores["skill_horizon"] == 5
    assert scores["rmse_lead_10"] == pytest.approx(1.0)
    assert scores["selection_score"] == pytest.approx(selection_score(5, 1.0))


def test_selection_scores_use_the_last_lead_when_the_forecast_is_shorter():
    series = make_series(n=15)  # 5 scored steps

    scores = selection_scores(
        Offset(series, 10, lambda lead: lead), series, 10, np.array([1.0, 1.0]), 0.5, 10
    )

    assert scores["rmse_lead_10"] == pytest.approx(5.0)


@pytest.mark.parametrize("chunk_t", [1, 3, 64])
def test_selection_scores_rmse_does_not_depend_on_the_chunk_size(chunk_t):
    series = make_series()
    model = Offset(series, 10, lambda lead: 0.07 * lead)
    expected = run(model, series).rmse

    scores = selection_scores(model, series, 10, np.array([1.0, 1.0]), 0.5, 10, chunk_t=chunk_t)

    assert scores["skill_horizon"] == 7
    assert scores["rmse_lead_10"] == expected[9]
