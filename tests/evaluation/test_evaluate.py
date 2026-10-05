import numpy as np
import pytest

from mhd_surrogate.evaluation.evaluate import evaluate, train_eval_datasets
from mhd_surrogate.models.baselines import Persistence

DX, DY = 0.5, 0.25


class Oracle:
    """Predicts the true continuation (it is handed the whole series)."""

    window = 0

    def __init__(self, series, context_steps):
        self.future = series[context_steps:]

    def predict(self, context, n_steps):
        return self.future[:n_steps]


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


def test_train_eval_datasets_accepts_a_subset_of_the_train_datasets():
    assert train_eval_datasets(["b"], ["a", "b", "c"]) == ["b"]
    assert train_eval_datasets([], ["a"]) == []


def test_train_eval_datasets_rejects_anything_that_is_not_a_train_dataset():
    with pytest.raises(ValueError, match="test_set"):
        train_eval_datasets(["a", "test_set"], ["a", "b"])
