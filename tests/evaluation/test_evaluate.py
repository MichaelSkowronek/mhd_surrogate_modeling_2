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
        block_steps=5,
        max_ratio=2.0,
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
    assert result.scores["stable_steps"] == 20
    assert result.scores["energy_peak_ratio"] == pytest.approx(1.0)


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


class Growing:
    """Predicts the true continuation scaled by `growth ** lead`: a forecast
    whose energy grows without bound."""

    window = 0

    def __init__(self, series, context_steps, growth):
        self.future = series[context_steps:]
        self.growth = growth

    def predict(self, context, n_steps):
        factor = self.growth ** np.arange(1, n_steps + 1, dtype=float)
        return self.future[:n_steps] * factor[:, None, None, None]


def select(model, series, tie_break_lead=10, **kwargs):
    defaults = dict(dx=DX, dy=DY, block_steps=5, max_ratio=2.0)
    return selection_scores(
        model, series, 10, np.array([1.0, 1.0]), 0.5, tie_break_lead, **{**defaults, **kwargs}
    )


def test_selection_scores_match_the_skill_horizon_and_the_tie_break_lead():
    series = make_series()
    # Error 0.1 * lead: within the 0.5 threshold up to lead 5.
    model = Offset(series, 10, lambda lead: 0.1 * lead)

    scores = select(model, series)

    assert scores["skill_horizon"] == 5
    assert scores["rmse_lead_10"] == pytest.approx(1.0)
    # The offset adds o**2 to the energy (~1 for this noise): the third
    # block's mean (o = 1.1..1.5) is the first over 2x the truth's.
    assert scores["stable_steps"] == 10
    assert scores["selection_score"] == pytest.approx(selection_score(5, 1.0, 10, 20))


def test_selection_scores_use_the_last_lead_when_the_forecast_is_shorter():
    series = make_series(n=15)  # 5 scored steps

    scores = select(Offset(series, 10, lambda lead: lead), series)

    assert scores["rmse_lead_10"] == pytest.approx(5.0)


@pytest.mark.parametrize("chunk_t", [1, 3, 64])
def test_selection_scores_rmse_does_not_depend_on_the_chunk_size(chunk_t):
    series = make_series()
    model = Offset(series, 10, lambda lead: 0.07 * lead)
    expected = run(model, series).rmse

    scores = select(model, series, chunk_t=chunk_t)

    assert scores["skill_horizon"] == 7
    assert scores["rmse_lead_10"] == expected[9]


def test_a_blowing_up_forecast_loses_its_stable_steps_and_ranks_below_a_stable_one():
    series = make_series()
    # Energy grows as 1.1 ** (2 * lead): the second 5-step block's mean is
    # over 2x the truth's (white noise, so its block means are all ~1).
    blowing_up = Growing(series, 10, 1.1)
    # Off by 0.6 everywhere: no skill (RMSE over 0.5), but bounded.
    stable = Offset(series, 10, lambda lead: 0.6 + 0.0 * lead)

    unstable_scores = select(blowing_up, series)
    stable_scores = select(stable, series)

    assert unstable_scores["stable_steps"] == 5
    assert unstable_scores["energy_peak_ratio"] > 2.0
    assert stable_scores["stable_steps"] == 20
    assert unstable_scores["skill_horizon"] > stable_scores["skill_horizon"] == 0
    assert unstable_scores["selection_score"] < stable_scores["selection_score"]


def test_evaluate_and_selection_scores_agree_on_stability():
    series = make_series()
    model = Growing(series, 10, 1.1)

    full = run(model, series).scores
    monitor = select(model, series)

    for key in ("stable_steps", "energy_peak_ratio", "enstrophy_peak_ratio"):
        assert full[key] == pytest.approx(monitor[key])


class NoisyOracle:
    """A stochastic model: the true continuation plus N(0, noise^2) noise
    drawn from `seed`."""

    window = 0
    stochastic = True

    def __init__(self, series, context_steps, noise):
        self.future = series[context_steps:]
        self.noise = noise
        self.seeds = []

    def predict(self, context, n_steps, seed=0):
        self.seeds.append(seed)
        rng = np.random.default_rng(seed)
        return self.future[:n_steps] + self.noise * rng.normal(size=self.future[:n_steps].shape)


def test_a_deterministic_model_is_a_one_member_ensemble_whose_crps_is_its_error():
    series = make_series()

    result = run(Persistence(), series, n_members=8)

    assert result.scores["ensemble_size"] == 1
    assert result.ensemble.size == 1
    assert "spread_skill_lead_1" not in result.scores
    targets, prediction = series[10:], np.broadcast_to(series[9], (20, 2, 8, 6))
    expected = np.abs(prediction - targets).mean(axis=(1, 2, 3))
    assert result.scores["crps_lead_5"] == pytest.approx(expected[4])
    assert result.scores["crps_mean"] == pytest.approx(expected.mean())


def test_a_stochastic_model_is_scored_on_its_ensemble_and_its_seed_0_member():
    series = make_series()
    model = NoisyOracle(series, 10, noise=0.3)

    result = run(model, series, n_members=4)

    assert sorted(model.seeds) == [0, 1, 2, 3]
    assert result.scores["ensemble_size"] == 4
    # The pointwise scores are the seed-0 member's...
    member_0 = NoisyOracle(series, 10, noise=0.3).predict(None, 20, seed=0)
    assert result.rmse == pytest.approx(
        np.sqrt(((member_0 - series[10:]) ** 2).mean(axis=(1, 2, 3)))
    )
    # ...the ensemble's spread is the noise level, and it is overconfident
    # (members scatter around the truth, not with it).
    assert result.ensemble.spread == pytest.approx(np.full(20, 0.3), rel=0.25)
    assert "spread_skill_lead_5" in result.scores
    assert result.scores["crps_lead_1"] < result.scores["rmse_lead_1"]


class BatchedNoisyOracle(NoisyOracle):
    def __init__(self, *args):
        super().__init__(*args)
        self.batches = []

    def predict_members(self, context, n_steps, seeds):
        self.batches.append(list(seeds))
        return np.stack([NoisyOracle.predict(self, context, n_steps, seed=s) for s in seeds])


def test_batching_members_changes_how_they_are_sampled_not_the_scores():
    series = make_series()
    model = BatchedNoisyOracle(series, 10, 0.3)

    batched = run(model, series, n_members=6, member_batch=4)
    alone = run(NoisyOracle(series, 10, 0.3), series, n_members=6)

    assert model.batches == [[1, 2, 3, 4], [5]]  # seed 0 is the single `forecast`
    assert batched.scores == pytest.approx(alone.scores)
    np.testing.assert_allclose(batched.ensemble.crps, alone.ensemble.crps)
