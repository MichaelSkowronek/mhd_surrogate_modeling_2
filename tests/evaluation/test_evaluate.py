import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from mhd_surrogate.data.climatology import climatology_from_series, domain_mean_series
from mhd_surrogate.evaluation.bootstrap import BlockBootstrap
from mhd_surrogate.evaluation.evaluate import (
    StartSkill,
    candidate_fraction,
    evaluate,
    train_eval_datasets,
)
from mhd_surrogate.evaluation.metrics import CrpsAccumulator, selection_score, skill_horizon
from mhd_surrogate.evaluation.selection import selection_scores, start_ensembles
from mhd_surrogate.evaluation.usable import UsableCheck
from mhd_surrogate.models.baselines import Persistence

DX, DY = 0.5, 0.25


class Oracle:
    """Predicts the true continuation of its context: it is handed the whole
    series and finds where the context ends in it (every frame is unique)."""

    window = 1

    def __init__(self, series):
        self.series = np.asarray(series)

    def start(self, context):
        """The frame the forecast from `context` starts at."""
        (last,) = np.flatnonzero((self.series == context[-1]).all(axis=(1, 2, 3)))
        return last + 1

    def future(self, context, n_steps):
        # A rollout runs past the series: from its end the truth repeats from
        # its start, so its block means stay the truth's.
        start = self.start(context)
        return np.take(self.series, np.arange(start, start + n_steps), axis=0, mode="wrap")

    def predict(self, context, n_steps):
        return self.future(context, n_steps)


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
        stride=5,
        leads=10,
        chunk_t=8,
        nperseg=8,
    )
    return evaluate(model, series, **{**defaults, **kwargs})


def test_a_perfect_forecast_has_zero_error_and_full_skill():
    series = make_series()

    result = run(Oracle(series), series)

    assert result.rmse.shape == (20,)
    assert result.rmse == pytest.approx(np.zeros(20))
    assert result.scores["rmse_mean"] == pytest.approx(0.0)
    assert result.scores["crps_skill_horizon"] == 20
    assert result.scores["energy_rel_error"] == pytest.approx(0.0, abs=1e-12)
    assert result.scores["stable_steps"] == 20
    assert result.scores["energy_peak_ratio"] == pytest.approx(1.0)
    # From starts 10, 15 and 20, 10 leads each: the skill is capped there.
    assert result.multistart.starts == [10, 15, 20]
    assert result.scores["multistart_starts"] == 3
    assert result.scores["multistart_crps_skill_horizon"] == 10
    assert result.scores["multistart_crps_skill_horizon_p05"] == 10
    assert result.scores["multistart_crps_skill_horizon_p95"] == 10
    assert result.scores["multistart_crps_mean"] == pytest.approx(0.0, abs=1e-7)


def test_scores_include_reported_leads_and_skip_leads_past_the_forecast():
    series = make_series()

    result = run(Persistence(), series)

    assert result.scores["rmse_lead_1"] == pytest.approx(result.rmse[0])
    assert result.scores["rmse_lead_5"] == pytest.approx(result.rmse[4])
    assert "rmse_lead_100" not in result.scores
    assert result.scores["multistart_crps_lead_5"] == pytest.approx(result.multistart.curve[4])
    assert "multistart_crps_lead_100" not in result.scores
    assert result.scores["rmse_mean"] == pytest.approx(result.rmse.mean())
    # Independent noise: persistence is off by ~sqrt(2) std at every lead,
    # an absolute error (its CRPS) of 2 / sqrt(pi) ~ 1.13.
    assert result.scores["crps_skill_horizon"] == 0


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


class Offset(Oracle):
    """Predicts the true continuation plus `offset(lead, start)` everywhere."""

    def __init__(self, series, offset):
        super().__init__(series)
        self.offset = offset

    def predict(self, context, n_steps):
        leads = np.arange(1, n_steps + 1, dtype=float)
        offset = self.offset(leads, self.start(context))
        return self.future(context, n_steps) + offset[:, None, None, None]


class Growing(Oracle):
    """Predicts the true continuation scaled by `growth ** lead`: a forecast
    whose energy grows without bound."""

    def __init__(self, series, growth):
        super().__init__(series)
        self.growth = growth

    def predict(self, context, n_steps):
        factor = self.growth ** np.arange(1, n_steps + 1, dtype=float)
        return self.future(context, n_steps) * factor[:, None, None, None]


def usable_check(series, **kwargs):
    """Rollouts of 20 steps in 5-step blocks, judged against the climatology
    of the series' own 10-frame windows (white noise: offsets ~0.3 units)."""
    climatology = climatology_from_series(
        {"series": domain_mean_series(series)}, ["u_x", "u_y"], 10, 5, 5
    )
    defaults = dict(steps=20, block_steps=5, max_ratio=2.0, slow_from=0)
    return UsableCheck(climatology, **{**defaults, **kwargs})


def select(model, series, tie_break_lead=10, check=None, **kwargs):
    defaults = dict(dx=DX, dy=DY, stride=5, leads=10)
    return selection_scores(
        model,
        series,
        10,
        np.array([1.0, 1.0]),
        0.5,
        tie_break_lead,
        check=check or usable_check(series),
        **{**defaults, **kwargs},
    )


def test_selection_scores_match_the_skill_horizon_and_the_tie_break_lead():
    series = make_series()
    # Off by 0.1 * lead everywhere, so the CRPS (the absolute error) is
    # 0.1 * lead, from every start: within the 0.5 threshold up to lead 5.
    model = Offset(series, lambda lead, start: 0.1 * lead)

    scores = select(model, series)

    assert scores["multistart_crps_skill_horizon"] == scores["crps_skill_horizon"] == 5
    assert scores["multistart_crps_lead_10"] == pytest.approx(1.0)
    assert scores["multistart_crps_mean"] == pytest.approx(0.55)
    assert scores["crps_lead_10"] == pytest.approx(1.0)
    # The offset adds o**2 to the energy (~1 for this noise): the rollout's
    # third block's mean (o = 1.1..1.5) is the first over 2x the truth's.
    assert scores["rollout_stable_steps"] == 10
    peak = max(scores["rollout_energy_peak_ratio"], scores["rollout_enstrophy_peak_ratio"])
    assert scores["rollout_peak_ratio"] == peak
    assert scores["selection_score"] == pytest.approx(selection_score(5, 1.0, 10, 20, peak))


def test_selection_scores_rank_by_the_skill_averaged_over_the_starts():
    series = make_series()
    # From the context's end (frame 10) 0.1 * lead off, from frames 15 and
    # 20 0.2 and 0.3 * lead: averaged, 0.2 * lead, within 0.5 up to lead 2.
    model = Offset(series, lambda lead, start: 0.1 * lead * (start - 5) / 5)

    scores = select(model, series)

    assert scores["crps_skill_horizon"] == 5
    assert scores["multistart_crps_skill_horizon"] == 2
    assert scores["multistart_crps_lead_10"] == pytest.approx(2.0)
    expected = selection_score(
        2, 2.0, scores["rollout_stable_steps"], 20, scores["rollout_peak_ratio"]
    )
    assert scores["selection_score"] == pytest.approx(expected)


def test_at_the_cap_selection_scores_break_ties_by_the_mean_crps():
    series = make_series()
    # 0.01 * lead off: within 0.5 over all 10 leads of every start.
    model = Offset(series, lambda lead, start: 0.01 * lead)

    scores = select(model, series)

    assert scores["multistart_crps_skill_horizon"] == 10
    assert scores["rollout_stable_steps"] == 20
    # Its one rollout is usable (0.055 off on average: within the band).
    assert scores["usable_rollouts"] == 1 and scores["selection_tie_break"] == pytest.approx(0.055)
    assert scores["selection_score"] == pytest.approx(
        selection_score(10, 0.055, 20, 20, 1.0, usable_fraction=1.0)
    )


def test_selection_scores_need_the_tie_break_lead_among_the_skill_leads():
    series = make_series()

    with pytest.raises(ValueError, match="tie_break_lead 10 is not one of the 9"):
        select(Oracle(series), series, leads=9)
    with pytest.raises(ValueError, match="no start for 25 leads"):
        select(Oracle(series), series, leads=25)


@pytest.mark.parametrize("chunk_t", [1, 3, 64])
def test_selection_scores_crps_does_not_depend_on_the_chunk_size(chunk_t):
    series = make_series()
    model = Offset(series, lambda lead, start: 0.07 * lead)
    expected = run(model, series).ensemble.crps

    scores = select(model, series, chunk_t=chunk_t)

    assert scores["crps_skill_horizon"] == scores["multistart_crps_skill_horizon"] == 7
    assert scores["crps_lead_10"] == expected[9]


def test_a_blowing_up_forecast_loses_its_stable_steps_and_ranks_below_a_stable_one():
    series = make_series()
    # Energy grows as 1.1 ** (2 * lead): the second 5-step block's mean is
    # over 2x the truth's (white noise, so its block means are all ~1).
    blowing_up = Growing(series, 1.1)
    # Off by 0.6 everywhere: no skill (CRPS over 0.5), but bounded.
    stable = Offset(series, lambda lead, start: 0.6 + 0.0 * lead)

    unstable_scores = select(blowing_up, series)
    stable_scores = select(stable, series)

    assert unstable_scores["rollout_stable_steps"] == 5
    assert unstable_scores["rollout_energy_peak_ratio"] > 2.0
    assert stable_scores["rollout_stable_steps"] == 20
    assert unstable_scores["crps_skill_horizon"] > stable_scores["crps_skill_horizon"] == 0
    assert unstable_scores["selection_score"] < stable_scores["selection_score"]


def test_of_two_forecasts_failing_at_the_same_block_the_lesser_blow_up_ranks_first():
    series = make_series()
    # Both block means first exceed 2x the truth's in the second 5-step block;
    # by the end one is ~6x the truth's energy, the other ~30x.
    slower, faster = (
        select(Growing(series, 1.05), series),
        select(Growing(series, 1.1), series),
    )

    assert slower["rollout_stable_steps"] == faster["rollout_stable_steps"] == 5
    assert slower["rollout_peak_ratio"] < faster["rollout_peak_ratio"]
    assert slower["selection_score"] > faster["selection_score"]


def test_evaluate_and_selection_scores_agree_on_stability():
    series = make_series()
    model = Growing(series, 1.1)

    full = run(model, series).scores
    # A rollout from the context's end as long as the validation forecast.
    monitor = select(model, series)

    for key in ("stable_steps", "energy_peak_ratio", "enstrophy_peak_ratio"):
        assert full[key] == pytest.approx(monitor[f"rollout_{key}"])


class NoisyOracle(Oracle):
    """A stochastic model: the true continuation plus N(0, noise^2) noise
    drawn from `seed`."""

    stochastic = True

    def __init__(self, series, noise):
        super().__init__(series)
        self.noise = noise
        self.seeds = []

    def predict(self, context, n_steps, seed=0):
        self.seeds.append(seed)
        future = self.future(context, n_steps)
        return future + self.noise * np.random.default_rng(seed).normal(size=future.shape)


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


def test_a_stochastic_model_is_scored_on_its_ensemble_and_its_members_median():
    series = make_series()
    model = NoisyOracle(series, noise=0.3)

    result = run(model, series, n_members=4)

    # From the context's end, then from each of the 3 starts, each with its
    # own seeds (protocol.start_seeds); the first start is the context's end.
    assert model.seeds == [0, 1, 2, 3] * 2 + [1000, 1001, 1002, 1003, 2000, 2001, 2002, 2003]
    assert result.scores["ensemble_size"] == 4
    # Each member's own scores, by seed...
    sampler = NoisyOracle(series, noise=0.3)
    rmse = np.stack(
        [
            np.sqrt(
                ((sampler.predict(series[9:10], 20, seed=k) - series[10:]) ** 2).mean(
                    axis=(1, 2, 3)
                )
            )
            for k in range(4)
        ]
    )
    assert [m["rmse_mean"] for m in result.members] == pytest.approx(rmse.mean(axis=1))
    # ...and the scores of a trajectory are their median.
    assert result.rmse == pytest.approx(np.median(rmse, axis=0))
    assert result.scores["rmse_mean"] == pytest.approx(np.median(rmse.mean(axis=1)))
    assert result.scores["energy_rel_error"] == pytest.approx(
        np.median([m["energy_rel_error"] for m in result.members])
    )
    # The skill horizon is the ensemble's, on its CRPS.
    assert result.scores["crps_skill_horizon"] == skill_horizon(result.ensemble.crps, 0.5)
    # The ensemble's spread is the noise level, and it is overconfident
    # (members scatter around the truth, not with it).
    assert result.ensemble.spread == pytest.approx(np.full(20, 0.3), rel=0.25)
    assert "spread_skill_lead_5" in result.scores
    assert result.scores["crps_lead_1"] < result.scores["rmse_lead_1"]


class Seeded:
    """A stochastic model whose member m of every start is `models[m]`'s
    forecast: `protocol.start_seeds` gives it seed 1000 k + m when scored,
    1000 k + 500 + m at a validation."""

    window = 1
    stochastic = True

    def __init__(self, *models):
        self.models = models
        self.seeds = []

    def predict(self, context, n_steps, seed=0):
        self.seeds.append(seed)
        return self.models[seed % 500].predict(context, n_steps)


def test_one_blown_up_member_does_not_decide_an_ensembles_stability():
    series = make_series()
    bounded, blowing_up = Offset(series, lambda lead, start: 0.0 * lead), Growing(series, 1.1)

    result = run(Seeded(bounded, blowing_up, bounded), series, n_members=3)

    assert [m["stable_steps"] for m in result.members] == [20, 5, 20]
    assert result.scores["stable_steps"] == 20
    assert result.scores["energy_peak_ratio"] == pytest.approx(1.0)


class BatchedNoisyOracle(NoisyOracle):
    def __init__(self, *args):
        super().__init__(*args)
        self.batches = []

    def predict_members(self, context, n_steps, seeds):
        self.batches.append(list(seeds))
        return np.stack([NoisyOracle.predict(self, context, n_steps, seed=s) for s in seeds])


def test_batching_members_changes_how_they_are_sampled_not_the_scores():
    series = make_series()
    model = BatchedNoisyOracle(series, 0.3)

    batched = run(model, series, n_members=6, member_batch=4)
    alone = run(NoisyOracle(series, 0.3), series, n_members=6)

    # Seed 0 is the single `forecast`; then each start's members, its own seeds.
    assert model.batches == [[1, 2, 3, 4], [5]] + [
        [0, 1, 2, 3],
        [4, 5],
        [1000, 1001, 1002, 1003],
        [1004, 1005],
        [2000, 2001, 2002, 2003],
        [2004, 2005],
    ]
    assert batched.scores == pytest.approx(alone.scores)
    np.testing.assert_allclose(batched.ensemble.crps, alone.ensemble.crps)
    np.testing.assert_allclose(batched.multistart.crps, alone.multistart.crps)


def test_selection_scores_score_a_stochastic_model_as_an_ensemble_of_its_members():
    series = make_series()
    model = NoisyOracle(series, noise=0.3)

    scores = select(model, series, members=2)

    # Each of the 3 starts with the validations' seeds, none of them the
    # scoring's (protocol.start_seeds); then the rollout from the context's
    # end, past the 2 members (protocol.rollout_seed, validation 0).
    validation = [[500, 501], [1500, 1501], [2500, 2501]]
    assert model.seeds == [seed for seeds in validation for seed in seeds] + [502]
    assert scores["members"] == 2 and scores["rollouts"] == 1
    sampler = NoisyOracle(series, noise=0.3)
    curves = []
    for start, seeds in zip((10, 15, 20), validation):
        crps = CrpsAccumulator(series[start : start + 10], np.array([1.0, 1.0]))
        for seed in seeds:
            crps.add(sampler.predict(series[start - 1 : start], 10, seed=seed))
        curves.append(crps.crps())
    # The first start is the context's end: its own skill, under the old names.
    assert scores["crps_lead_10"] == pytest.approx(curves[0][9])
    assert scores["crps_skill_horizon"] == skill_horizon(curves[0], 0.5)
    averaged = np.mean(curves, axis=0)
    assert scores["multistart_crps_lead_10"] == pytest.approx(averaged[9])
    assert scores["multistart_crps_skill_horizon"] == skill_horizon(averaged, 0.5)


class ByStart:
    """A deterministic model that forecasts as `models[start]` from each
    start frame (`default` from the others)."""

    window = 1

    def __init__(self, default, **models):
        self.default, self.models = default, {int(k[1:]): m for k, m in models.items()}

    def predict(self, context, n_steps):
        model = self.models.get(self.default.start(context), self.default)
        return model.predict(context, n_steps)


def test_selection_scores_take_the_median_of_the_rollouts_stability():
    series = make_series()
    bounded, blowing_up = Oracle(series), Growing(series, 1.1)
    alone = select(blowing_up, series)

    # Rollouts from frames 10 and 15: bounded from 10, blowing up from 15.
    two = select(ByStart(bounded, f15=blowing_up), series, rollout_starts=(0, 5))
    three = select(ByStart(bounded, f15=blowing_up), series, rollout_starts=(0, 5, 10))

    # Two rollouts' median is their mean: an unstable candidate, ranked by
    # its stable steps, then by its (median) peak ratio.
    assert two["rollout_stable_steps"] == 12.5
    assert two["rollouts"] == 2 and two["usable_rollouts"] == 1
    peak = (1.0 + alone["rollout_peak_ratio"]) / 2
    assert two["rollout_peak_ratio"] == pytest.approx(peak, rel=0.1)
    assert two["selection_score"] == pytest.approx(
        selection_score(
            two["multistart_crps_skill_horizon"],
            two["multistart_crps_lead_10"],
            12.5,
            20,
            two["rollout_peak_ratio"],
            usable_fraction=0.5,
            n_forecasts=2,
        )
    )
    # Each rollout's own stable steps and peak ratio, for RunSelection's windows.
    assert [two["rollout_0_stable_steps"], two["rollout_1_stable_steps"]] == [20, 5]
    assert two["rollout_1_peak_ratio"] > 2.0
    # Of three, two stable ones decide.
    assert three["rollout_stable_steps"] == 20 and three["usable_rollouts"] == 2
    assert three["selection_score"] > two["selection_score"]


def test_a_stable_rollout_outside_the_drift_band_is_not_usable():
    series = make_series()
    # 0.6 off everywhere: bounded, but its domain mean ~4 units off.
    shifted = select(Offset(series, lambda lead, start: 0.6 + 0.0 * lead), series)
    truth = select(Oracle(series), series)

    assert shifted["rollout_stable_steps"] == 20 and shifted["in_band_rollouts"] == 0
    assert shifted["usable_rollouts"] == 0 and shifted["rollout_u_x_clim_offset"] > 2
    assert truth["in_band_rollouts"] == truth["usable_rollouts"] == 1


def test_a_validations_rollouts_draw_new_seeds_and_start_at_multistart_starts():
    series = make_series()
    model = NoisyOracle(series, noise=0.3)

    select(model, series, members=2, validation=3, rollout_starts=(0, 10))

    # The skill's members keep their seeds; the rollouts from the 1st and
    # 3rd starts draw validation 3's (protocol.rollout_seed).
    assert model.seeds[:6] == [500, 501, 1500, 1501, 2500, 2501]
    assert model.seeds[6:] == [505, 2505]
    with pytest.raises(ValueError, match="no multi-start start"):
        select(model, series, rollout_starts=(3,))
    with pytest.raises(ValueError, match="no rollout starts"):
        select(model, series, rollout_starts=())


def test_selection_scores_give_a_deterministic_model_one_member_whatever_asked():
    series = make_series()

    scores = select(Persistence(), series, members=4)

    assert scores["members"] == 1


def test_selection_scores_sample_members_in_batches_where_the_model_can():
    series = make_series()
    model = BatchedNoisyOracle(series, 0.3)

    batched = select(model, series, members=4, member_batch=2)

    # Each of the 3 starts' members, then the rollout from the context's end.
    assert model.batches == [
        [500, 501],
        [502, 503],
        [1500, 1501],
        [1502, 1503],
        [2500, 2501],
        [2502, 2503],
        [504],
    ]
    assert batched == pytest.approx(select(NoisyOracle(series, 0.3), series, members=4))


def test_start_ensembles_need_the_truth_for_every_lead():
    series = make_series()

    with pytest.raises(ValueError, match=r"starts \[25\] have no truth for 10 leads in 30"):
        start_ensembles(Oracle(series), series, [10, 25], 10, [[0], [1000]], np.ones(2))


def test_the_final_scoring_averages_the_skill_over_the_starts():
    series = make_series()
    # As in the selection scores: 0.1, 0.2 and 0.3 * lead off from frames
    # 10, 15 and 20, 0.2 * lead on average.
    model = Offset(series, lambda lead, start: 0.1 * lead * (start - 5) / 5)

    result = run(model, series)

    leads = np.arange(1, 11)
    expected = np.stack([0.1 * leads, 0.2 * leads, 0.3 * leads])
    np.testing.assert_allclose(result.multistart.crps, expected, rtol=1e-6)
    np.testing.assert_allclose(result.multistart.curve, 0.2 * leads, rtol=1e-6)
    assert result.scores["crps_skill_horizon"] == 5
    assert result.scores["multistart_crps_skill_horizon"] == 2
    assert result.scores["multistart_crps_lead_5"] == pytest.approx(1.0)
    assert result.scores["multistart_crps_mean"] == pytest.approx(1.1)
    low, high = BlockBootstrap().interval(result.multistart.crps, 0.5)
    assert result.scores["multistart_crps_skill_horizon_p05"] == low >= 1
    assert result.scores["multistart_crps_skill_horizon_p95"] == high <= 5


def test_the_final_scoring_takes_its_bootstrap_and_scores_an_ensemble_from_every_start():
    series = make_series()
    model = NoisyOracle(series, noise=0.3)
    bootstrap = BlockBootstrap(block=1, resamples=50, seed=3)

    result = run(model, series, n_members=4, bootstrap=bootstrap)

    skill = result.multistart
    assert skill.size == 4 and skill.crps.shape == skill.spread.shape == (3, 10)
    assert (
        result.scores["multistart_crps_skill_horizon_p05"] == bootstrap.interval(skill.crps, 0.5)[0]
    )
    assert result.scores["multistart_spread_skill_lead_5"] == pytest.approx(skill.spread_skill[4])
    # Members scatter around the truth, so the ensemble mean is closer to it
    # than calibrated: its RMSE ~0.3 / 2, against a spread of ~0.3.
    assert skill.spread_skill == pytest.approx(np.full(10, np.sqrt(5)), rel=0.25)


def test_a_deterministic_models_multistart_skill_has_no_calibration():
    result = run(Persistence(), make_series(), n_members=8)

    assert result.multistart.size == 1
    assert np.isnan(result.multistart.spread_skill).all()
    assert not any(key.startswith("multistart_spread_skill") for key in result.scores)


def test_start_skill_pools_the_starts_calibration_and_is_json_ready():
    spread = np.array([[1.0, 2.0], [3.0, 2.0]])
    error = np.array([[2.0, 2.0], [2.0, 2.0]])
    skill = StartSkill([5, 9], 3, [[0, 1, 2], [1000, 1001, 1002]], np.ones((2, 2)), spread, error)

    # Pooled: the root of the mean squares, sqrt(5) and 2, over 2.
    expected = np.sqrt(4 / 3) * np.array([np.sqrt(5.0), 2.0]) / 2.0
    np.testing.assert_allclose(skill.spread_skill, expected)
    record = json.loads(json.dumps(skill.as_dict()))
    assert record["starts"] == [5, 9] and record["members"] == 3
    assert record["spread"] == spread.tolist()
    assert record["seeds"] == [[0, 1, 2], [1000, 1001, 1002]]
    assert set(record) == {"starts", "members", "seeds", "crps", "spread", "ensemble_mean_rmse"}


class Trained:
    """An iteratively trained model whose training recorded `fit_info`."""

    iterative = True

    def __init__(self, fit_info):
        self.fit_info = fit_info


def scored(*stable_steps, n_steps=20):
    """An evaluation over `n_steps` whose members are stable for `stable_steps`."""
    return SimpleNamespace(
        rmse=np.zeros(n_steps), members=[{"stable_steps": float(s)} for s in stable_steps]
    )


def test_a_closed_form_models_fraction_is_its_forecasts_stability():
    model = Growing(make_series(), 1.0)

    assert candidate_fraction(model, scored(20)) == ("stable_fraction", 1.0)
    assert candidate_fraction(model, scored(15)) == ("stable_fraction", 0.0)
    # A stochastic one's: the share of its members stable.
    assert candidate_fraction(model, scored(20, 15, 20, 20)) == ("stable_fraction", 0.75)


def test_an_iterative_models_fraction_is_its_training_runs():
    usable = Trained({"usable_fraction": 0.6, "stable_fraction": 0.9})
    assert candidate_fraction(usable, scored(20)) == ("usable_fraction", 0.6)
    # Trained before the rule judged rollouts: the stable fraction it has.
    assert candidate_fraction(Trained({"stable_fraction": 0.7}), scored(20)) == (
        "stable_fraction",
        0.7,
    )
    # From before either: unknown, not its own forecast's.
    assert candidate_fraction(Trained({"best_epoch": 3.0}), scored(20)) is None
