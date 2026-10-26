import math

import numpy as np
import pytest

from mhd_surrogate.evaluation.metrics import (
    peak_ratio_term,
    rmse_per_step,
    selection_score,
    skill_horizon,
)


def test_rmse_per_step_of_a_constant_offset_is_that_offset_at_every_step():
    target = np.zeros((5, 2, 3, 4))
    prediction = target + 2.0

    assert rmse_per_step(prediction, target) == pytest.approx(np.full(5, 2.0))


def test_rmse_per_step_is_computed_separately_for_each_lead_time():
    target = np.zeros((3, 2, 3, 4))
    prediction = target.copy()
    prediction[1] += 1.0
    prediction[2] += 3.0

    assert rmse_per_step(prediction, target) == pytest.approx([0.0, 1.0, 3.0])


def test_rmse_per_step_scale_divides_each_channel_by_its_own_amplitude():
    target = np.zeros((1, 2, 3, 4))
    prediction = target.copy()
    prediction[:, 0] = 2.0  # channel 0 off by 2, scale 2 -> 1
    prediction[:, 1] = 6.0  # channel 1 off by 6, scale 3 -> 2

    got = rmse_per_step(prediction, target, scale=np.array([2.0, 3.0]))

    assert got == pytest.approx([np.sqrt((1.0**2 + 2.0**2) / 2)])


def test_skill_horizon_counts_steps_before_the_threshold_is_first_exceeded():
    curve = np.array([0.1, 0.2, 0.5, 0.3, 0.9])

    assert skill_horizon(curve, 0.4) == 2
    assert skill_horizon(curve, 0.5) == 4  # equal to the threshold still counts as skill
    assert skill_horizon(curve, 1.0) == 5  # never exceeded: the whole length
    assert skill_horizon(curve, 0.0) == 0


def test_skill_horizon_treats_a_nan_error_as_no_skill():
    """A forecast that blew up must not score the whole length."""
    assert skill_horizon(np.array([0.1, 0.2, np.nan, 0.1]), 0.5) == 2
    assert skill_horizon(np.array([np.nan, np.nan]), 0.5) == 0


# A stable forecast's peak ratio (unused) and an unstable one's.
WITHIN, BLOWN_UP = 0.9, 50.0


def test_selection_score_ranks_by_skill_then_by_lower_tie_break_rmse():
    ranked = [(7, 0.5), (7, 0.9), (6, 0.0), (6, 5.0), (6, math.inf), (5, 0.0)]
    scores = [selection_score(skill, rmse, 20, 20, WITHIN) for skill, rmse in ranked]

    assert scores == sorted(scores, reverse=True)
    assert scores[2] > scores[3] > scores[4]
    assert scores[4] >= scores[5]  # infinite RMSE ties at worst with one step less skill


def test_selection_score_ranks_stable_steps_above_any_peak_skill_and_rmse():
    n_steps = 20
    # One more stable step beats the lowest peak, the most skill and the
    # least RMSE possible.
    ranked = [
        (20, 0, 9.0, 1e6),
        (19, 20, 0.0, 1.0),
        (19, 0, 9.0, 1e6),
        (5, 3, 0.1, 2.5),
        (0, 20, 0.0, 2.5),
    ]
    scores = [
        selection_score(skill, rmse, stable, n_steps, peak) for stable, skill, rmse, peak in ranked
    ]

    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)
    assert selection_score(0, 1e9, 6, n_steps, math.nan) > selection_score(
        n_steps, 0.0, 5, n_steps, 1.0
    )


def test_selection_score_ranks_unstable_forecasts_by_their_peak_ratio_not_skill():
    def score(skill, peak):
        return selection_score(skill, 0.5, 100, 837, peak)

    ranked = [score(0, 2.1), score(20, 3.0), score(20, 1000.0), score(20, math.inf)]

    assert ranked == sorted(ranked, reverse=True)
    assert len(set(ranked)) == len(ranked)
    assert score(0, math.nan) == score(20, math.inf)  # blew up to NaN: the worst peak


def test_peak_ratio_term_decreases_from_n_steps_to_zero():
    terms = [peak_ratio_term(p, 20) for p in (0.5, 1.0, 2.0, 1e3, 1e30, math.inf, math.nan)]

    assert terms[0] == terms[1] == 20  # at most 1 counts as 1
    assert terms[1] > terms[2] > terms[3] > terms[4] > 0
    assert terms[5] == terms[6] == 0


def test_selection_score_of_a_fully_stable_forecast_orders_like_skill_and_rmse():
    stable = selection_score(7, 0.5, 20, 20, WITHIN)

    assert stable == pytest.approx(20 * 21 + 7 - 0.5 / 1.5)
    assert selection_score(7, 0.5, 20, 20, 1.9) == stable  # its peak isn't used


def test_selection_score_of_a_nan_rmse_is_the_worst_possible():
    assert selection_score(10, math.nan, 20, 20, WITHIN) == -math.inf


def test_selection_score_ranks_a_stable_forecasts_fraction_above_any_skill_and_rmse():
    n_steps, m = 20, 5

    def score(fraction, skill, rmse):
        return selection_score(skill, rmse, n_steps, n_steps, WITHIN, fraction, n_forecasts=m)

    # Fractions of up to 5 forecasts: 2/3 and 3/5 are the closest distinct pair.
    ranked = [(1.0, 0, 9.0), (0.8, 20, 0.0), (0.8, 3, 0.1), (2 / 3, 0, 9.0), (3 / 5, 20, 0.0)]
    scores = [score(*candidate) for candidate in ranked]

    assert scores == sorted(scores, reverse=True)
    assert score(2 / 3, 0, math.inf) >= score(3 / 5, n_steps, 0.0)  # ties at worst


def test_selection_score_ignores_the_fraction_of_an_unstable_forecast():
    with_fraction = selection_score(7, 0.5, 19, 20, BLOWN_UP, stable_fraction=1.0, n_forecasts=5)

    assert with_fraction == selection_score(7, 0.5, 19, 20, BLOWN_UP)
    # A stable forecast without a bonus still scores at least any unstable one.
    assert selection_score(0, math.inf, 20, 20, WITHIN) >= selection_score(20, 0.0, 19, 20, 1.0)


@pytest.mark.parametrize("fraction, n_forecasts", [(-0.1, 1), (1.1, 1), (0.5, 0)])
def test_selection_score_rejects_an_impossible_fraction_or_count(fraction, n_forecasts):
    with pytest.raises(ValueError):
        selection_score(7, 0.5, 20, 20, WITHIN, stable_fraction=fraction, n_forecasts=n_forecasts)
