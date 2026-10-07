import math

import numpy as np
import pytest

from mhd_surrogate.evaluation.metrics import rmse_per_step, selection_score, skill_horizon


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


def test_selection_score_ranks_by_skill_then_by_lower_tie_break_rmse():
    ranked = [(7, 0.5), (7, 0.9), (6, 0.0), (6, 5.0), (6, math.inf), (5, 0.0)]
    scores = [selection_score(skill, rmse, 20, 20) for skill, rmse in ranked]

    assert scores == sorted(scores, reverse=True)
    assert scores[2] > scores[3] > scores[4]
    assert scores[4] >= scores[5]  # infinite RMSE ties at worst with one step less skill


def test_selection_score_ranks_stable_steps_above_any_skill_and_rmse():
    n_steps = 20
    # One more stable step beats the most skill and the least RMSE possible.
    ranked = [(20, 0, 9.0), (19, 20, 0.0), (19, 0, 9.0), (5, 3, 0.1), (5, 3, 0.2), (0, 20, 0.0)]
    scores = [selection_score(skill, rmse, stable, n_steps) for stable, skill, rmse in ranked]

    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)
    # Without a skill step of its own, a stable step still wins.
    assert selection_score(0, 1e9, 6, n_steps) > selection_score(n_steps, 0.0, 5, n_steps)


def test_selection_score_of_a_fully_stable_forecast_orders_like_skill_and_rmse():
    stable = selection_score(7, 0.5, 20, 20)

    assert stable == pytest.approx(20 * 21 + 7 - 0.5 / 1.5)


def test_selection_score_of_a_nan_rmse_is_the_worst_possible():
    assert selection_score(10, math.nan, 20, 20) == -math.inf
