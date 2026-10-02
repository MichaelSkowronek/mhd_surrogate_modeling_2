import numpy as np
import pytest

from mhd_surrogate.evaluation.metrics import rmse_per_step, skill_horizon


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
