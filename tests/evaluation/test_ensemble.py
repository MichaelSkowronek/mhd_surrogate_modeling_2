import math

import numpy as np
import pytest

from mhd_surrogate.evaluation.ensemble import EnsembleAccumulator
from mhd_surrogate.evaluation.metrics import rmse_per_step

SHAPE = (6, 2, 40, 30)  # 6 lead times, 2400 values per lead


def scores_of(targets, members, scale=(1.0, 1.0), chunk_t=4):
    accumulator = EnsembleAccumulator(targets, np.array(scale), chunk_t)
    for member in members:
        accumulator.add(member)
    return accumulator.scores()


def test_a_single_member_has_crps_equal_to_its_absolute_error_and_no_spread():
    rng = np.random.default_rng(0)
    targets = rng.normal(size=SHAPE)
    member = targets + rng.normal(size=SHAPE)

    scores = scores_of(targets, [member])

    assert scores.size == 1
    assert scores.crps == pytest.approx(np.abs(member - targets).mean(axis=(1, 2, 3)))
    assert np.all(scores.spread == 0.0)
    assert scores.ensemble_mean_rmse == pytest.approx(rmse_per_step(member, targets))
    assert np.all(np.isnan(scores.spread_skill))


def test_a_calibrated_gaussian_ensemble_scores_the_analytic_crps_and_spread():
    """Truth and members drawn from the same N(0, s^2): expected CRPS is
    s / sqrt(pi), the spread is s and the corrected spread/skill ratio 1."""
    rng = np.random.default_rng(1)
    s, m = 0.7, 6
    shape = (3, 2, 200, 100)
    targets = s * rng.normal(size=shape)
    members = [s * rng.normal(size=shape) for _ in range(m)]

    scores = scores_of(targets, members)

    assert scores.size == m
    assert scores.crps == pytest.approx(np.full(3, s / math.sqrt(math.pi)), rel=0.02)
    assert scores.spread == pytest.approx(np.full(3, s), rel=0.02)
    assert scores.ensemble_mean_rmse == pytest.approx(
        np.full(3, s * math.sqrt(1 + 1 / m)), rel=0.02
    )
    assert scores.spread_skill == pytest.approx(np.ones(3), rel=0.03)


def test_an_overconfident_ensemble_has_a_spread_skill_ratio_below_one():
    rng = np.random.default_rng(2)
    targets = rng.normal(size=SHAPE)
    members = [0.2 * rng.normal(size=SHAPE) for _ in range(4)]

    scores = scores_of(targets, members)

    assert np.all(scores.spread_skill < 0.5)


def test_identical_members_score_like_one():
    rng = np.random.default_rng(3)
    targets = rng.normal(size=SHAPE)
    member = rng.normal(size=SHAPE)

    one = scores_of(targets, [member])
    three = scores_of(targets, [member, member, member])

    assert three.crps == pytest.approx(one.crps)
    assert np.all(three.spread == 0.0)


def test_scores_are_scaled_per_channel_and_do_not_depend_on_the_chunk():
    rng = np.random.default_rng(4)
    targets = rng.normal(size=SHAPE)
    members = [targets + rng.normal(size=SHAPE) for _ in range(3)]
    scaled_targets = targets * np.array([2.0, 5.0])[None, :, None, None]
    scaled_members = [x * np.array([2.0, 5.0])[None, :, None, None] for x in members]

    reference = scores_of(targets, members, chunk_t=1)
    scaled = scores_of(scaled_targets, scaled_members, scale=(2.0, 5.0), chunk_t=4)

    assert scaled.crps == pytest.approx(reference.crps)
    assert scaled.spread == pytest.approx(reference.spread)
    assert scaled.ensemble_mean_rmse == pytest.approx(reference.ensemble_mean_rmse)


def test_a_member_of_the_wrong_shape_and_an_empty_ensemble_are_refused():
    accumulator = EnsembleAccumulator(np.zeros(SHAPE), np.ones(2))

    with pytest.raises(ValueError, match="no members"):
        accumulator.scores()
    with pytest.raises(ValueError, match="shape"):
        accumulator.add(np.zeros((5, 2, 40, 30)))
