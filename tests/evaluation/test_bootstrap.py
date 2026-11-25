import numpy as np
import pytest

from mhd_surrogate.evaluation.bootstrap import BlockBootstrap
from mhd_surrogate.evaluation.metrics import skill_horizon


def test_a_resample_is_circular_runs_of_consecutive_starts():
    indices = BlockBootstrap(block=3, resamples=200, seed=1).indices(7)

    assert indices.shape == (200, 7)
    # Runs of 3 consecutive starts, wrapping from the last to the first
    # (ceil(7 / 3) = 3 runs, the last cut short).
    for run in (indices[:, 0:3], indices[:, 3:6]):
        np.testing.assert_array_equal((run[:, 1:] - run[:, :-1]) % 7, 1)
    assert set(indices.ravel()) == set(range(7))


def test_resamples_reproduce_from_the_seed():
    a, b, c = (BlockBootstrap(seed=seed).indices(16) for seed in (0, 0, 1))

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_identical_starts_give_an_interval_of_their_horizon():
    curves = np.tile(np.linspace(0.1, 0.6, 11), (16, 1))  # over 0.32 from lead 6

    assert BlockBootstrap().interval(curves, 0.32) == (5.0, 5.0)


def test_the_interval_spans_the_starts_horizons():
    rng = np.random.default_rng(0)
    slopes = rng.uniform(0.01, 0.05, size=16)
    curves = slopes[:, None] * np.arange(1, 51)  # each start's own slope

    low, high = BlockBootstrap().interval(curves, 0.32)

    point = skill_horizon(curves.mean(axis=0), 0.32)
    alone = [skill_horizon(c, 0.32) for c in curves]
    assert min(alone) < low <= point <= high < max(alone)


def test_averaged_curves_are_the_resampled_starts_means():
    bootstrap = BlockBootstrap(block=2, resamples=5)
    curves = np.arange(12.0).reshape(4, 3)

    averaged = bootstrap.averaged(curves)

    np.testing.assert_allclose(averaged, curves[bootstrap.indices(4)].mean(axis=1))


def test_a_clearly_better_model_ranks_first_in_every_resample():
    leads = np.arange(1, 51)
    better = np.tile(0.01 * leads, (8, 1))  # horizon 32
    worse = np.tile(0.02 * leads, (8, 1))  # horizon 16

    comparison = BlockBootstrap(resamples=100).compare(better, worse, 0.32, 10)

    assert comparison.a_first == 1.0 and comparison.b_first == 0.0
    assert comparison.difference == (16.0, 16.0)


def test_equal_horizons_are_ranked_by_the_tie_break_and_identical_curves_tie():
    leads = np.arange(1, 21, dtype=float)
    a = np.tile(np.where(leads < 6, 0.1, 0.5), (4, 1))
    b = np.tile(np.where(leads < 6, 0.2, 0.5), (4, 1))
    b[:, 9] = 0.4  # same horizon (5), a lower CRPS at lead 10 for b

    assert BlockBootstrap(resamples=20).compare(a, b, 0.32, 10).b_first == 1.0
    same = BlockBootstrap(resamples=20).compare(a, a, 0.32, 10)
    assert same.a_first == same.b_first == 0.0


def test_a_blown_up_forecast_ranks_last_and_curves_must_match():
    good = np.tile(np.full(20, 0.5), (4, 1))  # no skill, but finite
    blown = good.copy()
    blown[:, 9] = np.nan

    comparison = BlockBootstrap(resamples=20).compare(blown, good, 0.32, 10)

    assert comparison.b_first == 1.0
    with pytest.raises(ValueError, match="not the same starts"):
        BlockBootstrap().compare(good, good[:3], 0.32, 10)


@pytest.mark.parametrize("kwargs", [{"block": 0}, {"resamples": 0}])
def test_a_bootstrap_needs_a_block_and_resamples(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        BlockBootstrap(**kwargs)
    with pytest.raises(ValueError, match="no starts"):
        BlockBootstrap().indices(0)
