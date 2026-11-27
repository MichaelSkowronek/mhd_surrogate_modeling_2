import numpy as np
import pytest

from mhd_surrogate.data.climatology import Climatology, domain_mean_series
from mhd_surrogate.evaluation.quantities import to_device
from mhd_surrogate.evaluation.stability import energy_and_enstrophy
from mhd_surrogate.evaluation.usable import (
    DeviceRolloutQuantities,
    UsableCheck,
    band_scores,
    covering_windows,
    in_band,
    rollout_scores,
    slow_std_ratios,
)

DX, DY = 0.5, 0.25
CLIMATOLOGY = Climatology(
    channel_names=["u_x", "u_y"],
    window_steps=10,
    stride=5,
    block_steps=5,
    level=[1.0, 0.0],
    unit=[0.1, 0.2],
    slow_std=[0.05, 0.1],
    max_window_offset=[0.3, 0.1],
    std_ratio_range=[(0.9, 1.1), (0.95, 1.05)],
    n_windows=4,
    source_datasets=["a"],
)
CHECK = UsableCheck(CLIMATOLOGY, steps=20, block_steps=5, max_ratio=2.0, slow_from=0)


def bulk_flow(n, offset=(0.0, 0.0), amplitude=(0.1, 0.2)):
    """A (n, 2) domain-mean series: the climatology's level plus `offset`
    units and an alternating +-`amplitude` (std over time = amplitude)."""
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)[:, None]
    return np.array([1.0, 0.0]) + np.array(offset) * [0.1, 0.2] + signs * np.array(amplitude)


def test_band_scores_are_the_offset_and_std_ratio_in_the_climatologys_units():
    scores = band_scores(bulk_flow(10, offset=(0.5, -0.25), amplitude=(0.15, 0.2)), CLIMATOLOGY)

    assert scores == pytest.approx(
        {
            "u_x_clim_offset": 0.5,
            "u_x_clim_std_ratio": 1.5,
            "u_y_clim_offset": 0.25,
            "u_y_clim_std_ratio": 1.0,
        }
    )


@pytest.mark.parametrize(
    "offset, amplitude, inside",
    [
        ((0.0, 0.0), (0.1, 0.2), True),
        ((0.99, -0.99), (0.199, 0.398), True),  # just inside the limits
        ((1.2, 0.0), (0.1, 0.2), False),
        ((0.0, -1.2), (0.1, 0.2), False),
        ((0.0, 0.0), (0.25, 0.2), False),
    ],
)
def test_in_band_needs_every_channel_within_both_limits(offset, amplitude, inside):
    assert in_band(band_scores(bulk_flow(10, offset, amplitude), CLIMATOLOGY), CHECK) is inside


def test_a_rollout_that_blew_up_is_outside_the_band():
    flow = bulk_flow(10)
    flow[3, 0] = np.nan

    assert not in_band(band_scores(flow, CLIMATOLOGY), CHECK)


@pytest.mark.parametrize(
    "n_steps, window, starts",
    [(3000, 837, [0, 837, 1674, 2163]), (20, 10, [0, 10]), (25, 10, [0, 10, 15]), (10, 10, [0])],
)
def test_covering_windows_cover_the_rollout_the_last_ending_with_it(n_steps, window, starts):
    assert covering_windows(n_steps, window) == starts


def test_covering_windows_need_a_window_that_fits():
    with pytest.raises(ValueError, match="no 11-step window fits 10"):
        covering_windows(10, 11)


def test_the_slow_ratio_compares_the_windows_block_means_spread_with_the_real_ones():
    # Block means alternating +-0.1 around the level: their std 0.1 within
    # every 10-step window, twice the climatology's 0.05 on u_x.
    blocks = np.repeat(np.where(np.arange(6) % 2 == 0, 0.1, -0.1), 5)
    flow = np.stack([1.0 + blocks, np.zeros(30)], axis=1)

    ratios = slow_std_ratios(flow, CLIMATOLOGY, first=0)

    assert ratios["u_x_slow_std_ratio"] == pytest.approx(2.0)
    assert ratios["u_y_slow_std_ratio"] == 0.0
    assert np.isnan(slow_std_ratios(flow, CLIMATOLOGY, first=25)["u_x_slow_std_ratio"])


def rollout(energy=None, flow=None, n=20):
    energy = np.ones(n) if energy is None else energy
    return {
        "energy": energy,
        "enstrophy": np.ones(n),
        "mean_velocity": bulk_flow(n) if flow is None else flow,
    }


TRUTH = {"energy": np.ones(20), "enstrophy": np.ones(20)}


def test_a_stable_rollout_in_the_band_is_usable():
    scores = rollout_scores(rollout(), TRUTH, CHECK)

    assert scores["stable_steps"] == 20 and scores["energy_peak_ratio"] == 1.0
    assert scores["in_band"] == scores["usable"] == scores["windows_in_band"] == 1.0
    assert scores["u_x_clim_offset"] == pytest.approx(0.0, abs=1e-12)
    assert scores["energy_rel_error"] == scores["enstrophy_rel_error"] == 0.0


def test_a_rollout_blowing_up_late_is_not_usable_though_its_first_window_is_in_band():
    energy = np.ones(20)
    energy[15:] = 3.0  # the 4th 5-step block over 2x the truth's

    scores = rollout_scores(rollout(energy), TRUTH, CHECK)

    assert scores["stable_steps"] == 15 and scores["energy_peak_ratio"] == 3.0
    assert scores["in_band"] == 1.0 and scores["usable"] == 0.0


def test_a_stable_rollout_whose_bulk_flow_drifts_away_is_not_usable():
    flow = bulk_flow(20, offset=(1.5, 0.0))

    scores = rollout_scores(rollout(flow=flow), TRUTH, CHECK)

    assert scores["stable_steps"] == 20
    assert scores["u_x_clim_offset"] == pytest.approx(1.5)
    assert scores["in_band"] == scores["usable"] == 0.0


def test_later_windows_are_scored_but_only_the_first_decides():
    flow = bulk_flow(20)
    flow[10:, 0] += 0.15  # the second window 1.5 units off

    scores = rollout_scores(rollout(flow=flow), TRUTH, CHECK)

    assert scores["usable"] == 1.0
    assert scores["windows_in_band"] == 0.5
    assert scores["u_x_largest_clim_offset"] == pytest.approx(1.5)


def test_rollout_scores_need_the_checks_length():
    with pytest.raises(ValueError, match="a rollout of 19 steps, expected 20"):
        rollout_scores(rollout(n=19), TRUTH, CHECK)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"steps": 9}, "shorter than the band's 10-step window"),
        ({"max_offset": 0.0}, "max_offset"),
        ({"block_steps": 0}, "block_steps"),
        ({"slow_from": -1}, "slow_from"),
    ],
)
def test_a_usable_check_refuses_impossible_settings(overrides, match):
    with pytest.raises(ValueError, match=match):
        UsableCheck(CLIMATOLOGY, **{"steps": 20, **overrides})


def test_the_device_reduction_matches_the_numpy_references():
    frames = np.random.default_rng(0).normal(1.0, 0.3, size=(13, 2, 8, 6)).astype(np.float32)
    quantities = DeviceRolloutQuantities(DX, DY)

    for start in range(0, 13, 4):
        quantities.add(to_device(frames[start : start + 4]))
    series = quantities.series()

    reference = energy_and_enstrophy(frames, DX, DY)
    np.testing.assert_allclose(series["energy"], reference["energy"], rtol=1e-5)
    np.testing.assert_allclose(series["enstrophy"], reference["enstrophy"], rtol=1e-5)
    np.testing.assert_allclose(series["mean_velocity"], domain_mean_series(frames), atol=1e-6)
    with pytest.raises(ValueError, match="no frames added"):
        DeviceRolloutQuantities(DX, DY).series()
