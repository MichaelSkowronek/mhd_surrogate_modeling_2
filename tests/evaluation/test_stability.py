import numpy as np
import pytest

from mhd_surrogate.evaluation.diagnostics import enstrophy, kinetic_energy
from mhd_surrogate.evaluation.stability import (
    block_means,
    energy_and_enstrophy,
    stability_scores,
)

DX, DY = 0.5, 0.25


def series_of(energy, enstrophy=None):
    """Per-step quantities as `energy_and_enstrophy` returns them."""
    energy = np.asarray(energy, dtype=float)
    return {"energy": energy, "enstrophy": energy if enstrophy is None else np.asarray(enstrophy)}


def test_block_means_average_consecutive_blocks_and_keep_a_short_remainder():
    values = np.array([1.0, 3.0, 5.0, 7.0, 10.0])

    assert block_means(values, 2) == pytest.approx([2.0, 6.0, 10.0])
    assert block_means(values, 5) == pytest.approx([5.2])
    assert block_means(values, 9) == pytest.approx([5.2])


def test_block_means_reject_an_empty_block():
    with pytest.raises(ValueError, match="block_steps"):
        block_means(np.ones(3), 0)


def test_energy_and_enstrophy_match_the_diagnostics_and_do_not_depend_on_the_chunk():
    rng = np.random.default_rng(0)
    series = rng.normal(size=(7, 2, 6, 5)).astype(np.float32)
    expected = kinetic_energy(series.astype(np.float64))

    for chunk_t in (1, 3, 7):
        got = energy_and_enstrophy(series, DX, DY, chunk_t)

        assert got["energy"] == pytest.approx(expected)
        assert got["enstrophy"] == pytest.approx(enstrophy(series.astype(np.float64), DX, DY))


def test_a_forecast_within_the_truths_range_is_stable_for_its_whole_length():
    true = series_of(np.linspace(1.0, 2.0, 10))
    predicted = series_of(np.full(10, 1.5))

    scores = stability_scores(predicted, true, block_steps=4, max_ratio=2.0)

    assert scores["stable_steps"] == 10
    # Truth's largest block mean is the 2-step remainder, (1.889 + 2) / 2.
    assert scores["energy_peak_ratio"] == pytest.approx(1.5 / (17.0 / 9 + 2.0) * 2)


def test_stable_steps_is_the_start_of_the_first_block_over_the_threshold():
    true = series_of(np.ones(10))
    # Block means (4 steps): 1, 2.5 (over 2x), 1: the second block fails.
    predicted = series_of([1, 1, 1, 1, 1, 4, 4, 1, 1, 1])

    scores = stability_scores(predicted, true, block_steps=4, max_ratio=2.0)

    assert scores["stable_steps"] == 4
    assert scores["energy_peak_ratio"] == pytest.approx(2.5)


def test_a_block_exactly_at_the_threshold_still_counts_as_stable():
    true = series_of(np.ones(8))

    scores = stability_scores(series_of(np.full(8, 2.0)), true, block_steps=4, max_ratio=2.0)

    assert scores["stable_steps"] == 8


def test_either_quantity_fails_a_block():
    true = series_of(np.ones(8), np.ones(8))
    # Energy stays put; enstrophy blows up in the second block only.
    predicted = series_of(np.ones(8), [1, 1, 1, 1, 9, 9, 9, 9])

    scores = stability_scores(predicted, true, block_steps=4, max_ratio=2.0)

    assert scores["stable_steps"] == 4
    assert scores["energy_peak_ratio"] == pytest.approx(1.0)
    assert scores["enstrophy_peak_ratio"] == pytest.approx(9.0)


def test_damping_is_not_instability():
    """One-sided: a forecast decaying to rest is bounded (the guardrails
    judge it, not this check)."""
    true = series_of(np.ones(8))

    scores = stability_scores(series_of(np.zeros(8)), true, block_steps=4, max_ratio=2.0)

    assert scores["stable_steps"] == 8
    assert scores["energy_peak_ratio"] == 0.0


def test_a_nan_forecast_fails_from_its_first_nan_block():
    true = series_of(np.ones(12))
    predicted = series_of([1, 1, 1, 1, 1, 1, 1, 1, np.nan, np.nan, np.nan, np.nan])

    scores = stability_scores(predicted, true, block_steps=4, max_ratio=2.0)

    assert scores["stable_steps"] == 8
    assert np.isnan(scores["energy_peak_ratio"])


def test_a_forecast_failing_its_first_block_has_no_stable_steps():
    true = series_of(np.ones(8))

    scores = stability_scores(series_of(np.full(8, 3.0)), true, block_steps=4, max_ratio=2.0)

    assert scores["stable_steps"] == 0


def test_series_of_different_lengths_are_rejected():
    with pytest.raises(ValueError, match="steps"):
        stability_scores(series_of(np.ones(7)), series_of(np.ones(8)), 4, 2.0)
