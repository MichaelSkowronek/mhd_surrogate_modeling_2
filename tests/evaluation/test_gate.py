import math

import pytest

from mhd_surrogate.evaluation.gate import gate_failures, member_group, member_scores

LIMITS = {"energy_rel_error": 0.3, "spectrum_x_lsd": 2.5}
DRIFT = {"u_x_mean_offset": 1.0}
PASSING = {"stable_steps": 837.0, "energy_rel_error": 0.3, "spectrum_x_lsd": 1.0}


def gate(scores, rollout=3000.0, max_scores=LIMITS, **kwargs):
    """A deterministic model's gate: one member, its own scores."""
    return gate_failures(scores, [scores], 837, [rollout], 3000, max_scores, **kwargs)


def test_a_model_meeting_every_check_passes():
    """At the limit is within it."""
    assert gate(PASSING) == []


def test_each_failed_check_is_reported():
    scores = {"stable_steps": 400.0, "energy_rel_error": 0.83, "spectrum_x_lsd": 1.0}

    failures = gate(scores, rollout=1700.0)

    assert failures == [
        "validation forecast stable for 400 of its 837 steps",
        "rollout stable for 1700 of its 3000 steps",
        "energy_rel_error 0.83 over its limit 0.3",
    ]


def test_a_missing_or_undefined_score_fails_its_check():
    """metrics.json leaves NaN scores out; neither may pass by absence."""
    missing = {k: v for k, v in PASSING.items() if k != "spectrum_x_lsd"}
    undefined = {**PASSING, "energy_rel_error": math.nan}

    assert gate(missing) == ["spectrum_x_lsd nan over its limit 2.5"]
    assert gate(undefined) == ["energy_rel_error nan over its limit 0.3"]
    assert gate({}, max_scores={}) == ["validation forecast stable for nan of its 837 steps"]


def test_a_score_under_its_minimum_fails_and_at_the_minimum_passes():
    minimum = {"stable_fraction": 0.8}

    assert gate({**PASSING, "stable_fraction": 0.8}, max_scores={}, min_scores=minimum) == []
    assert gate({**PASSING, "stable_fraction": 0.6}, max_scores={}, min_scores=minimum) == [
        "stable_fraction 0.6 under its limit 0.8"
    ]
    # A checkpoint without one (trained before it existed) fails too.
    assert gate(PASSING, max_scores={}, min_scores=minimum) == [
        "stable_fraction nan under its limit 0.8"
    ]


def test_a_single_members_drift_over_its_limit_fails_like_any_score():
    assert gate({**PASSING, "u_x_mean_offset": 1.3}, member_max_scores=DRIFT) == [
        "u_x_mean_offset 1.3 over its limit 1"
    ]


def ensemble_gate(members, rollout=None, **kwargs):
    """An 8-member ensemble's gate, its median scores passing."""
    rollout = [3000.0] * len(members) if rollout is None else rollout
    return gate_failures(PASSING, members, 837, rollout, 3000, LIMITS, DRIFT, **kwargs)


def members(n_failing, failing, n=8):
    return [
        {**PASSING, "u_x_mean_offset": 0.1, **(failing if k < n_failing else {})} for k in range(n)
    ]


def test_at_least_7_of_8_members_must_be_stable_over_the_validation_forecast():
    unstable = {"stable_steps": 400.0}

    assert ensemble_gate(members(1, unstable)) == []
    assert ensemble_gate(members(2, unstable)) == [
        "6 of 8 members pass every absolute check, under 80% of them: "
        "seed 0: validation forecast stable for 400 of its 837 steps; "
        "seed 1: validation forecast stable for 400 of its 837 steps"
    ]


def test_at_least_7_of_8_members_must_be_stable_over_the_rollout():
    assert ensemble_gate(members(0, {}), rollout=[100.0] + [3000.0] * 7) == []
    assert ensemble_gate(members(0, {}), rollout=[100.0, 0.0] + [3000.0] * 6) == [
        "6 of 8 members pass every absolute check, under 80% of them: "
        "seed 0: rollout stable for 100 of its 3000 steps; "
        "seed 1: rollout stable for 0 of its 3000 steps"
    ]


def test_at_least_7_of_8_members_must_stay_within_the_drift_band():
    drifting = {"u_x_mean_offset": 1.5}

    assert ensemble_gate(members(1, drifting)) == []
    assert ensemble_gate(members(2, drifting)) == [
        "6 of 8 members pass every absolute check, under 80% of them: "
        "seed 0: u_x_mean_offset 1.5 over its limit 1; seed 1: u_x_mean_offset 1.5 over its limit 1"
    ]


def test_a_member_failing_any_absolute_check_is_broken_whichever():
    """One member drifts and another blows up: 6 of 8 usable forecasts."""
    ensemble = members(0, {})
    ensemble[1] = {**ensemble[1], "u_x_mean_offset": 1.5}

    assert ensemble_gate(ensemble) == []
    assert ensemble_gate(ensemble, rollout=[3000.0] * 7 + [0.0]) == [
        "6 of 8 members pass every absolute check, under 80% of them: "
        "seed 1: u_x_mean_offset 1.5 over its limit 1; "
        "seed 7: rollout stable for 0 of its 3000 steps"
    ]
    # Failing several checks, a member is still one broken forecast.
    ensemble[1] = {**ensemble[1], "stable_steps": 400.0}
    assert ensemble_gate(ensemble, rollout=[3000.0, 0.0] + [3000.0] * 6) == []


def test_every_member_needs_its_rollout():
    with pytest.raises(ValueError, match="8 members' validation scores but 7 rollouts"):
        ensemble_gate(members(0, {}), rollout=[3000.0] * 7)


def test_the_relative_limits_are_checked_on_the_median_not_the_members():
    ensemble = members(3, {"energy_rel_error": 9.0})  # 3 of 8 over: the median isn't

    assert ensemble_gate(ensemble) == []
    assert gate_failures(
        {**PASSING, "energy_rel_error": 0.5}, ensemble, 837, [3000.0] * 8, 3000, LIMITS, DRIFT
    ) == ["energy_rel_error 0.5 over its limit 0.3"]


def test_member_scores_read_the_members_groups_in_seed_order():
    metrics = {"val": {"a": 1.0}, member_group("val", 1): {"a": 3.0}, "val.member_0": {"a": 2.0}}

    assert member_scores(metrics) == [{"a": 2.0}, {"a": 3.0}]
    assert member_scores({"val": {"a": 1.0}}) == []
