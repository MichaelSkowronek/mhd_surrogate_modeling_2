import math

from mhd_surrogate.evaluation.gate import gate_failures

LIMITS = {"energy_rel_error": 0.3, "spectrum_x_lsd": 2.5}
PASSING = {"stable_steps": 837.0, "energy_rel_error": 0.3, "spectrum_x_lsd": 1.0}


def test_a_model_meeting_every_check_passes():
    """At the limit is within it."""
    assert gate_failures(PASSING, 837, 3000, 3000, LIMITS) == []


def test_each_failed_check_is_reported():
    scores = {"stable_steps": 400.0, "energy_rel_error": 0.83, "spectrum_x_lsd": 1.0}

    failures = gate_failures(scores, 837, 1700, 3000, LIMITS)

    assert failures == [
        "validation forecast stable for 400 of its 837 steps",
        "rollout stable for 1700 of its 3000 steps",
        "energy_rel_error 0.83 over its limit 0.3",
    ]


def test_a_missing_or_undefined_score_fails_its_check():
    """metrics.json leaves NaN scores out; neither may pass by absence."""
    missing = {k: v for k, v in PASSING.items() if k != "spectrum_x_lsd"}
    undefined = {**PASSING, "energy_rel_error": math.nan}

    assert gate_failures(missing, 837, 3000, 3000, LIMITS) == [
        "spectrum_x_lsd nan over its limit 2.5"
    ]
    assert gate_failures(undefined, 837, 3000, 3000, LIMITS) == [
        "energy_rel_error nan over its limit 0.3"
    ]
    assert gate_failures({}, 837, 3000, 3000, {}) == [
        "validation forecast stable for nan of its 837 steps"
    ]
