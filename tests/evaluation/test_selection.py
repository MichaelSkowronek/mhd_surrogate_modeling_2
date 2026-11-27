import pytest

from mhd_surrogate.evaluation.metrics import member_median, selection_score
from mhd_surrogate.evaluation.selection import RunSelection

N_STEPS = 10


def validation(
    usable: int, rollouts: int = 2, skill: float = 1.0, crps: float = 0.5, stable: int | None = None
) -> dict[str, float]:
    """A validation as `selection_scores` gives it: its first `usable`
    rollouts usable, the next ones up to `stable` (default `usable`) stable
    but outside the drift band, the rest blowing up at step 5 (peak 5x)."""
    stable = usable if stable is None else stable
    steps = [float(N_STEPS if k < stable else N_STEPS // 2) for k in range(rollouts)]
    peaks = [1.0 if k < stable else 5.0 for k in range(rollouts)]
    own = selection_score(
        skill,
        crps,
        member_median(steps),
        N_STEPS,
        member_median(peaks),
        usable / rollouts,
        n_forecasts=rollouts,
    )
    return {
        "selection_score": own,
        "multistart_crps_skill_horizon": skill,
        "selection_tie_break": crps,
        "rollout_stable_steps": member_median(steps),
        "rollout_steps": float(N_STEPS),
        "rollouts": float(rollouts),
        "usable_rollouts": float(usable),
        **{f"rollout_{k}_stable_steps": s for k, s in enumerate(steps)},
        **{f"rollout_{k}_peak_ratio": p for k, p in enumerate(peaks)},
    }


def run(pattern: str, neighbors: int = 1, skills=None, threshold: float = 0.8) -> RunSelection:
    """A run of one rollout per validation, usable where `pattern` has an S."""
    skills = skills or [1.0] * len(pattern)
    history = [validation(int(c == "S"), 1, k) for c, k in zip(pattern, skills)]
    return RunSelection(neighbors, history, threshold)


def test_a_window_is_the_nearest_validations_shifted_inward_at_the_ends():
    selection = run("." * 7, neighbors=2)

    assert selection.window(3) == range(1, 6)
    assert selection.window(0) == range(0, 5)
    assert selection.window(1) == range(0, 5)
    assert selection.window(6) == range(2, 7)


def test_a_short_run_judges_every_validation_with_all_of_them():
    selection = run("...", neighbors=2)

    assert [selection.window(i) for i in range(3)] == [range(0, 3)] * 3


def test_a_window_settles_once_its_neighbors_follow_and_it_is_full():
    selection = run("....", neighbors=1)

    assert [selection.settled(i) for i in range(4)] == [True, True, True, False]
    assert not run("..", neighbors=1).settled(0)  # 2 validations: no full window yet


def test_the_usable_fraction_is_the_share_of_the_windows_usable_rollouts():
    selection = run(".S.SSS", neighbors=1)

    assert [selection.usable_fraction(i) for i in range(6)] == pytest.approx(
        [1 / 3, 1 / 3, 2 / 3, 2 / 3, 1.0, 1.0]
    )


def test_a_windows_fraction_pools_its_validations_rollouts():
    selection = RunSelection(1, [validation(k) for k in (2, 2, 0, 1, 2)])

    # Windows of 3 validations, shifted inward at the ends: 6 rollouts each.
    assert [selection.usable_fraction(i) for i in range(5)] == pytest.approx(
        [4 / 6, 4 / 6, 3 / 6, 3 / 6, 3 / 6]
    )


def test_the_score_pools_the_windows_rollouts_but_keeps_the_validations_skill():
    history = [validation(1, 1, 4.0, 0.3), validation(1, 1, 2.0), validation(0, 1)]
    selection = RunSelection(1, history, threshold=0.6)

    # One window for all three: stable steps 10, 10, 5 (median 10), 2 of 3
    # usable, so each is judged stable, by its own skill.
    for index, (skill, crps) in enumerate([(4.0, 0.3), (2.0, 0.5)]):
        assert selection.score(index) == pytest.approx(
            selection_score(skill, crps, N_STEPS, N_STEPS, 1.0, 2 / 3, 0.6, n_forecasts=3)
        )
    # The third's own rollout blew up: it doesn't count as reaching the
    # threshold, but the window's median still judges it stable.
    assert selection.score(2) == pytest.approx(
        selection_score(1.0, 0.5, N_STEPS, N_STEPS, 1.0, 2 / 3, 0.6, 3, own_usable=False)
    )
    assert selection.score(2) > history[2]["selection_score"]


def test_an_epoch_whose_own_rollouts_both_failed_does_not_win_on_its_neighbors():
    skills = (1.0, 3.0, 9.0, 2.0, 1.0)
    history = [validation(k, skill=s) for k, s in zip((2, 2, 0, 2, 2), skills)]
    selection = RunSelection(2, history)

    # One window of 8 of 10 usable rollouts, at the threshold: skill decides
    # between the epochs with a usable rollout of their own, and the best
    # skill, whose own two failed, doesn't count as reaching it.
    assert all(selection.usable_fraction(i) == pytest.approx(0.8) for i in range(5))
    assert selection.best() == 1
    assert selection.score(2) < min(selection.score(i) for i in (0, 1, 3, 4))


def test_an_epoch_whose_own_rollouts_failed_is_kept_if_no_other_reaches_the_threshold():
    usable, skills = (0, 2, 2, 0, 2, 2, 0), (9.0, 9.0, 9.0, 1.0, 9.0, 9.0, 9.0)
    selection = RunSelection(2, [validation(k, skill=s) for k, s in zip(usable, skills)])

    # Only epoch 3's window (1-5) reaches 8 of 10; the others hold 6.
    assert [selection.usable_fraction(i) for i in range(7)] == pytest.approx(
        [0.6, 0.6, 0.6, 0.8, 0.6, 0.6, 0.6]
    )
    assert selection.best() == 3


def test_below_the_threshold_the_windows_pooled_stability_comes_next():
    # No rollout usable; 0 and 1's window holds 4 stable (out of band) of 6,
    # 3's (2-4) 2 of 6.
    history = [validation(0, stable=2), validation(0, stable=2), validation(0), validation(0)]
    selection = RunSelection(1, history)

    assert selection.usable_fraction(0) == selection.usable_fraction(3) == 0.0
    assert selection.score(0) > selection.score(3)
    assert selection.best() == 0


def test_a_usable_stretch_beats_a_lone_usable_validation_with_more_skill():
    selection = run(".S.SSS", neighbors=1, skills=[1, 9, 1, 2, 2, 3])

    assert selection.best() == 5
    assert run(".S.SSS", neighbors=0, skills=[1, 9, 1, 2, 2, 3]).best() == 1


def test_windows_at_the_threshold_rank_by_skill_whatever_their_fraction():
    usable, skills = (2, 2, 2, 1, 2), (1.0, 1.0, 9.0, 1.0, 1.0)
    selection = RunSelection(1, [validation(k, skill=s) for k, s in zip(usable, skills)])

    # Validation 2's window holds 5 of 6 usable, validation 0's all 6: both
    # reach 0.8, so 2's skill decides.
    assert selection.usable_fraction(0) == 1.0
    assert selection.usable_fraction(2) == pytest.approx(5 / 6)
    assert selection.best() == 2
    # With a threshold of 1, validation 2's window falls short and 0's wins.
    strict = RunSelection(1, selection.history, threshold=1.0)
    assert strict.best() == 0


def test_a_pooled_fraction_is_weighted_for_every_rollout_of_a_full_window():
    selection = RunSelection(1, [validation(2, skill=4.0), validation(1), validation(2)])

    # Stable steps 10, 10, 10, 5, 10, 10: the median stable.
    assert selection.score(0) == pytest.approx(
        selection_score(4.0, 0.5, N_STEPS, N_STEPS, 1.0, 5 / 6, 0.8, n_forecasts=6)
    )


def test_judged_alone_a_validation_keeps_its_own_score():
    history = [validation(1), validation(2, skill=3.0)]
    selection = RunSelection(0, history)

    assert [selection.score(i) for i in range(2)] == [h["selection_score"] for h in history]
    assert selection.best() == 1
    # Only the score is needed then.
    assert RunSelection(0, [{"selection_score": 1.0}, {"selection_score": 3.0}]).best() == 1


def test_a_window_needs_the_validations_components():
    selection = RunSelection(1, [{"selection_score": 1.0}, {"selection_score": 3.0}])

    with pytest.raises(ValueError, match="lacks .* to be judged with its window"):
        selection.best()


def test_the_best_so_far_is_provisional_until_its_window_settles():
    selection = run(".S", neighbors=1)
    assert selection.usable_fraction(1) == 0.5

    selection.add(validation(0, 1))

    assert selection.usable_fraction(1) == pytest.approx(1 / 3)


def test_validations_sharing_a_window_are_ordered_by_their_own_scores():
    # All three share the run's one window (unstable pooled median): the
    # one whose own rollout is usable goes first.
    selection = run(".S.", neighbors=1)

    assert selection.score(0) == selection.score(1) == selection.score(2)
    assert selection.best() == 1
    assert selection.rank(1) > selection.rank(0)


def test_a_tie_goes_to_the_first_validation():
    assert run("SSS", neighbors=1).best() == 0


def test_neighbors_and_the_threshold_must_make_sense_and_best_needs_a_validation():
    with pytest.raises(ValueError, match="neighbors"):
        RunSelection(-1)
    for threshold in (0.0, 1.5):
        with pytest.raises(ValueError, match="threshold"):
            RunSelection(1, threshold=threshold)
    with pytest.raises(ValueError, match="no validations"):
        RunSelection(1).best()
