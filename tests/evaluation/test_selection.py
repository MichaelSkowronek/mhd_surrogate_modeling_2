import pytest

from mhd_surrogate.evaluation.metrics import selection_score
from mhd_surrogate.evaluation.selection import RunSelection

N_STEPS = 10


def scores(stable: bool, skill: float = 1.0, rmse: float = 0.5) -> dict[str, float]:
    """One validation's scores, as `selection_scores` gives them."""
    steps = float(N_STEPS if stable else N_STEPS // 2)
    return {
        "selection_score": selection_score(skill, rmse, steps, N_STEPS, 1.0 if stable else 5.0),
        "stable_steps": steps,
        "forecast_steps": float(N_STEPS),
    }


def run(pattern: str, neighbors: int = 1, skills=None) -> RunSelection:
    """A run whose validations are stable where `pattern` has an S."""
    skills = skills or [1.0] * len(pattern)
    return RunSelection(neighbors, [scores(c == "S", k) for c, k in zip(pattern, skills)])


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


def test_stable_fraction_is_the_share_of_the_window_stable_over_its_whole_forecast():
    selection = run(".S.SSS", neighbors=1)

    assert [selection.stable_fraction(i) for i in range(6)] == pytest.approx(
        [1 / 3, 1 / 3, 2 / 3, 2 / 3, 1.0, 1.0]
    )


def test_the_score_is_the_selection_score_with_the_stable_fraction():
    selection = RunSelection(1, [scores(True, 4.0, 0.3), scores(True, 2.0), scores(False)])

    assert selection.score(0) == pytest.approx(
        selection_score(4.0, 0.3, N_STEPS, N_STEPS, 1.0, stable_fraction=2 / 3, n_forecasts=3)
    )
    # An unstable validation gets no bonus: its forecast's own score.
    assert selection.score(2) == selection.history[2]["selection_score"]


def test_a_stable_stretch_beats_a_lone_stable_validation_with_more_skill():
    selection = run(".S.SSS", neighbors=1, skills=[1, 9, 1, 2, 2, 3])

    assert selection.best() == 5
    assert run(".S.SSS", neighbors=0, skills=[1, 9, 1, 2, 2, 3]).best() == 1


def test_without_neighbors_only_the_selection_score_is_needed():
    selection = RunSelection(0, [{"selection_score": 1.0}, {"selection_score": 3.0}])

    assert selection.best() == 1
    assert selection.score(1) == 3.0
    assert selection.settled(1)


def test_the_best_so_far_is_provisional_until_its_window_settles():
    selection = run(".S", neighbors=1)
    assert selection.best() == 1 and selection.stable_fraction(1) == 0.5

    selection.add(scores(False))

    assert selection.stable_fraction(1) == pytest.approx(1 / 3)


def test_a_tie_goes_to_the_first_validation():
    assert run("SSS", neighbors=1).best() == 0


def test_neighbors_must_not_be_negative_and_best_needs_a_validation():
    with pytest.raises(ValueError, match="neighbors"):
        RunSelection(-1)
    with pytest.raises(ValueError, match="no validations"):
        RunSelection(1).best()
