"""Pointwise forecast-error metrics.

Arrays are (T, C, Nx, Ny): T forecast steps, C velocity channels. The flow is
chaotic, so pointwise error saturates after some lead time for any model;
these metrics are the short-horizon half of the evaluation, and
`diagnostics` is the long-horizon half (does the forecast stay on the
attractor). The CRPS (`CrpsAccumulator`) scores a forecast as an ensemble,
a deterministic one as a single member; the skill horizon and the selection
rule's tie-break are measured on it.
"""

from __future__ import annotations

import math

import numpy as np

from mhd_surrogate.evaluation.quantities import DEFAULT_CHUNK_T

AXES = (1, 2, 3)  # channels and space


def rmse_per_step(
    prediction: np.ndarray, target: np.ndarray, scale: np.ndarray | None = None
) -> np.ndarray:
    """RMSE over channels and space at every lead time, shape (T,).

    `scale` is an optional per-channel (C,) divisor (e.g. the training std),
    so the error is relative to the flow's natural amplitude and channels of
    different magnitude count equally.
    """
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if scale is not None:
        error = error / np.asarray(scale, dtype=np.float64).reshape(1, -1, 1, 1)
    return np.sqrt((error**2).mean(axis=AXES))


class CrpsAccumulator:
    """The CRPS of an ensemble forecast per lead time: feed it the members
    with `add`, then read `crps()`, shape (T,), index 0 = lead time 1.

    CRPS (continuous ranked probability score): E|X - y| - E|X - X'| / 2 for
    members X, X' and truth y, averaged over channels and pixels, each
    channel divided by its `scale` (C,) like `rmse_per_step`. A proper
    score -- it is minimized by forecasting the true distribution -- so
    unlike a member's own error it doesn't reward collapsing toward the
    mean: a realistic member that has lost the truth's phase is penalized
    twice, for missing the truth's features and for placing its own where
    the truth has none, and the second term credits an ensemble whose
    members disagree as much as they miss. For a single member it *is* the
    absolute error, so deterministic and generative models share the score.

    Members of an 837-step validation forecast are ~1 GB each, so they're
    never all held at once: the pairwise term uses the cycle of members,
    the consecutive pairs (X_k, X_{k-1}) and, from three members on, (X_0,
    X_{M-1}) closing it -- M pairs (one for two members), each of two
    independent members, so an unbiased estimate of E|X - X'| as all
    M (M - 1) / 2 pairs would be, at the cost of some variance. Each member
    is in two of them and |X - X'| <= |X - y| + |X' - y|, so like all pairs
    the term never exceeds the first and the estimate is never negative;
    the consecutive pairs alone would be, far below 0 for an ensemble with
    one member blown up. Memory is the targets and three members (the
    first, the last and the one being added); work is done `chunk_t` lead
    times at a time, so float64 temporaries stay small.
    """

    PAIR_TERMS = ("abs",)

    def __init__(self, targets: np.ndarray, scale: np.ndarray, chunk_t: int = DEFAULT_CHUNK_T):
        self.targets = targets
        self.scale = np.asarray(scale, dtype=np.float64).reshape(1, -1, 1, 1)
        self.chunk_t = chunk_t
        self.size = 0
        self.abs_error = np.zeros(len(targets))
        # The pairwise terms' sums over the consecutive pairs.
        self.pair_sums = {name: np.zeros(len(targets)) for name in self.PAIR_TERMS}
        self.first: np.ndarray | None = None
        self.previous: np.ndarray | None = None

    def _chunks(self):
        return range(0, len(self.targets), self.chunk_t)

    def _scaled(self, array: np.ndarray, start: int) -> np.ndarray:
        return np.asarray(array[start : start + self.chunk_t], dtype=np.float64) / self.scale

    def _pair_terms(self, difference: np.ndarray) -> dict[str, np.ndarray]:
        """The `PAIR_TERMS` of two members, per lead time, from their scaled
        `difference` over a chunk of lead times."""
        return {"abs": np.abs(difference).mean(axis=AXES)}

    def _add_pair(self, sums: dict[str, np.ndarray], a: np.ndarray, b: np.ndarray) -> None:
        for t in self._chunks():
            difference = self._scaled(a, t) - self._scaled(b, t)
            leads = slice(t, t + len(difference))
            for name, value in self._pair_terms(difference).items():
                sums[name][leads] += value

    def add(self, member: np.ndarray) -> None:
        member = np.asarray(member)
        if member.shape != self.targets.shape:
            raise ValueError(f"member shape {member.shape}, expected {self.targets.shape}")
        for t in self._chunks():
            x = self._scaled(member, t)
            self.abs_error[t : t + len(x)] += np.abs(x - self._scaled(self.targets, t)).mean(
                axis=AXES
            )
        if self.previous is not None:
            self._add_pair(self.pair_sums, member, self.previous)
        if self.first is None:
            self.first = member
        self.previous = member
        self.size += 1

    def _cycle(self) -> tuple[dict[str, np.ndarray], int]:
        """The pairwise terms' sums over the cycle of members and how many
        pairs it has."""
        if self.size == 0:
            raise ValueError("no members added")
        if self.size < 3:
            return self.pair_sums, max(self.size - 1, 1)
        sums = {name: values.copy() for name, values in self.pair_sums.items()}
        self._add_pair(sums, self.first, self.previous)
        return sums, self.size

    def _crps(self, sums: dict[str, np.ndarray], pairs: int) -> np.ndarray:
        return self.abs_error / self.size - 0.5 * sums["abs"] / pairs

    def crps(self) -> np.ndarray:
        return self._crps(*self._cycle())


def member_median(values) -> np.ndarray | float:
    """The median over an ensemble's members (axis 0) of a score each member
    has, a NaN (a member that blew up) counting as larger than any value --
    the worst, for an error or a peak ratio: the median is NaN only if it
    falls on one. For a single member, its own score."""
    ordered = np.sort(np.asarray(values, dtype=np.float64), axis=0)  # NaN sorts last
    n = len(ordered)
    median = (ordered[(n - 1) // 2] + ordered[n // 2]) / 2
    return float(median) if median.ndim == 0 else median


def skill_horizon(curve: np.ndarray, threshold: float) -> int:
    """Number of leading steps whose error stays at or below `threshold`
    (the whole length if it never exceeds it). A NaN error counts as above
    it: a forecast that blew up has no skill from there on."""
    above = np.flatnonzero(~(np.asarray(curve) <= threshold))
    return int(above[0]) if above.size else len(curve)


def selection_score(
    skill: float,
    crps_at_tie_break_lead: float,
    stable_steps: float,
    n_steps: int,
    peak_ratio: float,
    stable_fraction: float = 0.0,
    n_forecasts: int = 1,
) -> float:
    """One number that ranks candidates like the selection rule (CLAUDE.md,
    "Data analysis"): by stable steps (how long the forecast stays bounded,
    `evaluation.stability`); then a stable forecast by stable fraction, by
    skill horizon and, as a tie-break, by the CRPS at the tie-break lead,
    and an unstable one by its peak ratio, lower first. A higher score is a
    better candidate under the rule; early stopping and the hyperparameter
    search maximize it.

    The CRPS term c / (1 + c) lies in [0, 1) and grows with c, so it orders
    equal skill horizons without ever outweighing one step of skill (an
    infinite CRPS scores 1, at worst tying the next skill down). Skill minus
    that term lies in (-1, n_steps] for a forecast of `n_steps` steps, so
    weighting the stable steps by n_steps + 1 makes one more stable step
    outweigh any skill and CRPS (again tying at worst for an infinite CRPS).
    Scores are therefore comparable only between forecasts of the same
    length, i.e. on the same dataset.

    `peak_ratio` is how far the forecast strayed: its largest block mean of
    energy or enstrophy over the truth's (`stability.stability_scores`'
    peak ratios, the larger of the two; NaN for a forecast that produced
    NaN). An unstable forecast is ranked by it instead of by skill: the
    stable steps of most unstable forecasts are 0 or 100 while their peaks
    range over decades (3x to 60,000x in the U-Net runs), and a forecast
    that strays 3x is far closer to a usable one than one that blows up
    1000x, whatever its short-range skill. Its term (`peak_ratio_term`)
    lies in [0, n_steps], so one more stable step still outweighs it, and
    an unstable forecast never outscores a stable one (tying at worst). A
    stable forecast's peak ratio isn't used: below the truth's range it
    means damping, which the stability check deliberately doesn't reward.

    `stable_fraction` is the share of stable forecasts among the at most
    `n_forecasts` the candidate is judged with (`selection.RunSelection`:
    the members of the validations nearest it in its training run), and
    counts only for a forecast stable over all `n_steps`: it orders
    candidates that are, which the stable steps can't. The default 0 scores
    a forecast alone. Distinct fractions of at most `n_forecasts` forecasts
    differ by at least 1 / n_forecasts^2, and a stable forecast's skill and
    CRPS terms span n_steps + 1, so weighting the fraction by (n_steps + 1)
    n_forecasts^2 makes a larger one outweigh them (tying at worst); a
    stable forecast with no bonus still scores at least any unstable one.
    Scores with a fraction are comparable only for the same `n_forecasts`
    too, and only under the same rule: the score follows the rule as it
    changes (the stable fraction, the peak ratio and the CRPS each changed
    it), so runs scored before a change don't compare with runs after."""
    bonus = stable_fraction_bonus(stable_fraction, stable_steps, n_steps, n_forecasts)
    c = crps_at_tie_break_lead
    if math.isnan(c):
        return float("-inf")
    if stable_steps < n_steps:
        return float(stable_steps) * (n_steps + 1) + peak_ratio_term(peak_ratio, n_steps)
    return (
        float(stable_steps) * (n_steps + 1)
        + bonus
        + float(skill)
        - (1.0 if math.isinf(c) else c / (1.0 + c))
    )


def peak_ratio_term(peak_ratio: float, n_steps: int) -> float:
    """What `selection_score` adds for an unstable forecast's peak ratio:
    n_steps / (1 + log10 peak_ratio), in [0, n_steps] and decreasing as the
    peak grows (a peak ratio at most 1 counts as 1); 0 for a NaN or
    infinite one, the worst. Only its order matters, not its shape."""
    if not math.isfinite(peak_ratio):
        return 0.0
    return n_steps / (1.0 + math.log10(max(peak_ratio, 1.0)))


def stable_fraction_bonus(
    stable_fraction: float, stable_steps: float, n_steps: int, n_forecasts: int
) -> float:
    """What `selection_score` adds for a stable fraction: the fraction times
    (n_steps + 1) n_forecasts^2 for a forecast stable over all `n_steps`,
    0 for one that isn't."""
    if not 0.0 <= stable_fraction <= 1.0:
        raise ValueError(f"stable_fraction must be in [0, 1], got {stable_fraction}")
    if n_forecasts < 1:
        raise ValueError(f"n_forecasts must be >= 1, got {n_forecasts}")
    if stable_steps < n_steps:
        return 0.0
    return stable_fraction * (n_steps + 1) * n_forecasts**2
