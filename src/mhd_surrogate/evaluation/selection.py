"""The selection rule's scores, cheap enough to run every epoch.

Early stopping and the hyperparameter search score a model in training by
`selection_scores`, and choose between a run's validations with
`RunSelection`; `evaluate.evaluate` scores the final model in full. In a
module of its own (with `metrics`, `protocol`, `stability` and
`quantities`), so that training -- and the DVC `train` stage's
dependencies -- reach only this part of the evaluation code: a change to
the diagnostics or the ensemble's spread re-runs the `evaluate` stage on the
stored checkpoint instead of retraining the model.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from mhd_surrogate.evaluation.metrics import (
    CrpsAccumulator,
    member_median,
    selection_score,
    skill_horizon,
    stable_fraction_bonus,
)
from mhd_surrogate.evaluation.protocol import (
    ForecastModel,
    ensemble_size,
    forecast_members,
    split_context,
)
from mhd_surrogate.evaluation.quantities import DEFAULT_CHUNK_T
from mhd_surrogate.evaluation.stability import (
    QUANTITIES,
    energy_and_enstrophy,
    stability_scores,
)


def selection_scores(
    model: ForecastModel,
    series,
    context_steps: int,
    scale: np.ndarray,
    skill_threshold: float,
    tie_break_lead: int,
    dx: float,
    dy: float,
    block_steps: int,
    max_ratio: float,
    members: int = 1,
    member_batch: int = 1,
    chunk_t: int = DEFAULT_CHUNK_T,
) -> dict[str, float]:
    """The scores the selection rule ranks by, without the physics
    guardrails: `crps_skill_horizon` and `crps_lead_<tie_break_lead>`, the
    stability scores (`stable_steps` and the peak ratios), their
    combination `selection_score` (of this forecast alone; `RunSelection`
    adds the stable fraction) and the forecast's length, `forecast_steps`.
    Cheap enough to run during training (early stopping, the
    hyperparameter search; the stability check adds ~3 s to an 837-step
    validation forecast); `evaluate` scores the final model in full.

    A stochastic model is scored as an ensemble of `members` (seeds 0, 1,
    ..., the same at every validation; a deterministic model has one
    member, whatever `members`), sampled `member_batch` per call where the
    model can (`protocol.forecast_members`): the skill horizon and
    tie-break on the ensemble's CRPS, each stability score the median over
    the members (`metrics.member_median`), and `members` and
    `stable_members` (how many are stable over the whole forecast), which
    `RunSelection` pools into the stable fraction."""
    _, targets = split_context(series, context_steps)
    targets = np.asarray(targets)
    truth = energy_and_enstrophy(targets, dx, dy, chunk_t)
    crps = CrpsAccumulator(targets, scale, chunk_t)
    stability = []
    seeds = range(ensemble_size(model, members))
    for member in forecast_members(model, series, context_steps, seeds, member_batch):
        crps.add(member)
        stability.append(
            stability_scores(
                energy_and_enstrophy(member, dx, dy, chunk_t), truth, block_steps, max_ratio
            )
        )
    curve = crps.crps()
    n_steps = len(curve)
    skill = skill_horizon(curve, skill_threshold)
    tie_break = float(curve[min(tie_break_lead, n_steps) - 1])
    scores = {key: member_median([s[key] for s in stability]) for key in stability[0]}
    # A member's peak ratio is the larger of its two, NaN if either is
    # (np.max propagates NaN).
    peak = member_median([np.max([s[f"{q}_peak_ratio"] for q in QUANTITIES]) for s in stability])
    return {
        "crps_skill_horizon": float(skill),
        f"crps_lead_{tie_break_lead}": tie_break,
        **scores,
        "selection_score": selection_score(skill, tie_break, scores["stable_steps"], n_steps, peak),
        "forecast_steps": float(n_steps),
        "members": float(len(stability)),
        "stable_members": float(sum(s["stable_steps"] >= n_steps for s in stability)),
    }


class RunSelection:
    """The selection rule over one training run's validations: which one's
    network early stopping keeps, and the run's score in the search.

    A network that is stable on the validation forecast at one epoch and not
    at the next is a lucky draw: a refit for the same number of epochs (the
    final train + val refit, a retrain) is likely to land on an unstable
    one. So a validation is judged with the `2 * neighbors + 1` validations
    nearest it (itself included), shifted inward at the ends of the run so
    that every validation is judged with as many (fewer only while the run
    has fewer): its `stable_fraction` is the share of their forecasts stable
    over their whole length -- for a stochastic model the members of all of
    them, pooled (a stable sample is a lucky draw too) -- and its score is
    `metrics.selection_score` with that fraction, which ranks it right after
    the stable steps. With `neighbors` 0 a validation is judged alone, and
    only its `selection_score` is needed; otherwise each validation's scores
    also need `stable_steps`, `forecast_steps`, `members` and
    `stable_members` (`selection_scores` gives them all; a validation
    without the last two counts as one member, stable if its
    `stable_steps` are).

    A validation's window settles once `neighbors` more validations follow
    it (and the run has a full window); until then a later validation can
    still change its score, so the best validation so far is provisional:
    the one early stopping would keep if the run ended now.
    """

    def __init__(self, neighbors: int, history: Iterable[Mapping[str, float]] = ()) -> None:
        if neighbors < 0:
            raise ValueError(f"neighbors must be >= 0, got {neighbors}")
        self.neighbors = neighbors
        self.history = [dict(scores) for scores in history]

    def __len__(self) -> int:
        return len(self.history)

    def add(self, scores: Mapping[str, float]) -> None:
        """Record the next validation's scores."""
        self.history.append(dict(scores))

    @property
    def size(self) -> int:
        """How many validations a full window holds."""
        return 2 * self.neighbors + 1

    def window(self, index: int) -> range:
        """The validations `index` is judged with, as they stand."""
        m = min(self.size, len(self))
        start = min(max(index - self.neighbors, 0), len(self) - m)
        return range(start, start + m)

    def settled(self, index: int) -> bool:
        """Whether `index`'s window (so its score) is final."""
        return len(self) >= max(self.size, index + self.neighbors + 1)

    def members(self, index: int) -> tuple[float, float]:
        """(stable members, members) of validation `index`."""
        scores = self.history[index]
        if "members" in scores:
            return scores["stable_members"], scores["members"]
        return float(scores["stable_steps"] >= scores["forecast_steps"]), 1.0

    def stable_fraction(self, index: int) -> float:
        counts = [self.members(i) for i in self.window(index)]
        return sum(stable for stable, _ in counts) / sum(total for _, total in counts)

    def score(self, index: int) -> float:
        """The validation's `selection_score` with its stable fraction."""
        scores = self.history[index]
        if self.neighbors == 0:
            return scores["selection_score"]
        return scores["selection_score"] + stable_fraction_bonus(
            self.stable_fraction(index),
            scores["stable_steps"],
            int(scores["forecast_steps"]),
            # Forecasts in a full window: the run's members per validation
            # are fixed.
            self.size * int(self.members(index)[1]),
        )

    def best(self) -> int:
        """The best validation so far (the first, on a tie)."""
        if not self.history:
            raise ValueError("no validations recorded")
        return int(np.argmax([self.score(i) for i in range(len(self))]))
