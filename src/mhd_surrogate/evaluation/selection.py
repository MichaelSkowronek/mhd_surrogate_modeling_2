"""The selection rule's scores, cheap enough to run every epoch.

Early stopping and the hyperparameter search score a model in training by
`selection_scores`, and choose between a run's validations with
`RunSelection`; `evaluate.evaluate` scores the final model in full. In a
module of its own (with `metrics`, `protocol`, `stability` and
`quantities`), so that training -- and the DVC `train` stage's
dependencies -- reach only this part of the evaluation code: a change to
the diagnostics or the ensemble scores re-runs the `evaluate` stage on the
stored checkpoint instead of retraining the model.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from mhd_surrogate.evaluation.metrics import (
    rmse_per_step,
    selection_score,
    skill_horizon,
    stable_fraction_bonus,
)
from mhd_surrogate.evaluation.protocol import ForecastModel, forecast
from mhd_surrogate.evaluation.quantities import DEFAULT_CHUNK_T
from mhd_surrogate.evaluation.stability import energy_and_enstrophy, stability_scores


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
    chunk_t: int = DEFAULT_CHUNK_T,
) -> dict[str, float]:
    """The scores the selection rule ranks by, without the physics
    guardrails: `skill_horizon`, `rmse_lead_<tie_break_lead>`, the stability
    scores (`stable_steps` and the peak ratios), their combination
    `selection_score` (of this forecast alone; `RunSelection` adds the
    stable fraction) and the forecast's length, `forecast_steps`. Cheap
    enough to run during training (early stopping, the hyperparameter
    search; the stability check adds ~3 s to an 837-step validation
    forecast); `evaluate` scores the final model in full."""
    prediction, targets = forecast(model, series, context_steps)
    # A chunk of lead times at a time: the per-step float64 error of a whole
    # ~840-step forecast would take ~6 GB of temporaries, at every validation.
    rmse = np.concatenate(
        [
            rmse_per_step(prediction[t : t + chunk_t], targets[t : t + chunk_t], scale)
            for t in range(0, len(targets), chunk_t)
        ]
    )
    skill = skill_horizon(rmse, skill_threshold)
    tie_break = float(rmse[min(tie_break_lead, len(rmse)) - 1])
    stability = stability_scores(
        energy_and_enstrophy(prediction, dx, dy, chunk_t),
        energy_and_enstrophy(targets, dx, dy, chunk_t),
        block_steps,
        max_ratio,
    )
    return {
        "skill_horizon": float(skill),
        f"rmse_lead_{tie_break_lead}": tie_break,
        **stability,
        "selection_score": selection_score(skill, tie_break, stability["stable_steps"], len(rmse)),
        "forecast_steps": float(len(rmse)),
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
    has fewer): its `stable_fraction` is the share of them stable over their
    whole forecast, and its score is `metrics.selection_score` with that
    fraction, which ranks it right after the stable steps. With `neighbors`
    0 a validation is judged alone, and only its `selection_score` is
    needed; otherwise each validation's scores also need `stable_steps` and
    `forecast_steps` (`selection_scores` gives all three).

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

    def stable(self, index: int) -> bool:
        scores = self.history[index]
        return scores["stable_steps"] >= scores["forecast_steps"]

    def stable_fraction(self, index: int) -> float:
        window = self.window(index)
        return sum(self.stable(i) for i in window) / len(window)

    def score(self, index: int) -> float:
        """The validation's `selection_score` with its stable fraction."""
        scores = self.history[index]
        if self.neighbors == 0:
            return scores["selection_score"]
        return scores["selection_score"] + stable_fraction_bonus(
            self.stable_fraction(index),
            scores["stable_steps"],
            int(scores["forecast_steps"]),
            self.size,
        )

    def best(self) -> int:
        """The best validation so far (the first, on a tie)."""
        if not self.history:
            raise ValueError("no validations recorded")
        return int(np.argmax([self.score(i) for i in range(len(self))]))
