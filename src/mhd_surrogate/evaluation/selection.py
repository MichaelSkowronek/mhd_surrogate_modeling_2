"""The selection rule's scores, cheap enough to run every epoch.

Early stopping and the hyperparameter search score a model in training by
`selection_scores`; `evaluate.evaluate` scores the final model in full. In
a module of its own (with `metrics`, `protocol`, `stability` and
`quantities`), so that training -- and the DVC `train` stage's
dependencies -- reach only this part of the evaluation code: a change to
the diagnostics or the ensemble scores re-runs the `evaluate` stage on the
stored checkpoint instead of retraining the model.
"""

from __future__ import annotations

import numpy as np

from mhd_surrogate.evaluation.metrics import rmse_per_step, selection_score, skill_horizon
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
    scores (`stable_steps` and the peak ratios) and their combination
    `selection_score`. Cheap enough to run during training (early stopping,
    the hyperparameter search; the stability check adds ~3 s to an 837-step
    validation forecast); `evaluate` scores the final model in full."""
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
    }
