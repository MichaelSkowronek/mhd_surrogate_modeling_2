"""Score one model on one held-out dataset: the protocol plus all metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mhd_surrogate.evaluation.diagnostics import DEFAULT_CHUNK_T, NPERSEG, compare_diagnostics
from mhd_surrogate.evaluation.metrics import rmse_per_step, skill_horizon
from mhd_surrogate.evaluation.protocol import ForecastModel, forecast


@dataclass(frozen=True)
class Evaluation:
    """`scores` are scalars (ready for `mlflow.log_metrics`); `rmse` is the
    per-step error curve, index 0 = lead time 1."""

    scores: dict[str, float]
    rmse: np.ndarray


def evaluate(
    model: ForecastModel,
    series,
    context_steps: int,
    dx: float,
    dy: float,
    scale: np.ndarray,
    skill_threshold: float,
    report_leads: list[int],
    chunk_t: int = DEFAULT_CHUNK_T,
    nperseg: int = NPERSEG,
) -> Evaluation:
    """Forecast `series` under the protocol and compute every metric.

    RMSE is scaled per channel by `scale` (the training std), so 1.0 means an
    error as large as the flow's own variability. `report_leads` (1-based) are
    also reported as scalars, `rmse_lead_<n>`; leads past the forecast are
    skipped.
    """
    prediction, targets = forecast(model, series, context_steps)
    rmse = rmse_per_step(prediction, targets, scale)
    scores = {
        "rmse_mean": float(rmse.mean()),
        "skill_horizon": float(skill_horizon(rmse, skill_threshold)),
    }
    for lead in report_leads:
        if 1 <= lead <= len(rmse):
            scores[f"rmse_lead_{lead}"] = float(rmse[lead - 1])
    scores.update(compare_diagnostics(prediction, targets, dx, dy, chunk_t, nperseg))
    return Evaluation(scores=scores, rmse=rmse)
