"""Score one model on one held-out dataset: the protocol plus all metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from mhd_surrogate.evaluation.diagnostics import (
    DEFAULT_CHUNK_T,
    NPERSEG,
    compare_summaries,
    series_summary,
)
from mhd_surrogate.evaluation.ensemble import EnsembleAccumulator, EnsembleScores
from mhd_surrogate.evaluation.metrics import rmse_per_step, skill_horizon
from mhd_surrogate.evaluation.protocol import (
    ForecastModel,
    ensemble_size,
    forecast,
    forecast_members,
    is_stochastic,
)
from mhd_surrogate.evaluation.stability import QUANTITIES, stability_scores


def train_eval_datasets(requested: list[str], train_datasets: list[str]) -> list[str]:
    """The training datasets to also score a model on, checked to really be
    training datasets: scoring "train" on anything else would mislabel a
    held-out dataset's scores (and could read the test dataset)."""
    unknown = [name for name in requested if name not in train_datasets]
    if unknown:
        raise ValueError(f"evaluation.train_datasets {unknown} are not in data.train_datasets")
    return list(requested)


@dataclass(frozen=True)
class Evaluation:
    """`scores` are scalars (ready for `mlflow.log_metrics`); `rmse` is the
    per-step error curve, index 0 = lead time 1, and `ensemble` the
    ensemble's per-step curves (CRPS, spread). `predict_seconds` is the wall
    time of one member's forecast alone (no data reading or scoring); it's
    kept out of `scores`, which must be reproducible."""

    scores: dict[str, float]
    rmse: np.ndarray
    ensemble: EnsembleScores
    predict_seconds: float

    @property
    def seconds_per_frame(self) -> float:
        """Forecast cost per predicted frame, comparable across datasets of
        different lengths (and with the solver's cost per time step)."""
        return self.predict_seconds / len(self.rmse)


class _TimedModel:
    """Wraps a model to time its `predict` call. The result is converted to
    NumPy inside the timed block so JAX's asynchronous dispatch is waited
    for, not just launched."""

    def __init__(self, model: ForecastModel) -> None:
        self.model = model
        self.window = model.window
        self.stochastic = is_stochastic(model)
        self.seconds = 0.0

    def predict(self, context: np.ndarray, n_steps: int, **seed) -> np.ndarray:
        start = time.perf_counter()
        prediction = np.asarray(self.model.predict(context, n_steps, **seed))
        self.seconds = time.perf_counter() - start
        return prediction


def evaluate(
    model: ForecastModel,
    series,
    context_steps: int,
    dx: float,
    dy: float,
    scale: np.ndarray,
    skill_threshold: float,
    report_leads: list[int],
    block_steps: int,
    max_ratio: float,
    n_members: int = 1,
    member_batch: int = 1,
    chunk_t: int = DEFAULT_CHUNK_T,
    nperseg: int = NPERSEG,
) -> Evaluation:
    """Forecast `series` under the protocol and compute every metric.

    RMSE is scaled per channel by `scale` (the training std), so 1.0 means an
    error as large as the flow's own variability. `report_leads` (1-based) are
    also reported as scalars, `rmse_lead_<n>`; leads past the forecast are
    skipped. `block_steps` and `max_ratio` configure the stability check
    (`stability.stability_scores`).

    A stochastic model forecasts an ensemble of `n_members` (seeds 0, 1,
    ...; a deterministic model has one member, whatever `n_members`). Every
    score above is the seed-0 member's, the single trajectory `predict`
    gives; the ensemble adds `crps_mean`, `crps_lead_<n>`,
    `ensemble_size` and, for more than one member, `spread_skill_lead_<n>`
    (`ensemble.EnsembleScores`). For one member the CRPS is the absolute
    error, so deterministic and generative models share that score.
    `member_batch` members are sampled per call where the model can
    (`protocol.forecast_members`).
    """
    timed = _TimedModel(model)
    prediction, targets = forecast(timed, series, context_steps)
    rmse = rmse_per_step(prediction, targets, scale)
    scores = {
        "rmse_mean": float(rmse.mean()),
        "skill_horizon": float(skill_horizon(rmse, skill_threshold)),
    }
    for lead in report_leads:
        if 1 <= lead <= len(rmse):
            scores[f"rmse_lead_{lead}"] = float(rmse[lead - 1])
    pred = series_summary(prediction, dx, dy, chunk_t)
    true = series_summary(targets, dx, dy, chunk_t)
    scores.update(compare_summaries(pred, true, nperseg))
    scores.update(
        stability_scores(
            {q: pred[q] for q in QUANTITIES},
            {q: true[q] for q in QUANTITIES},
            block_steps,
            max_ratio,
        )
    )

    members = EnsembleAccumulator(targets, scale, chunk_t)
    members.add(prediction)
    del prediction  # one member at a time from here on (~1 GB each)
    size = ensemble_size(model, n_members)
    members_after_0 = forecast_members(
        model, series, context_steps, range(1, size), batch_size=member_batch
    )
    for member in members_after_0:
        members.add(member)
    ensemble = members.scores()
    scores["ensemble_size"] = float(ensemble.size)
    scores["crps_mean"] = float(ensemble.crps.mean())
    for lead in report_leads:
        if 1 <= lead <= len(rmse):
            scores[f"crps_lead_{lead}"] = float(ensemble.crps[lead - 1])
            if ensemble.size > 1:
                scores[f"spread_skill_lead_{lead}"] = float(ensemble.spread_skill[lead - 1])
    return Evaluation(scores=scores, rmse=rmse, ensemble=ensemble, predict_seconds=timed.seconds)
