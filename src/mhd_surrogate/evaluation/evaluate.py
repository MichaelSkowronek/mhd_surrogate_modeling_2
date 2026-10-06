"""Score one model on one held-out dataset: the protocol plus all metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from mhd_surrogate.evaluation.diagnostics import DEFAULT_CHUNK_T, NPERSEG, compare_diagnostics
from mhd_surrogate.evaluation.metrics import rmse_per_step, skill_horizon
from mhd_surrogate.evaluation.protocol import ForecastModel, forecast


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
    per-step error curve, index 0 = lead time 1. `predict_seconds` is the
    wall time of the model's forecast alone (no data reading or scoring);
    it's kept out of `scores`, which must be reproducible."""

    scores: dict[str, float]
    rmse: np.ndarray
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
        self.seconds = 0.0

    def predict(self, context: np.ndarray, n_steps: int) -> np.ndarray:
        start = time.perf_counter()
        prediction = np.asarray(self.model.predict(context, n_steps))
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
    chunk_t: int = DEFAULT_CHUNK_T,
    nperseg: int = NPERSEG,
) -> Evaluation:
    """Forecast `series` under the protocol and compute every metric.

    RMSE is scaled per channel by `scale` (the training std), so 1.0 means an
    error as large as the flow's own variability. `report_leads` (1-based) are
    also reported as scalars, `rmse_lead_<n>`; leads past the forecast are
    skipped.
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
    scores.update(compare_diagnostics(prediction, targets, dx, dy, chunk_t, nperseg))
    return Evaluation(scores=scores, rmse=rmse, predict_seconds=timed.seconds)
