"""Shared helpers for logging to MLflow."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from mlflow.entities import Metric

import mlflow

# MLflow's limit on metrics per log_batch call.
MAX_BATCH = 1000


def flatten_for_mlflow(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into dot-separated keys, for `mlflow.log_params`.

    `mlflow.log_params` takes a flat mapping; a resolved Hydra/OmegaConf
    config is nested (e.g. `{"data": {"dataset": "re16k_t400_0"}}`), so this
    turns that into `{"data.dataset": "re16k_t400_0"}`.
    """
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_for_mlflow(value, full_key))
        else:
            flat[full_key] = value
    return flat


def finite_metrics(metrics: Mapping[str, float]) -> tuple[dict[str, float], list[str]]:
    """Split `metrics` into the finite ones and the names of the rest.

    NaN/inf mark a metric that's undefined for this run (e.g. the oscillation
    period of a forecast with no oscillation); they're left out of MLflow,
    where they would break sorting and plotting, and reported by name instead.
    """
    finite = {k: float(v) for k, v in metrics.items() if math.isfinite(v)}
    return finite, [k for k in metrics if k not in finite]


def log_metric_series(
    key: str, values: Sequence[float], start_step: int = 0, model_id: str | None = None
) -> None:
    """Log `values` as one metric's history, value i at step `start_step + i`,
    in batches rather than one request per step; `model_id` links it to a
    logged model as well as the run."""
    run_id = mlflow.active_run().info.run_id
    timestamp = int(time.time() * 1000)
    metrics = [
        Metric(key, float(v), timestamp, start_step + i, model_id=model_id, run_id=run_id)
        for i, v in enumerate(values)
    ]
    client = mlflow.MlflowClient()
    for i in range(0, len(metrics), MAX_BATCH):
        client.log_batch(run_id, metrics=metrics[i : i + MAX_BATCH])
