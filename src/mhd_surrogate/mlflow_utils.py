"""Shared helpers for logging to MLflow."""

from __future__ import annotations

from typing import Any


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
