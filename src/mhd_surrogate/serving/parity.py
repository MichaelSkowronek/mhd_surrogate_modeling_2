"""Serve parity: the forecast the service serves is the model's own.

The body of the DVC `serve_parity` stage (`scripts/serving/serve_parity.py`).
It runs the canonical checkpoint's validation forecast through the forecast
service in process -- the FastAPI app with the deployment's defaults, the
`.npy` stream, the client reading it back (`serving/app.py`, `stream.py`,
`client.py`) -- and compares it with the forecast of the evaluation
protocol (`evaluation/protocol.py`'s `forecast`), the one the scores and
the gate are of. The stage's deps are the serving code it runs, so a change
to that code re-runs it (CI's dvc.lock check) and must pass: the serving
code is checked against the real model before it merges, where CI's smoke
deploy only has a toy one.

It forecasts on the CPU (the stage runs with JAX_PLATFORMS=cpu), as the
deployed pod does. The service streams the forecast in blocks, so it differs
from the one-call forecast by float rounding (5.2e-6 of the truth's
standard deviation for Hankel DMD); the tolerance (1e-4,
`configs/serving/default.yaml`) is about 20 times that, so a change to the
block arithmetic passes and a change in what the service serves doesn't.
The test dataset is never opened.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from fastapi.testclient import TestClient
from omegaconf import DictConfig, OmegaConf

from mhd_surrogate.evaluation.protocol import (
    ForecastModel,
    check_readable,
    check_window,
    forecast,
)
from mhd_surrogate.models.registry import load_model
from mhd_surrogate.serving.app import create_app
from mhd_surrogate.serving.client import ForecastClient
from mhd_surrogate.serving.fetch import FetchedModel
from mhd_surrogate.serving.schemas import ServeSettings
from mhd_surrogate.training.export import write_metrics
from mhd_surrogate.training.startup import configure_process

log = logging.getLogger(__name__)

VERDICT_FILE = "serve_parity.json"


@dataclass(frozen=True)
class ParityResult:
    """The largest difference between the served and the offline forecast,
    in units of the truth's standard deviation, over `steps` steps."""

    max_difference: float
    tolerance: float
    steps: int

    @property
    def passed(self) -> bool:
        return self.max_difference <= self.tolerance

    def verdict(self) -> dict[str, float]:
        return {
            "passed": float(self.passed),
            "max_difference": self.max_difference,
            "tolerance": self.tolerance,
            "steps": float(self.steps),
        }


def relative_difference(served: np.ndarray, offline: np.ndarray, truth: np.ndarray) -> float:
    """The largest pointwise difference of two forecasts, in units of the
    truth's standard deviation."""
    return float(np.max(np.abs(served - offline)) / np.std(truth))


def served_forecast(
    model: ForecastModel, frame_shape: tuple[int, ...], context: np.ndarray, n_steps: int
) -> np.ndarray:
    """`model`'s forecast of `n_steps` from `context` as the service serves
    it: through the app (in process; FastAPI's TestClient is its ASGI
    transport), with the deployment's settings but the step limit, streamed
    and read back by the client. One member (seed 0)."""
    fetched = FetchedModel(
        model_uri="in-process",
        name=model.name,
        version=None,
        model_id="in-process",
        model_type=model.name,
        checkpoint_sha256="",
        source_commit=None,
        validated_steps=n_steps,
        frame_shape=frame_shape,
    )
    settings = ServeSettings(model_dir=".", max_context_frames=max(len(context), 1))
    with TestClient(create_app(model, fetched, settings)) as http:
        (served,) = ForecastClient(http=http).forecast(context, n_steps)
    return served


def serve_parity(
    model: ForecastModel, series, context_steps: int, tolerance: float
) -> ParityResult:
    """Compare `model`'s served forecast of `series` (after its first
    `context_steps` steps) with its forecast under the protocol."""
    offline, truth = forecast(model, series, context_steps)
    context = np.asarray(series[:context_steps], dtype=np.float32)
    served = served_forecast(model, tuple(series.shape[1:]), context, len(truth))
    return ParityResult(relative_difference(served, offline, truth), tolerance, len(truth))


def run_parity(cfg: DictConfig) -> ParityResult:
    """Check the checkpoint `cfg.checkpoint` on the validation dataset and
    write the verdict to `<cfg.export.dir>/serve_parity.json`, failing or
    not."""
    if not cfg.checkpoint or not cfg.export.dir:
        raise ValueError("checkpoint=<checkpoint directory> and export.dir=<dir> are required")
    configure_process(cfg)
    dataset = cfg.data.val_dataset
    check_readable(dataset, OmegaConf.to_container(cfg.data, resolve=True))
    model = load_model(Path(cfg.checkpoint))
    check_window(model.window, cfg.data.context_steps)
    series = zarr.open_group(store=cfg.data.zarr_store, mode="r")[dataset]
    log.info("serving %s's forecast of %s", model.name, dataset)
    result = serve_parity(model, series, cfg.data.context_steps, cfg.serving.parity_tolerance)
    path = Path(cfg.export.dir) / VERDICT_FILE
    write_metrics(path, {"val": result.verdict()})
    log.info("wrote %s", path)
    return result
