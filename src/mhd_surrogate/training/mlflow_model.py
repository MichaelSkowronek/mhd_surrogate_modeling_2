"""Log a fitted surrogate as an MLflow model.

Wraps a checkpoint (see `models/base.py`) as a `pyfunc` model, so MLflow
knows the run produced a model: it shows up in the runs table's Models
column and gets its own page, its scores can be linked to it, it can be
registered in the Model Registry, and `mlflow.pyfunc.load_model` /
`mlflow models serve` can load and serve it.

The logged model is self-contained: it carries the checkpoint as an artifact
and this package's source as `code_paths`, so loading it needs only the
pinned third-party requirements, not an installed `mhd_surrogate`.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np
from mlflow.models import ModelSignature
from mlflow.models.model import ModelInfo
from mlflow.types import ParamSchema, ParamSpec, Schema, TensorSpec

import mlflow
from mhd_surrogate.models.base import SurrogateModel

PACKAGE_DIR = Path(__file__).resolve().parents[1]
# Third-party packages every model needs at load/predict time (the baselines
# import jax; the pyfunc wrapper itself needs mlflow). A model adds its own
# as a `requirements` class attribute (e.g. the neural ones: equinox).
REQUIREMENTS = ("numpy", "jax", "mlflow")


class SurrogatePyfunc(mlflow.pyfunc.PythonModel):
    """`predict(context, params={"n_steps": n})` -> `model.predict(context, n)`.

    `context` is the last `window` frames, shape (window, C, Nx, Ny), raw
    units; extra leading frames are ignored, as under the forecast protocol.
    """

    def load_context(self, context) -> None:
        from mhd_surrogate.models.registry import load_model

        self.model = load_model(context.artifacts["checkpoint"])

    def predict(self, context, model_input, params: dict[str, Any] | None = None) -> np.ndarray:
        n_steps = int((params or {}).get("n_steps", 1))
        frames = np.asarray(model_input)
        visible = frames[frames.shape[0] - self.model.window :]
        return np.asarray(self.model.predict(visible, n_steps))


def signature(frame_shape: tuple[int, ...]) -> ModelSignature:
    """Input and output are stacks of frames of `frame_shape` (C, Nx, Ny);
    `n_steps` is a parameter."""
    spec = TensorSpec(np.dtype(np.float32), (-1, *frame_shape))
    return ModelSignature(
        inputs=Schema([spec]),
        outputs=Schema([spec]),
        params=ParamSchema([ParamSpec("n_steps", "long", 1)]),
    )


def pinned_requirements(extra: tuple[str, ...] = ()) -> list[str]:
    """REQUIREMENTS plus a model's own `extra` ones, pinned to the installed versions."""
    names = list(dict.fromkeys([*REQUIREMENTS, *extra]))
    return [f"{name}=={importlib.metadata.version(name)}" for name in names]


def log_surrogate(
    model: SurrogateModel,
    checkpoint: Path,
    frame_shape: tuple[int, ...],
    params: dict[str, Any] | None = None,
) -> ModelInfo:
    """Log `checkpoint` (written by `model.save`) as an MLflow model named
    after the model, in the active run; `params` (e.g. the model config) are
    attached to the logged model."""
    return mlflow.pyfunc.log_model(
        name=model.name,
        python_model=SurrogatePyfunc(),
        artifacts={"checkpoint": str(checkpoint)},
        code_paths=[str(PACKAGE_DIR)],
        signature=signature(frame_shape),
        pip_requirements=pinned_requirements(getattr(model, "requirements", ())),
        params={k: str(v) for k, v in (params or {}).items()},
    )


def load_logged_surrogate(model_id: str) -> tuple[SurrogateModel, str]:
    """The surrogate inside the logged model `model_id` (from the current
    tracking URI), plus the id of the run that logged it."""
    logged = mlflow.get_logged_model(model_id)
    pyfunc = mlflow.pyfunc.load_model(f"models:/{model_id}")
    return pyfunc.unwrap_python_model().model, logged.source_run_id
