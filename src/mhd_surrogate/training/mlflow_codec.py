"""Log a fitted codec as an MLflow model, and load one back.

A codec is logged as a surrogate is (`mlflow_model.py`): a `pyfunc` model
carrying the checkpoint and this package's source, so it shows up as the
run's model with a page of its own, and a latent model can reference it by
id (`models:/<model id>`). Its `predict` is the reconstruction,
decode(encode(frames)), so input and output are frames whatever the latent
shape; a latent model takes the codec itself (`load_logged_codec`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from mlflow.models import ModelSignature
from mlflow.models.model import ModelInfo
from mlflow.types import Schema, TensorSpec

import mlflow
from mhd_surrogate.models.codec import Codec
from mhd_surrogate.training.mlflow_model import PACKAGE_DIR, pinned_requirements


class CodecPyfunc(mlflow.pyfunc.PythonModel):
    """`predict(frames)` -> decode(encode(frames)), raw units, shape
    (T, C, Nx, Ny) in and out; the codec is `unwrap_python_model().codec`."""

    def load_context(self, context) -> None:
        from mhd_surrogate.models.codec import load_codec

        self.codec = load_codec(context.artifacts["checkpoint"])

    def predict(self, context, model_input, params: dict[str, Any] | None = None) -> np.ndarray:
        return self.codec.decode(self.codec.encode(np.asarray(model_input)))


def signature(frame_shape: tuple[int, ...]) -> ModelSignature:
    spec = TensorSpec(np.dtype(np.float32), (-1, *frame_shape))
    return ModelSignature(inputs=Schema([spec]), outputs=Schema([spec]))


def log_codec(
    codec: Codec,
    checkpoint: Path,
    frame_shape: tuple[int, ...],
    params: dict[str, Any] | None = None,
) -> ModelInfo:
    """Log `checkpoint` (written by `codec.save`) as an MLflow model named
    after the codec, in the active run; `params` (e.g. the codec config) are
    attached to the logged model, with its `n_parameters` and
    `latent_shape` beside them."""
    params = {
        **(params or {}),
        "n_parameters": codec.parameter_count(),
        "latent_shape": list(codec.latent_shape),
    }
    return mlflow.pyfunc.log_model(
        name=codec.name,
        python_model=CodecPyfunc(),
        artifacts={"checkpoint": str(checkpoint)},
        code_paths=[str(PACKAGE_DIR)],
        signature=signature(frame_shape),
        pip_requirements=pinned_requirements(getattr(codec, "requirements", ())),
        params={k: str(v) for k, v in params.items()},
    )


def load_logged_codec(uri: str) -> Codec:
    """The codec inside the logged model `uri` (`models:/<model id>`), from
    the current tracking URI."""
    return mlflow.pyfunc.load_model(uri).unwrap_python_model().codec
