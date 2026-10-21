"""Models that forecast in a codec's latent space.

A latent model encodes its context with a codec (`codec.py`), forecasts
the latents of the next frames, and decodes them: raw frames in and out,
like any model, so it is scored, streamed and served like one. Subclasses
supply only the latent dynamics (`fit_latents`, `forecast_latents`).

The codec is fitted on its own (`scripts/training/train_codec.py`) and the
model's config names it: `codec: models:/<model id>` (a logged codec, from
the current tracking URI) or a codec checkpoint directory. `fit` loads it
frozen -- the latent model never changes it -- and the model's checkpoint
carries a copy of it (`codec/`), so loading the model, its pyfunc model,
serving and the DVC stages need nothing else. The reference itself is part
of the model config and so logged as a run param: which codec a latent
model was trained on stays recorded.

Streamed in blocks (`predict_blocks`), the forecast carries its latent
window from block to block. Restarted from its own decoded frames instead,
the model would encode them again, which is the latent it forecast only for
a codec whose `encode(decode(z)) == z` (POD's is, up to rounding; an
autoencoder's isn't).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np

from mhd_surrogate.models.base import CHECKPOINT_META
from mhd_surrogate.models.codec import Codec, load_codec
from mhd_surrogate.training.mlflow_codec import load_logged_codec

CODEC_DIR = "codec"
ARRAYS_FILE = "latent_model.npz"


def resolve_codec(reference: str) -> Codec:
    """The codec `reference` names: a logged model (`models:/<model id>`,
    from the current tracking URI) or a codec checkpoint directory."""
    if reference.startswith("models:/"):
        return load_logged_codec(reference)
    return load_codec(reference)


class LatentSurrogate:
    """The shared part of a latent model: the codec, encoding and decoding,
    streaming, and the checkpoint. A subclass sets `name` and implements
    `fit_latents`, `forecast_latents`, `latent_parameter_count`, `config`
    (its constructor's arguments besides `codec` and `window`), `arrays`
    and `set_arrays` (what its checkpoint stores)."""

    name: str

    def __init__(self, codec: str, window: int = 1) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.codec_reference = codec
        self.window = window
        self.codec: Codec | None = None
        self.fit_info: dict[str, float] = {}
        self.datasets: list[str] = []

    @property
    def requirements(self) -> tuple[str, ...]:
        return tuple(getattr(self.codec, "requirements", ()))

    # -- what a subclass supplies --------------------------------------------

    def fit_latents(self, latents: Mapping[str, np.ndarray]) -> None:
        """Fit the dynamics on the training datasets' latents, {name:
        (T, *latent_shape)}."""
        raise NotImplementedError

    def forecast_latents(self, window: np.ndarray, n_steps: int) -> np.ndarray:
        """The latents of the `n_steps` frames after `window`, the latents
        of the last `self.window` frames, oldest first."""
        raise NotImplementedError

    def latent_parameter_count(self) -> int:
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        raise NotImplementedError

    def arrays(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def set_arrays(self, arrays: Mapping[str, np.ndarray]) -> None:
        raise NotImplementedError

    # -- the model interface -------------------------------------------------

    def fit(self, datasets: Mapping[str, Any]) -> None:
        self.codec = resolve_codec(self.codec_reference)
        self.fit_latents({name: self.codec.encode(array) for name, array in datasets.items()})
        self.datasets = list(datasets)

    def predict(self, context: np.ndarray, n_steps: int) -> np.ndarray:
        latents = self.forecast_latents(self._encode_window(context), n_steps)
        return self.codec.decode(latents)

    def predict_blocks(
        self, context: np.ndarray, n_steps: int, block_steps: int
    ) -> Iterator[np.ndarray]:
        """`predict`'s forecast, `block_steps` frames at a time, carrying the
        latent window from block to block."""
        window = self._encode_window(context)
        for start in range(0, n_steps, block_steps):
            block = self.forecast_latents(window, min(block_steps, n_steps - start))
            window = np.concatenate([window, block])[-self.window :]
            yield self.codec.decode(block)

    def _encode_window(self, context: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if len(context) < self.window:
            raise ValueError(f"context has {len(context)} frames, need window={self.window}")
        return self.codec.encode(np.asarray(context[len(context) - self.window :]))

    def parameter_count(self) -> int:
        """The latent dynamics' parameters plus the codec's: the model
        forecasts with both."""
        self._check_fitted()
        return self.latent_parameter_count() + self.codec.parameter_count()

    def _check_fitted(self) -> None:
        if self.codec is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted")

    # -- checkpoint ----------------------------------------------------------

    def save(self, directory: Path) -> None:
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.codec.save(directory / CODEC_DIR)
        np.savez(directory / ARRAYS_FILE, **self.arrays())
        meta = {
            "name": self.name,
            "codec": self.codec_reference,
            "window": self.window,
            **self.config(),
            "datasets": self.datasets,
            "fit_info": self.fit_info,
        }
        (directory / CHECKPOINT_META).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> Self:
        directory = Path(directory)
        meta = json.loads((directory / CHECKPOINT_META).read_text())
        kwargs = {k: v for k, v in meta.items() if k not in ("name", "datasets", "fit_info")}
        model = cls(**kwargs)
        model.codec = load_codec(directory / CODEC_DIR)
        with np.load(directory / ARRAYS_FILE) as arrays:
            model.set_arrays(dict(arrays))
        model.datasets = meta["datasets"]
        model.fit_info = meta["fit_info"]
        return model
