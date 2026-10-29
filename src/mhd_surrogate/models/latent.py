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
autoencoder's isn't). For a subclass whose state is exactly the latents it
returns, the blocks are its one forecast exactly.

A latent model trained iteratively (`iterative = True`: a network) gets
the training entry point's `FitHooks` (`base.py`) and passes them on to
`fit_latents`. A generative one (`stochastic = True`) builds on
`StochasticLatentSurrogate` and samples latents for a batch of seeds at
once (`sample_latents`); its `predict`, `predict_members` and
`predict_blocks` all go through that one method, so a member draws the
same noise whichever of them asks for it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np

from mhd_surrogate.models.base import CHECKPOINT_META, FitHooks
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
    # A subclass trained by gradient steps sets it: its `fit_latents` also
    # takes the `FitHooks`.
    iterative = False

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

    def fit_latents(self, latents: Mapping[str, np.ndarray], hooks: FitHooks | None = None) -> None:
        """Fit the dynamics on the training datasets' latents, {name:
        (T, *latent_shape)}; an iterative model's also takes the `FitHooks`
        (a closed-form one's none)."""
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

    def fit(self, datasets: Mapping[str, Any], hooks: FitHooks | None = None) -> None:
        self.codec = resolve_codec(self.codec_reference)
        latents = {name: self.codec.encode(array) for name, array in datasets.items()}
        if self.iterative:
            self.fit_latents(latents, hooks or FitHooks())
        else:
            self.fit_latents(latents)
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


class StochasticLatentSurrogate(LatentSurrogate):
    """A generative latent model: one sample of the forecast per seed
    (`stochastic = True`, see `base.py`). A subclass implements
    `sample_latents` in place of `forecast_latents`; `predict`,
    `predict_members` and `predict_blocks` are built on it."""

    stochastic = True

    def sample_latents(
        self, window: np.ndarray, n_steps: int, seeds: Sequence[int], first_step: int = 0
    ) -> np.ndarray:
        """The latents of the `n_steps` frames after `window` (as for
        `forecast_latents`) in the sample for each of `seeds`, shape
        (len(seeds), n_steps, *latent_shape). They are forecast steps
        `first_step` onward of that sample: its randomness drawn per step
        from its seed and the step's index alone, so streamed from its own
        window a block is the same as in one call, and a member draws the
        same noise in any batch (its rounding, which a long forecast
        amplifies, can differ with the batch)."""
        raise NotImplementedError

    def predict(self, context: np.ndarray, n_steps: int, seed: int = 0) -> np.ndarray:
        window = self._encode_window(context)
        return self.codec.decode(self.sample_latents(window, n_steps, [seed])[0])

    def predict_members(
        self, context: np.ndarray, n_steps: int, seeds: Sequence[int]
    ) -> np.ndarray:
        """The forecasts for `seeds`, (len(seeds), n_steps, C, Nx, Ny),
        sampled in one `sample_latents` call and decoded a member at a time
        (~1 GB each for a whole validation forecast: decoded at once, the
        codec's intermediates would be the whole batch's too)."""
        latents = self.sample_latents(self._encode_window(context), n_steps, list(seeds))
        first = self.codec.decode(latents[0])
        frames = np.empty((len(latents), *first.shape), dtype=first.dtype)
        frames[0] = first
        for k in range(1, len(latents)):
            frames[k] = self.codec.decode(latents[k])
        return frames

    def predict_blocks(
        self, context: np.ndarray, n_steps: int, block_steps: int, seed: int = 0
    ) -> Iterator[np.ndarray]:
        """`predict`'s forecast for `seed`, `block_steps` frames at a time,
        carrying the latent window from block to block."""
        window = self._encode_window(context)
        for start in range(0, n_steps, block_steps):
            n = min(block_steps, n_steps - start)
            block = self.sample_latents(window, n, [seed], first_step=start)[0]
            window = np.concatenate([window, block])[-self.window :]
            yield self.codec.decode(block)
