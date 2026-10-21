"""The interface every codec implements, and the codec registry.

A codec is an encode/decode pair: `encode` maps raw frames, shape
(T, C, Nx, Ny), to latents, shape (T, *latent_shape), and `decode` maps
latents back to raw frames. A latent model forecasts in a codec's latent
space instead of on the grid. POD is a codec (a linear one); so is an
autoencoder, a VAE or a VQ-VAE. Like a surrogate model (`base.py`), a codec
takes and returns raw fields and normalizes internally.

A codec is fitted, logged and referenced on its own, not as a first stage
of a latent model's `fit`: a sweep or a set of seeds of the latent model
would otherwise refit it in every trial, and the codec's own variance would
confound their comparison (as in latent diffusion, where the autoencoder is
trained first and frozen).

It forecasts nothing, so it isn't selected like a surrogate. A codec
trained iteratively early-stops on its validation reconstruction RMSE: with
the frame given, the RMSE-optimal reconstruction is the frame itself, not a
smoothed mean as for an uncertain forecast. Choosing between codecs (latent
size, architecture) is choosing between the latent models trained on them,
by their `selection_score`.

A checkpoint is a directory holding `codec.json` (the registry name plus the
codec's own metadata) and whatever arrays it needs; `load_codec` reads the
name and dispatches to the right class. It isn't a model's `model.json`, so
a codec's checkpoint can't be loaded as a model by mistake.

Codecs are registered by import path, as models are (`registry.py`): a
codec's module is imported only when that codec is built or loaded.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, Self

import numpy as np

CODEC_META = "codec.json"

# name -> "module:Class"; the name is also the class's `name` attribute.
CODECS: dict[str, str] = {
    "pod": "mhd_surrogate.models.pod_codec:PODCodec",
}


class Codec(Protocol):
    name: str
    # The shape of one frame's latent; set by `fit` (or `load`).
    latent_shape: tuple[int, ...]

    def fit(self, datasets: Mapping[str, Any]) -> None:
        """Fit on the training datasets, {name: array of shape (T, C, Nx, Ny)}
        (zarr arrays are fine)."""

    def encode(self, frames: Any) -> np.ndarray:
        """Latents of raw `frames`, (T, C, Nx, Ny) -> (T, *latent_shape)
        (a zarr array is fine)."""

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Raw frames of `latents`, (T, *latent_shape) -> (T, C, Nx, Ny)."""

    def parameter_count(self) -> int:
        """How many numbers the fitted codec learned (as for a model,
        `base.py`: normalization statistics don't count)."""

    def save(self, directory: Path) -> None:
        """Write the checkpoint into `directory` (created if missing)."""

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Rebuild the fitted codec from a checkpoint written by `save`."""


def codec_class(name: str) -> type:
    if name not in CODECS:
        raise ValueError(f"unknown codec {name!r}; known: {sorted(CODECS)}")
    module, _, cls = CODECS[name].partition(":")
    return getattr(importlib.import_module(module), cls)


def build_codec(config: Mapping[str, Any]) -> Codec:
    """`config` is the `codec` config group: `name` picks the class, every
    other key is passed to its constructor."""
    kwargs = dict(config)
    return codec_class(kwargs.pop("name"))(**kwargs)


def load_codec(directory: Path | str) -> Codec:
    meta = json.loads((Path(directory) / CODEC_META).read_text())
    return codec_class(meta["name"]).load(Path(directory))
