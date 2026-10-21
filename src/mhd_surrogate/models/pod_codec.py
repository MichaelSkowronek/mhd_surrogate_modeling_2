"""Proper orthogonal decomposition (POD) as a codec: a frame's latent is its
coefficients in the leading `rank` POD modes of the training frames.

The linear codec, in closed form: of all rank-`rank` linear codecs, the one
that reconstructs the (normalized) training frames best. Its basis is the
one DMD's dynamics live in (`snapshots.py`), except that it's the POD of all
training frames, not only of DMD's "before" frames: a codec reconstructs
every frame.

With z the normalization (the pooled training mean field, per-channel
fluctuation RMS) and U the (state x rank) basis, encode(x) = U^T z(x) and
decode(a) = z^-1(U a): decode(encode(x)) is x's orthogonal projection onto
the basis, in normalized units. What it drops is the trailing modes' share
of the variance, the small scales -- a latent model on it can't forecast
what its codec doesn't keep. The training frames' latents are uncorrelated,
each with the variance of its mode (the squared singular value).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np

from mhd_surrogate.models.codec import CODEC_META
from mhd_surrogate.models.snapshots import (
    DEFAULT_CHUNK_T,
    DEFAULT_COLUMN_BLOCK,
    gram_matrix,
    load_frames,
    normalize_in_place,
    project_basis,
    truncated_pod,
)

log = logging.getLogger(__name__)

DEFAULT_RANK = 750


class PODCodec:
    name = "pod"

    def __init__(
        self,
        rank: int = DEFAULT_RANK,
        chunk_t: int = DEFAULT_CHUNK_T,
        column_block: int = DEFAULT_COLUMN_BLOCK,
    ) -> None:
        self.rank = rank
        self.chunk_t = chunk_t
        self.column_block = column_block
        self.mean: np.ndarray | None = None  # (C, Nx, Ny)
        self.scale: np.ndarray | None = None  # (C,)
        self.basis: np.ndarray | None = None  # (state, r), POD modes U
        self.fit_info: dict[str, float] = {}
        self.datasets: list[str] = []

    @property
    def latent_shape(self) -> tuple[int, ...]:
        self._check_fitted()
        return (self.basis.shape[1],)

    def fit(self, datasets: Mapping[str, Any]) -> None:
        if not datasets or min(array.shape[0] for array in datasets.values()) < 1:
            raise ValueError("PODCodec.fit needs datasets with at least 1 frame each")
        frame_shape = tuple(next(iter(datasets.values())).shape[1:])
        frames = load_frames(datasets, self.chunk_t, self.name)
        self.mean, self.scale = normalize_in_place(frames, frame_shape, self.chunk_t)

        s2, s, v = truncated_pod(gram_matrix(frames, self.column_block), self.rank, self.name)
        r = len(s)
        self.basis = project_basis(frames, np.arange(len(frames)), v / s, self.column_block)
        self.datasets = list(datasets)
        self.fit_info = {
            "rank": float(r),
            "explained_variance": float(s2[:r].sum() / s2.sum()),
            "n_frames": float(len(frames)),
        }
        log.info("pod: %s", ", ".join(f"{k}={v:.4g}" for k, v in self.fit_info.items()))

    def encode(self, frames: Any) -> np.ndarray:
        """The POD coefficients U^T z(x), read `chunk_t` frames at a time."""
        self._check_fitted()
        latents = np.empty((frames.shape[0], *self.latent_shape), dtype=np.float32)
        for start in range(0, frames.shape[0], self.chunk_t):
            block = np.asarray(frames[start : start + self.chunk_t], dtype=np.float32)
            z = (block - self.mean[None]) / self.scale[None, :, None, None]
            latents[start : start + len(block)] = z.reshape(len(block), -1) @ self.basis
        return latents

    def decode(self, latents: np.ndarray) -> np.ndarray:
        self._check_fitted()
        z = (np.asarray(latents, dtype=np.float32) @ self.basis.T).reshape(-1, *self.mean.shape)
        return z * self.scale[None, :, None, None] + self.mean[None]

    def parameter_count(self) -> int:
        """The POD modes; the mean field and channel scales are normalization."""
        self._check_fitted()
        return self.basis.size

    def _check_fitted(self) -> None:
        if self.basis is None:
            raise RuntimeError("PODCodec is not fitted")

    # -- checkpoint ----------------------------------------------------------

    def save(self, directory: Path) -> None:
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "pod.npz", mean=self.mean, scale=self.scale, basis=self.basis)
        meta = {
            "name": self.name,
            "rank": self.rank,
            "chunk_t": self.chunk_t,
            "column_block": self.column_block,
            "datasets": self.datasets,
            "fit_info": self.fit_info,
        }
        (directory / CODEC_META).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> Self:
        directory = Path(directory)
        meta = json.loads((directory / CODEC_META).read_text())
        codec = cls(meta["rank"], meta["chunk_t"], meta["column_block"])
        with np.load(directory / "pod.npz") as arrays:
            codec.mean = arrays["mean"]
            codec.scale = arrays["scale"]
            codec.basis = arrays["basis"]
        codec.datasets = meta["datasets"]
        codec.fit_info = meta["fit_info"]
        return codec
