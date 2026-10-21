"""Dynamic Mode Decomposition as a forecasting model.

DMD fits the best linear operator A (least squares, z_{t+1} ~= A z_t) to the
training snapshots and forecasts by applying it repeatedly. It is the
simplest model with dynamics: between the baselines (none) and a nonlinear
network. Being linear, it can carry the flow's coherent oscillation forward,
but not the turbulence: decaying modes die out (the forecast relaxes to the
mean) and the rank truncation removes the small scales.

State. Each frame is a normalized snapshot z_t (`snapshots.py`): a
fluctuation about the pooled training mean field, scaled per channel by the
fluctuation's training RMS, flattened into one vector.

Fit (projected exact DMD over several trajectories, method of snapshots).
Snapshot pairs (z_t, z_{t+1}) are formed within each dataset only, never
across the boundary between two datasets. With X the "before" snapshots and
Y the "after" ones (one column per pair), the Gram matrix G = Z^T Z of all
frames gives X^T X = V S^2 V^T (POD basis U = X V S^-1, truncated to `rank`)
and the reduced operator A~ = U^T A U = S^-1 V^T (X^T Y) V S^-1, without
ever forming the (state x state) operator or holding X in a factorized form.

Forecast. The last context frame is projected onto U, evolved with the
eigendecomposition A~ = W diag(lambda) W^-1, and mapped back. With
`stabilize`, eigenvalues outside the unit circle are moved onto it (same
frequency, no growth): a single growing mode would otherwise blow the
forecast up over hundreds of steps. Streamed in blocks (`predict_blocks`),
every block evolves the same amplitudes, so a long forecast is one
trajectory.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np

from mhd_surrogate.models.base import CHECKPOINT_META, real_size
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

DEFAULT_RANK = 100


def stabilize_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
    """Move eigenvalues outside the unit circle onto it, keeping their phase."""
    magnitude = np.abs(eigenvalues)
    return np.where(magnitude > 1.0, eigenvalues / magnitude, eigenvalues)


def eigendynamics(reduced: np.ndarray, stabilize: bool) -> tuple[np.ndarray, np.ndarray, int]:
    """Eigenvalues and eigenvectors of the reduced operator A~ (with
    `stabilize`, growing eigenvalues moved onto the unit circle), and how
    many eigenvalues were outside it before that."""
    eigenvalues, eigenvectors = np.linalg.eig(reduced)
    n_unstable = int(np.sum(np.abs(eigenvalues) > 1.0))
    if stabilize:
        eigenvalues = stabilize_eigenvalues(eigenvalues)
    return eigenvalues, eigenvectors, n_unstable


def pair_indices(lengths: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Row indices (into the stacked frames of datasets with these lengths)
    of the "before" and "after" frame of every snapshot pair, never pairing
    the last frame of one dataset with the first of the next."""
    before = []
    offset = 0
    for length in lengths:
        before.append(np.arange(offset, offset + length - 1))
        offset += length
    before = np.concatenate(before)
    return before, before + 1


class DMD:
    name = "dmd"
    window = 1

    def __init__(
        self,
        rank: int = DEFAULT_RANK,
        stabilize: bool = True,
        chunk_t: int = DEFAULT_CHUNK_T,
        column_block: int = DEFAULT_COLUMN_BLOCK,
    ) -> None:
        self.rank = rank
        self.stabilize = stabilize
        self.chunk_t = chunk_t
        self.column_block = column_block
        self.mean: np.ndarray | None = None  # (C, Nx, Ny)
        self.scale: np.ndarray | None = None  # (C,)
        self.basis: np.ndarray | None = None  # (state, r), POD modes U
        self.eigenvalues: np.ndarray | None = None  # (r,) complex
        self.eigenvectors: np.ndarray | None = None  # (r, r) complex, W
        self.fit_info: dict[str, float] = {}
        self.datasets: list[str] = []

    # -- fitting -------------------------------------------------------------

    def fit(self, datasets: Mapping[str, Any]) -> None:
        frames, lengths = self._load(datasets)
        before, after = pair_indices(lengths)

        gram = gram_matrix(frames, self.column_block)
        s2, s, v = truncated_pod(gram[np.ix_(before, before)], self.rank, self.name)
        r = len(s)

        # A~ = S^-1 V^T (X^T Y) V S^-1, with X^T Y a block of the Gram matrix.
        reduced = (v / s).T @ gram[np.ix_(before, after)] @ (v / s)
        eigenvalues, eigenvectors, n_unstable = eigendynamics(reduced, self.stabilize)

        self.basis = project_basis(frames, before, v / s, self.column_block)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.datasets = list(datasets)
        self.fit_info = {
            "rank": float(r),
            "explained_variance": float(s2[:r].sum() / s2.sum()),
            "unstable_modes": float(n_unstable),
            "n_pairs": float(len(before)),
        }
        log.info("dmd: %s", ", ".join(f"{k}={v:.4g}" for k, v in self.fit_info.items()))

    def _load(self, datasets: Mapping[str, Any]) -> tuple[np.ndarray, list[int]]:
        """All frames as normalized snapshots, (frames, state) float32 in
        dataset order, and the datasets' lengths; fits the normalization."""
        lengths = [array.shape[0] for array in datasets.values()]
        if not lengths or min(lengths) < 2:
            raise ValueError("DMD.fit needs datasets with at least 2 frames each")
        frame_shape = tuple(next(iter(datasets.values())).shape[1:])
        frames = load_frames(datasets, self.chunk_t, self.name)
        self.mean, self.scale = normalize_in_place(frames, frame_shape, self.chunk_t)
        return frames, lengths

    # -- forecasting ---------------------------------------------------------

    def predict(self, context: np.ndarray, n_steps: int) -> np.ndarray:
        return self._frames(self._amplitudes(context), 1, n_steps)

    def predict_blocks(
        self, context: np.ndarray, n_steps: int, block_steps: int
    ) -> Iterator[np.ndarray]:
        """`predict`'s forecast, `block_steps` frames at a time: every block
        evolves the context's mode amplitudes, so the blocks are one
        trajectory. Restarted from its own last frames instead, the model
        would project them again, which for Hankel DMD isn't the state it
        evolved (see `hankel_dmd.py`)."""
        amplitudes = self._amplitudes(context)
        for start in range(0, n_steps, block_steps):
            yield self._frames(amplitudes, start + 1, min(start + block_steps, n_steps))

    def _amplitudes(self, context: np.ndarray) -> np.ndarray:
        """The mode amplitudes of the state the forecast starts from: the
        newest context frame projected onto the POD basis."""
        if self.basis is None:
            raise RuntimeError("DMD is not fitted")
        z0 = ((context[-1] - self.mean) / self.scale[:, None, None]).reshape(-1)
        return np.linalg.solve(self.eigenvectors, self.basis.T @ z0)

    def _frames(self, amplitudes: np.ndarray, first: int, last: int) -> np.ndarray:
        """Forecast steps `first` to `last` (1: the step after the context)."""
        powers = self.eigenvalues[None, :] ** np.arange(first, last + 1)[:, None]
        reduced = ((powers * amplitudes[None, :]) @ self.eigenvectors.T).real.astype(np.float32)
        frames = (reduced @ self.basis.T).reshape(-1, *self.mean.shape)
        return frames * self.scale[None, :, None, None] + self.mean[None]

    def parameter_count(self) -> int:
        """The POD modes and the reduced operator's eigenvalues and
        eigenvectors (complex: two numbers each); the mean field and the
        channel scales are normalization."""
        if self.basis is None:
            raise RuntimeError("DMD is not fitted")
        return sum(real_size(a) for a in (self.basis, self.eigenvalues, self.eigenvectors))

    # -- checkpoint ----------------------------------------------------------

    def save(self, directory: Path) -> None:
        if self.basis is None:
            raise RuntimeError("DMD is not fitted")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "dmd.npz",
            mean=self.mean,
            scale=self.scale,
            basis=self.basis,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
        )
        meta = {
            "name": self.name,
            "rank": self.rank,
            "stabilize": self.stabilize,
            "chunk_t": self.chunk_t,
            "column_block": self.column_block,
            "datasets": self.datasets,
            "fit_info": self.fit_info,
        }
        (directory / CHECKPOINT_META).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> Self:
        directory = Path(directory)
        meta = json.loads((directory / CHECKPOINT_META).read_text())
        model = cls(meta["rank"], meta["stabilize"], meta["chunk_t"], meta["column_block"])
        with np.load(directory / "dmd.npz") as arrays:
            model.mean = arrays["mean"]
            model.scale = arrays["scale"]
            model.basis = arrays["basis"]
            model.eigenvalues = arrays["eigenvalues"]
            model.eigenvectors = arrays["eigenvectors"]
        model.datasets = meta["datasets"]
        model.fit_info = meta["fit_info"]
        return model
