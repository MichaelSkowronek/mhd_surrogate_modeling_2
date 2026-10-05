"""Dynamic Mode Decomposition as a forecasting model.

DMD fits the best linear operator A (least squares, z_{t+1} ~= A z_t) to the
training snapshots and forecasts by applying it repeatedly. It is the
simplest model with dynamics: between the baselines (none) and a nonlinear
network. Being linear, it can carry the flow's coherent oscillation forward,
but not the turbulence: decaying modes die out (the forecast relaxes to the
mean) and the rank truncation removes the small scales.

State. Each frame is turned into a fluctuation about the pooled training
mean field, scaled per channel by the fluctuation's training RMS, so u_x and
u_y weigh the same in the SVD (u_y's std is about half of u_x's); channels
and space are flattened into one vector. Both the mean and the scale are
fitted here: normalization is the first step of fitting a model.

Fit (projected exact DMD over several trajectories, method of snapshots).
Snapshot pairs (z_t, z_{t+1}) are formed within each dataset only, never
across the boundary between two datasets. With X the "before" snapshots and
Y the "after" ones (one column per pair), the Gram matrix G = Z^T Z of all
frames gives X^T X = V S^2 V^T (POD basis U = X V S^-1, truncated to `rank`)
and the reduced operator A~ = U^T A U = S^-1 V^T (X^T Y) V S^-1, without
ever forming the (state x state) operator or holding X in a factorized form.
The state has ~292k entries and there are ~7k frames, so G is small; it is
computed on JAX's default device (the GPU with the `gpu` extra) in blocks of
state columns, so every frame crosses to the device once per pass.

Forecast. The last context frame is projected onto U, evolved with the
eigendecomposition A~ = W diag(lambda) W^-1, and mapped back. With
`stabilize`, eigenvalues outside the unit circle are moved onto it (same
frequency, no growth): a single growing mode would otherwise blow the
forecast up over hundreds of steps.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import jax
import jax.numpy as jnp
import numpy as np

from mhd_surrogate.models.base import CHECKPOINT_META

log = logging.getLogger(__name__)

DEFAULT_RANK = 100
DEFAULT_CHUNK_T = 64
DEFAULT_COLUMN_BLOCK = 32768
# Singular values below this fraction of the largest are numerically zero;
# keeping them would divide by ~0 in S^-1. The Gram matrix squares the
# singular values, so with float32 frames it resolves them only down to
# ~sqrt(float32 eps) ~ 3e-4 of the largest; 1e-3 stays clear of that floor.
RELATIVE_TOLERANCE = 1e-3


@jax.jit
def _gram(block: jax.Array) -> jax.Array:
    # HIGHEST: full float32 on GPUs that would otherwise use TF32 for matmuls.
    return jnp.matmul(block, block.T, precision=jax.lax.Precision.HIGHEST)


@jax.jit
def _project(block: jax.Array, coefficients: jax.Array) -> jax.Array:
    return jnp.matmul(block.T, coefficients, precision=jax.lax.Precision.HIGHEST)


def stabilize_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
    """Move eigenvalues outside the unit circle onto it, keeping their phase."""
    magnitude = np.abs(eigenvalues)
    return np.where(magnitude > 1.0, eigenvalues / magnitude, eigenvalues)


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
        self._normalize_in_place(frames)
        before, after = pair_indices(lengths)

        gram = self._gram(frames)
        s2, v = np.linalg.eigh(gram[np.ix_(before, before)])
        order = np.argsort(s2)[::-1]
        s2, v = s2[order], v[:, order]
        usable = int(np.sum(s2 > RELATIVE_TOLERANCE**2 * s2[0]))
        r = min(self.rank, usable)
        if r < self.rank:
            log.warning("dmd: rank %d requested, only %d usable singular values", self.rank, r)
        s, v = np.sqrt(s2[:r]), v[:, :r]

        # A~ = S^-1 V^T (X^T Y) V S^-1, with X^T Y a block of the Gram matrix.
        reduced = (v / s).T @ gram[np.ix_(before, after)] @ (v / s)
        eigenvalues, eigenvectors = np.linalg.eig(reduced)
        n_unstable = int(np.sum(np.abs(eigenvalues) > 1.0))
        if self.stabilize:
            eigenvalues = stabilize_eigenvalues(eigenvalues)

        self.basis = self._basis(frames, before, v / s)
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
        """All frames stacked as (frames, state) float32, in dataset order."""
        lengths = [array.shape[0] for array in datasets.values()]
        if not lengths or min(lengths) < 2:
            raise ValueError("DMD.fit needs datasets with at least 2 frames each")
        first = next(iter(datasets.values()))
        self._frame_shape = tuple(first.shape[1:])
        frames = np.empty((sum(lengths), int(np.prod(self._frame_shape))), dtype=np.float32)
        row = 0
        for name, array in datasets.items():
            for start in range(0, array.shape[0], self.chunk_t):
                block = np.asarray(array[start : start + self.chunk_t], dtype=np.float32)
                frames[row : row + len(block)] = block.reshape(len(block), -1)
                row += len(block)
            log.info("dmd: loaded %s (%d frames so far)", name, row)
        return frames, lengths

    def _normalize_in_place(self, frames: np.ndarray) -> None:
        """Subtract the per-pixel mean and divide each channel by the RMS of
        its fluctuation, both pooled over all frames."""
        n_channels = self._frame_shape[0]
        mean = frames.mean(axis=0, dtype=np.float64)
        frames -= mean.astype(np.float32)
        per_channel = frames.reshape(len(frames), n_channels, -1)
        # Sum of squares a chunk of frames at a time: a float64 square of the
        # whole array would double the ~8 GB the frames already take.
        squares = np.zeros(n_channels)
        for start in range(0, len(frames), self.chunk_t):
            chunk = per_channel[start : start + self.chunk_t].astype(np.float64)
            squares += np.square(chunk).sum(axis=(0, 2))
        scale = np.sqrt(squares / (len(frames) * per_channel.shape[2]))
        per_channel /= scale.astype(np.float32)[None, :, None]
        self.mean = mean.reshape(self._frame_shape).astype(np.float32)
        self.scale = scale.astype(np.float32)

    def _column_blocks(self, frames: np.ndarray):
        """(start, width, block) for every `column_block`-wide slice of the
        state, the last one zero-padded to full width: zero columns add
        nothing to a Gram matrix or a projection, and one block shape means
        one XLA compile (with GEMM autotuning, ~45 s on this GPU) not two."""
        for start in range(0, frames.shape[1], self.column_block):
            block = frames[:, start : start + self.column_block]
            width = block.shape[1]
            if width < self.column_block and frames.shape[1] > self.column_block:
                block = np.pad(block, ((0, 0), (0, self.column_block - width)))
            yield start, width, np.ascontiguousarray(block)

    def _gram(self, frames: np.ndarray) -> np.ndarray:
        gram = np.zeros((len(frames), len(frames)), dtype=np.float64)
        for _, _, block in self._column_blocks(frames):
            gram += np.asarray(_gram(block), dtype=np.float64)
        return gram

    def _basis(
        self, frames: np.ndarray, before: np.ndarray, coefficients: np.ndarray
    ) -> np.ndarray:
        """U = X V S^-1, one block of state rows at a time."""
        weights = np.zeros((len(frames), coefficients.shape[1]), dtype=np.float32)
        weights[before] = coefficients
        basis = np.empty((frames.shape[1], coefficients.shape[1]), dtype=np.float32)
        for start, width, block in self._column_blocks(frames):
            basis[start : start + width] = np.asarray(_project(block, weights))[:width]
        return basis

    # -- forecasting ---------------------------------------------------------

    def predict(self, context: np.ndarray, n_steps: int) -> np.ndarray:
        if self.basis is None:
            raise RuntimeError("DMD is not fitted")
        frame_shape = self.mean.shape
        z0 = ((context[-1] - self.mean) / self.scale[:, None, None]).reshape(-1)
        amplitudes = np.linalg.solve(self.eigenvectors, self.basis.T @ z0)
        powers = self.eigenvalues[None, :] ** np.arange(1, n_steps + 1)[:, None]
        reduced = ((powers * amplitudes[None, :]) @ self.eigenvectors.T).real.astype(np.float32)
        frames = (reduced @ self.basis.T).reshape(n_steps, *frame_shape)
        return frames * self.scale[None, :, None, None] + self.mean[None]

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
