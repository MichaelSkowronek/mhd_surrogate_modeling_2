"""The training frames as a snapshot matrix, and its POD.

Shared by the models built on a POD basis of the frames: DMD and Hankel DMD
(the basis their dynamics live in) and the POD codec (`pod_codec.py`, the
basis as a latent space).

Snapshots. Every frame of the training datasets is one column of the
snapshot matrix Z, held as its transpose, a (frames x state) float32 array
in dataset order. A column is a fluctuation about the pooled training mean
field, scaled per channel by the fluctuation's training RMS, so u_x and u_y
weigh the same in the POD (u_y's std is about half of u_x's); channels and
space are flattened into one vector. Both the mean and the scale are fitted
here: normalization is the first step of fitting a model.

POD (method of snapshots). The state has ~292k entries and there are ~7k
frames, so the Gram matrix G = Z^T Z is small. The POD of a set of columns
X of Z, X = U S V^T, takes S^2 and V from the eigenpairs of X^T X (a block
of G) and forms the basis U = X V S^-1 without ever factorizing X. G and U
are computed on JAX's default device (the GPU with the `gpu` extra) in
blocks of the state, so every frame crosses to the device once per pass.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

log = logging.getLogger(__name__)

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


def load_frames(datasets: Mapping[str, Any], chunk_t: int, label: str) -> np.ndarray:
    """All frames stacked as (frames, state) float32, in dataset order, read
    `chunk_t` frames at a time (zarr arrays are fine)."""
    lengths = [array.shape[0] for array in datasets.values()]
    first = next(iter(datasets.values()))
    frames = np.empty((sum(lengths), int(np.prod(first.shape[1:]))), dtype=np.float32)
    row = 0
    for name, array in datasets.items():
        for start in range(0, array.shape[0], chunk_t):
            block = np.asarray(array[start : start + chunk_t], dtype=np.float32)
            frames[row : row + len(block)] = block.reshape(len(block), -1)
            row += len(block)
        log.info("%s: loaded %s (%d frames so far)", label, name, row)
    return frames


def normalize_in_place(
    frames: np.ndarray, frame_shape: tuple[int, ...], chunk_t: int
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the per-pixel mean and divide each channel by the RMS of its
    fluctuation, both pooled over all frames; returns (mean, scale), shapes
    `frame_shape` and (C,)."""
    n_channels = frame_shape[0]
    mean = frames.mean(axis=0, dtype=np.float64)
    frames -= mean.astype(np.float32)
    per_channel = frames.reshape(len(frames), n_channels, -1)
    # Sum of squares a chunk of frames at a time: a float64 square of the
    # whole array would double the ~8 GB the frames already take.
    squares = np.zeros(n_channels)
    for start in range(0, len(frames), chunk_t):
        chunk = per_channel[start : start + chunk_t].astype(np.float64)
        squares += np.square(chunk).sum(axis=(0, 2))
    scale = np.sqrt(squares / (len(frames) * per_channel.shape[2]))
    per_channel /= scale.astype(np.float32)[None, :, None]
    return mean.reshape(frame_shape).astype(np.float32), scale.astype(np.float32)


def column_blocks(frames: np.ndarray, column_block: int) -> Iterator[tuple[int, int, np.ndarray]]:
    """(start, width, block) for every `column_block`-wide slice of the state,
    the last one zero-padded to full width: zero columns add nothing to a
    Gram matrix or a projection, and one block shape means one XLA compile
    (with GEMM autotuning, ~45 s on this GPU) not two."""
    for start in range(0, frames.shape[1], column_block):
        block = frames[:, start : start + column_block]
        width = block.shape[1]
        if width < column_block and frames.shape[1] > column_block:
            block = np.pad(block, ((0, 0), (0, column_block - width)))
        yield start, width, np.ascontiguousarray(block)


def gram_matrix(frames: np.ndarray, column_block: int) -> np.ndarray:
    """G = Z^T Z for the snapshots `frames` (one row per frame), in float64."""
    gram = np.zeros((len(frames), len(frames)), dtype=np.float64)
    for _, _, block in column_blocks(frames, column_block):
        gram += np.asarray(_gram(block), dtype=np.float64)
    return gram


def project_basis(
    frames: np.ndarray, rows: np.ndarray, coefficients: np.ndarray, column_block: int
) -> np.ndarray:
    """U = X V S^-1, with X the frames `rows` of `frames` and `coefficients`
    = V S^-1, one block of state rows at a time."""
    weights = np.zeros((len(frames), coefficients.shape[1]), dtype=np.float32)
    weights[rows] = coefficients
    basis = np.empty((frames.shape[1], coefficients.shape[1]), dtype=np.float32)
    for start, width, block in column_blocks(frames, column_block):
        basis[start : start + width] = np.asarray(_project(block, weights))[:width]
    return basis


def truncated_pod(
    gram: np.ndarray, rank: int, label: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """POD of the snapshots whose Gram matrix is `gram` (method of
    snapshots): (all squared singular values, largest first; the `rank`
    leading singular values S; their right singular vectors V). The rank is
    capped at the singular values above the numerical tolerance."""
    s2, v = np.linalg.eigh(gram)
    order = np.argsort(s2)[::-1]
    s2, v = s2[order], v[:, order]
    usable = int(np.sum(s2 > RELATIVE_TOLERANCE**2 * s2[0]))
    r = min(rank, usable)
    if r < rank:
        log.warning("%s: rank %d requested, only %d usable singular values", label, rank, r)
    return s2, np.sqrt(s2[:r]), v[:, :r]
