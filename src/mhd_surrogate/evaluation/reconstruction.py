"""How well a codec reconstructs a dataset.

A codec (`models/codec.py`) forecasts nothing, so it is scored on its
reconstruction decode(encode(x)) of every frame of a dataset -- no context,
no lead times:

- `reconstruction_rmse`: the RMSE over all frames, channels and points, in
  units of the training std per channel, as a forecast's RMSE. What an
  iteratively trained codec early-stops on.
- the long-horizon physics scores a forecast is checked with
  (`diagnostics.compare_diagnostics`): energy and enstrophy errors,
  divergence ratio, spatial spectrum distances, and the domain-mean
  velocity's temporal spectrum and drift. A latent model forecasts in the
  codec's latent space, so what the reconstruction loses its forecasts
  lose too (the small scales a truncated basis drops, say): these show what
  a codec costs before any latent model is trained on it. Reported, not
  gated -- a latent model is held to the forecast guardrails itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mhd_surrogate.evaluation.diagnostics import DEFAULT_CHUNK_T, compare_diagnostics
from mhd_surrogate.evaluation.metrics import rmse_per_step


def reconstruct(codec, series: Any, chunk_t: int = DEFAULT_CHUNK_T) -> np.ndarray:
    """decode(encode(x)) of every frame of `series` (a NumPy or zarr array,
    (T, C, Nx, Ny)), `chunk_t` frames at a time."""
    reconstruction = np.empty(series.shape, dtype=np.float32)
    for start in range(0, series.shape[0], chunk_t):
        block = np.asarray(series[start : start + chunk_t], dtype=np.float32)
        reconstruction[start : start + len(block)] = codec.decode(codec.encode(block))
    return reconstruction


def reconstruction_scores(
    codec, series: Any, scale: np.ndarray, dx: float, dy: float, chunk_t: int = DEFAULT_CHUNK_T
) -> dict[str, float]:
    """`reconstruction_rmse` (in units of the per-channel `scale`) and the
    `compare_diagnostics` scores of `codec`'s reconstruction of `series`."""
    truth = np.asarray(series, dtype=np.float32)
    reconstruction = reconstruct(codec, truth, chunk_t)
    squared = rmse_per_step(reconstruction, truth, scale) ** 2
    return {
        "reconstruction_rmse": float(np.sqrt(squared.mean())),
        **compare_diagnostics(reconstruction, truth, dx, dy, chunk_t),
    }
