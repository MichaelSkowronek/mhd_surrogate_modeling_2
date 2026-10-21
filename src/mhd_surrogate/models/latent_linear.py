"""Linear dynamics in a codec's latent space: a vector autoregression.

The simplest latent model (`latent.py`) and the reference for the plumbing:
the next latent is a linear function of the last `window` ones,

    a_{t+1} - mu = sum_{k < window} A_k^T (a_{t-k} - mu),

with mu the training latents' mean, fitted by least squares on every pair
of a window and the latent after it within a dataset (never across a
dataset boundary), optionally with a ridge penalty. On a POD codec with
`window=1` it is DMD without the eigenvalue stabilization, in the codec's
basis instead of DMD's (the POD of all frames rather than the "before"
frames); with more, it is the latent analog of Hankel DMD without the delay
truncation.

It is closed-form and has no stabilization, so it doesn't stand much
chance as a forecaster; `fit.spectral_radius` (the largest eigenvalue
magnitude of the fitted transition, above 1 if some direction grows) shows
whether its forecast will blow up. Its job is to exercise the latent-model
path -- a codec referenced by id, encoded training data, a checkpoint
carrying the codec, streaming -- end to end on the real data.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from mhd_surrogate.models.latent import LatentSurrogate

log = logging.getLogger(__name__)


def window_pairs(latents: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """(features, targets) of one dataset's flattened latents (T, r): every
    window of `window` consecutive latents, newest first, as one row of
    `window * r` features, and the latent after it."""
    n = len(latents) - window
    features = np.concatenate(
        [latents[window - 1 - k : window - 1 - k + n] for k in range(window)], 1
    )
    return features, latents[window:]


def transition_matrix(operator: np.ndarray, window: int) -> np.ndarray:
    """The (window r x window r) matrix advancing a window state (newest
    first, as a row vector) by one step: the operator's prediction becomes
    the newest block, the others shift one block older."""
    size, r = operator.shape
    transition = np.zeros((size, size))
    transition[:, :r] = operator
    transition[np.arange(size - r), np.arange(r, size)] = 1.0
    return transition


class LatentLinear(LatentSurrogate):
    name = "latent_linear"

    def __init__(self, codec: str, window: int = 1, ridge: float = 0.0) -> None:
        super().__init__(codec, window)
        self.ridge = ridge
        self.latent_mean: np.ndarray | None = None  # (r,)
        self.operator: np.ndarray | None = None  # (window * r, r), the A_k stacked

    def fit_latents(self, latents: Mapping[str, np.ndarray]) -> None:
        flat = [np.asarray(z, dtype=np.float64).reshape(len(z), -1) for z in latents.values()]
        short = [len(z) for z in flat if len(z) <= self.window]
        if short or not flat:
            raise ValueError(
                f"LatentLinear.fit needs datasets of more than window={self.window} frames"
            )
        self.latent_mean = np.concatenate(flat).mean(axis=0)
        pairs = [window_pairs(z - self.latent_mean, self.window) for z in flat]
        features = np.concatenate([f for f, _ in pairs])
        targets = np.concatenate([t for _, t in pairs])
        if self.ridge > 0:
            # Ridge as least squares on rows of sqrt(lambda) I appended, with
            # lambda relative to the features' mean square, so `ridge` means
            # the same whatever the latents' scale.
            penalty = np.sqrt(self.ridge * np.mean(features**2)) * np.eye(features.shape[1])
            features = np.concatenate([features, penalty])
            targets = np.concatenate([targets, np.zeros((len(penalty), targets.shape[1]))])
        self.operator, *_ = np.linalg.lstsq(features, targets, rcond=None)
        eigenvalues = np.linalg.eigvals(transition_matrix(self.operator, self.window))
        self.fit_info = {
            "n_pairs": float(sum(len(t) for _, t in pairs)),
            "spectral_radius": float(np.abs(eigenvalues).max()),
            "unstable_modes": float(np.sum(np.abs(eigenvalues) > 1.0)),
        }
        log.info("%s: %s", self.name, ", ".join(f"{k}={v:.4g}" for k, v in self.fit_info.items()))

    def forecast_latents(self, window: np.ndarray, n_steps: int) -> np.ndarray:
        latent_shape = window.shape[1:]
        state = np.asarray(window, dtype=np.float64).reshape(self.window, -1) - self.latent_mean
        state = state[::-1].reshape(-1)  # newest first
        r = len(self.latent_mean)
        forecast = np.empty((n_steps, r))
        for step in range(n_steps):
            forecast[step] = state @ self.operator
            state = np.concatenate([forecast[step], state[: len(state) - r]])
        return (forecast + self.latent_mean).astype(np.float32).reshape(n_steps, *latent_shape)

    def latent_parameter_count(self) -> int:
        """The operator; the latent mean is a centering, like a model's
        normalization."""
        return self.operator.size

    def config(self) -> dict[str, Any]:
        return {"ridge": self.ridge}

    def arrays(self) -> dict[str, np.ndarray]:
        return {"latent_mean": self.latent_mean, "operator": self.operator}

    def set_arrays(self, arrays: Mapping[str, np.ndarray]) -> None:
        self.latent_mean = arrays["latent_mean"]
        self.operator = arrays["operator"]
