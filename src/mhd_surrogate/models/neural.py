"""Autoregressive neural surrogates: everything but the network.

A neural surrogate here maps the last `window` frames to the next one and
forecasts by feeding its own predictions back in. This module holds the part
every such model shares, whatever the network (U-Net now; an FNO would only
supply a different `build_network`):

- Normalization, fitted in `fit` (it's the first step of fitting a model):
  each channel's mean and std pooled over every training frame and pixel, as
  in `data/normalization.py`, so the loss is in the same per-channel units
  the forecast RMSE is scored in.
- Constant pixels: pixels whose value never changes in the training data
  (the no-slip walls, the inlet's u_y) are overwritten with that value after
  every step, so the boundary conditions hold exactly instead of drifting
  over a long rollout.
- Residual update: the network predicts the change from the newest frame
  (`next = last + network(...)`), with the newest `window` frames and the
  grid coordinates as input channels, padded to the network's multiple.
- Loss: the mean squared error of a `rollout_steps`-step autoregressive
  rollout (1 = one-step teacher forcing). Training on its own rollouts shows
  the model the errors it will compound at forecast time. Each step is
  rematerialized (`jax.checkpoint`), so memory grows with the rollout only by
  one step's activations.
- Mixed precision: with `compute_dtype: bfloat16` the network runs in bf16
  (parameters, optimizer and the residual sum stay float32), ~1.8x faster
  on the RTX 3060.
- Training frames are held in host RAM as `host_dtype` (float16: ~4 GB for
  the 7 training datasets, so two trials of a sweep fit side by side).

Training itself is `training/trainer.py`'s loop: early stopping on the
validation selection score, resumable state, divergence checks.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mhd_surrogate.models.base import CHECKPOINT_META, FitHooks
from mhd_surrogate.training.trainer import TrainerConfig, WindowSampler, train

log = logging.getLogger(__name__)

NETWORK_FILE = "network.eqx"
NORMALIZATION_FILE = "normalization.npz"
DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}
# A pixel whose training std is below this fraction of its channel's std is
# constant (walls: exactly zero; inlet u_y: zero up to float rounding).
CONSTANT_TOLERANCE = 1e-6


def padded_size(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def pad_to_multiple(x: jax.Array, multiple: int) -> jax.Array:
    """Pad the last two axes at their far end up to a multiple of `multiple`,
    repeating the edge values (at a no-slip wall that's its zero velocity;
    at the outlet, the last column)."""
    h, w = x.shape[-2:]
    pad = [(0, 0)] * (x.ndim - 2) + [
        (0, padded_size(h, multiple) - h),
        (0, padded_size(w, multiple) - w),
    ]
    return jnp.pad(x, pad, mode="edge")


def crop(x: jax.Array, shape: tuple[int, int]) -> jax.Array:
    """Undo `pad_to_multiple`: keep the first `shape` entries of the last two axes."""
    return x[..., : shape[0], : shape[1]]


def coordinate_channels(shape: tuple[int, int]) -> jax.Array:
    """(2, H, W) grid coordinates, each scaled to [-1, 1].

    Convolutions are translation-equivariant, but this flow isn't
    homogeneous (an inlet, an outlet, two walls): concatenated to the input,
    the coordinates let the network tell where it is ("CoordConv").
    """
    h, w = shape
    x, y = jnp.meshgrid(jnp.linspace(-1.0, 1.0, h), jnp.linspace(-1.0, 1.0, w), indexing="ij")
    return jnp.stack([x, y])


def cast_floats(tree, dtype):
    return jax.tree.map(lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, tree)


class Frame(eqx.Module):
    """The fitted, non-learned parts of the model, in normalized units on the
    padded grid: coordinates, the constant-pixel mask and its values, and
    the mask of real (not padding) pixels the loss is averaged over."""

    coords: jax.Array  # (2, Hp, Wp)
    constant: jax.Array  # (C, Hp, Wp) bool
    constant_values: jax.Array  # (C, Hp, Wp)
    loss_weight: jax.Array  # (C, Hp, Wp): 1 on real, non-constant pixels


def step(network, frame: Frame, window: jax.Array, dtype) -> jax.Array:
    """The next normalized, padded frame from the last `window` ones,
    (window, C, Hp, Wp) -> (C, Hp, Wp)."""
    inputs = jnp.concatenate([window.reshape(-1, *window.shape[2:]), frame.coords])
    update = cast_floats(network, dtype)(inputs.astype(dtype)).astype(jnp.float32)
    return jnp.where(frame.constant, frame.constant_values, window[-1] + update)


def rollout(network, frame: Frame, window: jax.Array, n_steps: int, dtype) -> jax.Array:
    """`n_steps` frames autoregressively from `window`, (n_steps, C, Hp, Wp)."""

    def advance(window, _):
        prediction = step(network, frame, window, dtype)
        return jnp.concatenate([window[1:], prediction[None]]), prediction

    _, predictions = jax.lax.scan(advance, window, length=n_steps)
    return predictions


def rollout_loss(network, frame: Frame, batch: jax.Array, window: int, dtype) -> jax.Array:
    """Mean squared error of rolling each sequence of `batch` (B, window +
    rollout_steps, C, Hp, Wp) forward from its first `window` frames, against
    the rest; averaged over steps, real non-constant pixels and the batch."""

    def sequence_loss(sequence):
        @jax.checkpoint
        def advance(inputs, target):
            prediction = step(network, frame, inputs, dtype)
            error = jnp.sum(frame.loss_weight * (prediction - target) ** 2)
            return jnp.concatenate([inputs[1:], prediction[None]]), error

        _, errors = jax.lax.scan(advance, sequence[:window], sequence[window:])
        return errors.mean() / frame.loss_weight.sum()

    return jax.vmap(sequence_loss)(batch.astype(jnp.float32)).mean()


@eqx.filter_jit
def _forecast(network, frame, window, n_steps, dtype):
    return rollout(network, frame, window, n_steps, dtype)


class AutoregressiveSurrogate:
    """Base class; a subclass sets `name`, `pad_multiple()` and
    `build_network(in_channels, out_channels, key)`."""

    name = "autoregressive"
    iterative = True
    # Third-party packages a logged model needs to load and predict (optax
    # through the training module this one imports).
    requirements = ("equinox", "optax")

    def __init__(
        self,
        window: int = 4,
        rollout_steps: int = 1,
        compute_dtype: str = "float32",
        host_dtype: str = "float16",
        seed: int = 0,
        chunk_t: int = 64,
        training: Mapping[str, Any] | None = None,
    ) -> None:
        if window < 1 or rollout_steps < 1:
            raise ValueError(
                f"window and rollout_steps must be >= 1, got {window}, {rollout_steps}"
            )
        for dtype in (compute_dtype, host_dtype):
            if dtype not in DTYPES:
                raise ValueError(f"unknown dtype {dtype!r}, expected one of {sorted(DTYPES)}")
        self.window = window
        self.rollout_steps = rollout_steps
        self.compute_dtype = compute_dtype
        self.host_dtype = host_dtype
        self.seed = seed
        self.chunk_t = chunk_t
        self.training = dict(training or {})
        self.trainer_config = TrainerConfig(seed=seed, **self.training)
        self.network = None
        self.frame: Frame | None = None
        self.mean: np.ndarray | None = None  # (C,)
        self.std: np.ndarray | None = None  # (C,)
        self.constant: np.ndarray | None = None  # (C, H, W) bool
        self.constant_values: np.ndarray | None = None  # (C, H, W), raw units
        self.frame_shape: tuple[int, ...] | None = None
        self.fit_info: dict[str, float] = {}
        self.datasets: list[str] = []

    # -- what a subclass provides ----------------------------------------------

    def network_config(self) -> dict[str, Any]:
        return {}

    def pad_multiple(self) -> int:
        return 1

    def build_network(self, in_channels: int, out_channels: int, key: jax.Array):
        raise NotImplementedError

    # -- configuration -----------------------------------------------------------

    def config(self) -> dict[str, Any]:
        """The constructor arguments: what `load` rebuilds the model from."""
        return {
            "window": self.window,
            "rollout_steps": self.rollout_steps,
            "compute_dtype": self.compute_dtype,
            "host_dtype": self.host_dtype,
            "seed": self.seed,
            "chunk_t": self.chunk_t,
            "training": self.training,
            **self.network_config(),
        }

    def _new_network(self, n_channels: int):
        return self.build_network(
            self.window * n_channels + 2, n_channels, jax.random.key(self.seed)
        )

    # -- normalization -----------------------------------------------------------

    def normalize(self, x):
        return (x - self.mean[:, None, None]) / self.std[:, None, None]

    def denormalize(self, x):
        return x * self.std[:, None, None] + self.mean[:, None, None]

    def _fit_normalization(self, datasets: Mapping[str, Any]) -> None:
        """Per-channel mean/std over every frame and pixel, and the constant
        pixels, from per-pixel float64 sums (one streaming pass)."""
        total = squares = None
        n = 0
        for array in datasets.values():
            for start in range(0, array.shape[0], self.chunk_t):
                block = np.asarray(array[start : start + self.chunk_t], dtype=np.float64)
                total = block.sum(axis=0) + (0 if total is None else total)
                squares = (block**2).sum(axis=0) + (0 if squares is None else squares)
                n += len(block)
        pixel_mean = total / n
        pixel_var = np.maximum(squares / n - pixel_mean**2, 0.0)
        mean = pixel_mean.mean(axis=(1, 2))
        var = (pixel_var + pixel_mean**2).mean(axis=(1, 2)) - mean**2
        std = np.sqrt(var)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.constant = np.sqrt(pixel_var) <= CONSTANT_TOLERANCE * std[:, None, None]
        self.constant_values = np.where(self.constant, pixel_mean, 0.0).astype(np.float32)
        self.frame_shape = tuple(pixel_mean.shape)

    def _make_frame(self) -> Frame:
        multiple = self.pad_multiple()
        shape = self.frame_shape[1:]
        constant = jnp.asarray(self.constant)
        loss_weight = (~constant).astype(jnp.float32)
        return Frame(
            coords=pad_to_multiple(coordinate_channels(shape), multiple),
            # Padding is neither constant nor counted in the loss.
            constant=jnp.pad(constant, [(0, 0)] + _padding(shape, multiple)),
            constant_values=jnp.pad(
                self.normalize(jnp.asarray(self.constant_values)) * constant,
                [(0, 0)] + _padding(shape, multiple),
            ),
            loss_weight=jnp.pad(loss_weight, [(0, 0)] + _padding(shape, multiple)),
        )

    def _load_normalized(self, datasets: Mapping[str, Any]) -> list[np.ndarray]:
        dtype = np.dtype(self.host_dtype) if self.host_dtype != "bfloat16" else jnp.bfloat16
        frames = []
        for name, array in datasets.items():
            out = np.empty(array.shape, dtype=dtype)
            for start in range(0, array.shape[0], self.chunk_t):
                block = np.asarray(array[start : start + self.chunk_t], dtype=np.float32)
                out[start : start + len(block)] = self.normalize(block)
            frames.append(out)
            log.info("%s: loaded %s (%.2f GB in RAM)", self.name, name, out.nbytes / 1e9)
        return frames

    # -- fitting -----------------------------------------------------------------

    def fit(self, datasets: Mapping[str, Any], hooks: FitHooks | None = None) -> None:
        hooks = hooks or FitHooks()
        self._fit_normalization(datasets)
        self.frame = self._make_frame()
        frames = self._load_normalized(datasets)
        n_channels = self.frame_shape[0]
        self.network = self._new_network(n_channels)
        n_parameters = sum(x.size for x in jax.tree.leaves(eqx.filter(self.network, eqx.is_array)))
        log.info("%s: %d parameters", self.name, n_parameters)

        multiple, window, dtype = self.pad_multiple(), self.window, DTYPES[self.compute_dtype]
        frame = self.frame

        def loss_fn(network, batch):
            return rollout_loss(network, frame, pad_to_multiple(batch, multiple), window, dtype)

        validate = None
        if hooks.validate is not None:

            def validate(network):
                self.network = network
                return hooks.validate(self)

        result = train(
            self.network,
            loss_fn,
            WindowSampler(frames, self.window + self.rollout_steps),
            self.trainer_config,
            hooks,
            validate,
            fingerprint={"name": self.name, **self.config()},
        )
        self.network = result.network
        self.datasets = list(datasets)
        self.fit_info = {**result.info, "parameters": float(n_parameters)}

    # -- forecasting -------------------------------------------------------------

    def predict(self, context: np.ndarray, n_steps: int) -> np.ndarray:
        if self.network is None:
            raise RuntimeError(f"{self.name} is not fitted")
        window = pad_to_multiple(
            self.normalize(jnp.asarray(context[-self.window :], dtype=jnp.float32)),
            self.pad_multiple(),
        )
        predictions = _forecast(
            self.network, self.frame, window, n_steps, DTYPES[self.compute_dtype]
        )
        # Cropped, denormalized and the constant pixels set to exactly their
        # training value (normalizing and back is off by rounding) on the
        # device, so only the result (~1 GB for a whole validation forecast)
        # is copied to the host.
        frames = self.denormalize(crop(predictions, self.frame_shape[1:]))
        return np.asarray(jnp.where(self.constant, self.constant_values, frames))

    # -- checkpoint --------------------------------------------------------------

    def save(self, directory: Path) -> None:
        if self.network is None:
            raise RuntimeError(f"{self.name} is not fitted")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(directory / NETWORK_FILE, self.network)
        np.savez(
            directory / NORMALIZATION_FILE,
            mean=self.mean,
            std=self.std,
            constant=self.constant,
            constant_values=self.constant_values,
        )
        meta = {
            "name": self.name,
            "config": self.config(),
            "datasets": self.datasets,
            "fit_info": self.fit_info,
        }
        (directory / CHECKPOINT_META).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> Self:
        directory = Path(directory)
        meta = json.loads((directory / CHECKPOINT_META).read_text())
        model = cls(**meta["config"])
        with np.load(directory / NORMALIZATION_FILE) as arrays:
            model.mean = arrays["mean"]
            model.std = arrays["std"]
            model.constant = arrays["constant"]
            model.constant_values = arrays["constant_values"]
        model.frame_shape = model.constant.shape
        model.frame = model._make_frame()
        skeleton = model._new_network(model.frame_shape[0])
        model.network = eqx.tree_deserialise_leaves(directory / NETWORK_FILE, skeleton)
        model.datasets = meta["datasets"]
        model.fit_info = meta["fit_info"]
        return model


def _padding(shape: tuple[int, int], multiple: int) -> list[tuple[int, int]]:
    return [(0, padded_size(n, multiple) - n) for n in shape]
