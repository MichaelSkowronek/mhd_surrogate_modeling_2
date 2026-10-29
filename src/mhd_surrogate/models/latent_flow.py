"""Flow matching in a codec's latent space: the first generative model.

The project's cheap probabilistic baseline, there to put the probabilistic
pipeline (ensembles, the CRPS, the ensemble guardrails, seeded serving) to
work on a real generative model. The next latent given the last `window`
ones is sampled by flow matching (`flow.py`) with a residual MLP as the
velocity field, and the forecast rolls out autoregressively, one sample per
step. It is a `StochasticLatentSurrogate` (`latent.py`): raw frames in and
out, one forecast per seed, trained like the neural models by
`training/trainer.py`, early-stopping on the validation selection score of
`evaluation.monitor_members` members.

What is sampled, in which units:
- The next step's change in each latent mode, scaled to unit mean square
  over the training pairs: d = (a_{t+1} - a_t) / s per mode. The change,
  not the latent, as the U-Net's residual update: a step changes the state
  little, and d = 0 is persistence. Per mode, because POD modes' variances
  fall off by orders of magnitude: unscaled, the loss would be the leading
  modes' alone, and the trailing ones (the small scales) would be sampled
  from whatever the network happened to do with them.
- Given the last `window` latents standardized per mode (the training
  latents' mean and std): the network sees them with the point x_t on the
  path, and the flow time t through Fourier features that scale and shift
  every block (FiLM).
- The state carried from step to step, and from block to block when
  streamed (`StochasticLatentSurrogate.predict_blocks`), is the latents
  themselves, float32, standardized afresh for every step's network call:
  a forecast streamed in blocks is one call's forecast exactly, not up to
  the rounding of a round trip through the standardization.
- Each step's noise is drawn from the member's seed and the step's index
  (`flow.step_noise`), so a member is the same streamed, alone or in a
  batch (`predict_members`, one vectorized call for all of them).

The network (`VelocityMLP`): a linear layer from the context and x_t to
`width` features, `depth` residual blocks (LayerNorm, FiLM from the time
embedding, two linear layers with a GELU between), and a LayerNorm and a
linear head back to the latent size. The head is zero-initialized, as the
U-Net's: an untrained network's velocity is zero, so its samples are the
noise itself, each mode stepping like a random walk of its own step size.
Its matmuls run in full float32 on the GPU too, not the GPU's default
TF32, which rounds a batch of members differently from each one alone: a
member sampled in a batch would drift away from the same member alone
within a few steps. Alone, float32 costs nothing (sampling a member is
memory-bound); batched, a few percent (README "Latent flow matching").

`sampling_steps` Euler steps per forecast step (8 by default) is a
sampling setting, not a trained one: the same network is scored at any
step count. With 1 the forecast is the conditional mean, as far as the
network learned it (see `flow.py`): a near-deterministic ablation.

Input noise (`input_noise_std`, in each mode's step RMS; 0 = off): the
U-Net's recipe (`neural.py`, Sanchez-Gonzalez et al. 2020), since flow
matching trains on true contexts only, while a forecast conditions on its
own samples. Gaussian noise perturbs the context latents of every training
pair, and the target is the change from the perturbed newest latent to the
true next one, so the network learns to undo a perturbation. Scored
against the true change instead, it would learn to ignore it, and the
forecast, which adds its sample to the perturbed latent, would carry it
on. Step RMS, not std, because the leading modes are slow (one std is
several of their steps): in std units they would be perturbed by many
steps, and the target would ask for the whole correction in one (README
"Latent flow matching").
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mhd_surrogate.models.base import FitHooks
from mhd_surrogate.models.flow import euler_sample, flow_matching_loss, step_noise
from mhd_surrogate.models.latent import StochasticLatentSurrogate
from mhd_surrogate.training.trainer import TrainerConfig, WindowSampler, train

log = logging.getLogger(__name__)

NETWORK_FILE = "network.eqx"
# The flow time's Fourier features: sin and cos of pi * 2**k * t, k < 8
# (periods down to 1/64 of the path), embedded in TIME_WIDTH features.
TIME_FREQUENCIES = 8
TIME_WIDTH = 256


def time_features(t: jax.Array) -> jax.Array:
    angles = t * jnp.pi * 2.0 ** jnp.arange(TIME_FREQUENCIES)
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)])


def _zero_linear(layer: eqx.nn.Linear) -> eqx.nn.Linear:
    return eqx.tree_at(
        lambda lin: (lin.weight, lin.bias),
        layer,
        (jnp.zeros_like(layer.weight), jnp.zeros_like(layer.bias)),
    )


class ResidualBlock(eqx.Module):
    """h + W2 gelu(W1 (LayerNorm(h) (1 + scale) + shift)), with the scale
    and shift (FiLM) a linear function of the time embedding."""

    norm: eqx.nn.LayerNorm
    film: eqx.nn.Linear
    hidden: eqx.nn.Linear
    output: eqx.nn.Linear

    def __init__(self, width: int, *, key: jax.Array) -> None:
        film_key, hidden_key, output_key = jax.random.split(key, 3)
        self.norm = eqx.nn.LayerNorm(width, use_weight=False, use_bias=False)
        self.film = eqx.nn.Linear(TIME_WIDTH, 2 * width, key=film_key)
        self.hidden = eqx.nn.Linear(width, width, key=hidden_key)
        self.output = eqx.nn.Linear(width, width, key=output_key)

    def __call__(self, h: jax.Array, embedding: jax.Array) -> jax.Array:
        scale, shift = jnp.split(self.film(embedding), 2)
        y = self.norm(h) * (1.0 + scale) + shift
        return h + self.output(jax.nn.gelu(self.hidden(y)))


class VelocityMLP(eqx.Module):
    """The velocity field v(x, t, condition) on one sample: x (size,), t a
    scalar, condition (window, size)."""

    inputs: eqx.nn.Linear
    time: eqx.nn.MLP
    blocks: list[ResidualBlock]
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear

    def __init__(self, size: int, window: int, width: int, depth: int, *, key: jax.Array) -> None:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        keys = jax.random.split(key, depth + 3)
        self.inputs = eqx.nn.Linear((window + 1) * size, width, key=keys[0])
        self.time = eqx.nn.MLP(
            2 * TIME_FREQUENCIES,
            TIME_WIDTH,
            TIME_WIDTH,
            depth=1,
            activation=jax.nn.gelu,
            final_activation=jax.nn.gelu,
            key=keys[1],
        )
        self.blocks = [ResidualBlock(width, key=k) for k in keys[2:-1]]
        self.norm = eqx.nn.LayerNorm(width)
        self.head = _zero_linear(eqx.nn.Linear(width, size, key=keys[-1]))

    def __call__(self, x: jax.Array, t: jax.Array, condition: jax.Array) -> jax.Array:
        with jax.default_matmul_precision("float32"):
            h = self.inputs(jnp.concatenate([condition.reshape(-1), x]))
            embedding = self.time(time_features(t))
            for block in self.blocks:
                h = block(h, embedding)
            return self.head(self.norm(h))


class LatentScales(eqx.Module):
    """Per-mode training statistics: the latents' mean and std (the
    context's standardization) and the root mean square of their one-step
    change (the sampled change's unit)."""

    mean: jax.Array
    std: jax.Array
    step_rms: jax.Array

    def standardize(self, latents: jax.Array) -> jax.Array:
        return (latents - self.mean) / self.std


def fit_scales(arrays: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, std, step_rms) of `LatentScales` from the training datasets'
    latents, each (T, r); one-step changes only within a dataset. Summed in
    float64."""
    latents = np.concatenate(arrays).astype(np.float64)
    changes = np.concatenate([np.diff(a.astype(np.float64), axis=0) for a in arrays])
    mean, std = latents.mean(axis=0), latents.std(axis=0)
    step_rms = np.sqrt(np.mean(changes**2, axis=0))
    constant = np.flatnonzero((std == 0) | (step_rms == 0))
    if constant.size:
        raise ValueError(f"latent modes {constant.tolist()} don't vary over the training data")
    return mean.astype(np.float32), std.astype(np.float32), step_rms.astype(np.float32)


def latent_flow_loss(
    network,
    scales: LatentScales,
    batch: jax.Array,
    window: int,
    key: jax.Array,
    input_noise_std: float = 0.0,
) -> jax.Array:
    """The flow-matching loss (`flow.flow_matching_loss`) on a batch of
    `window + 1` consecutive latents, (B, window + 1, r): the first `window`
    standardized are the context, the change from the newest of them to the
    last latent in units of its RMS the target. With `input_noise_std` above
    0 the context latents are perturbed first, by Gaussian noise of that
    many step RMSs per mode (drawn from `key` like the rest), and the change
    is the one from the perturbed newest latent."""
    flow_key, noise_key = jax.random.split(key)
    context = batch[:, :window]
    if input_noise_std > 0:
        noise = jax.random.normal(noise_key, context.shape)
        context = context + input_noise_std * scales.step_rms * noise
    target = (batch[:, window] - context[:, -1]) / scales.step_rms
    return flow_matching_loss(network, scales.standardize(context), target, flow_key)


@eqx.filter_jit
def _sample(network, scales, window, keys, first_step, n_steps, sampling_steps):
    """Each key's forecast steps `first_step` onward from `window` (window,
    r), shape (len(keys), n_steps, r)."""

    def member(key):
        def advance(window, step):
            noise = step_noise(key, step, window.shape[1:])
            change = euler_sample(network, scales.standardize(window), noise, sampling_steps)
            latent = window[-1] + scales.step_rms * change
            return jnp.concatenate([window[1:], latent[None]]), latent

        _, latents = jax.lax.scan(advance, window, first_step + jnp.arange(n_steps))
        return latents

    return jax.vmap(member)(keys)


class LatentFlow(StochasticLatentSurrogate):
    name = "latent_flow"
    iterative = True

    def __init__(
        self,
        codec: str,
        window: int = 4,
        width: int = 1024,
        depth: int = 2,
        sampling_steps: int = 8,
        input_noise_std: float = 0.0,
        seed: int = 0,
        training: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(codec, window)
        for name, value in (("width", width), ("depth", depth), ("sampling_steps", sampling_steps)):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if not input_noise_std >= 0:
            raise ValueError(f"input_noise_std must be >= 0, got {input_noise_std}")
        self.width = width
        self.depth = depth
        self.sampling_steps = sampling_steps
        self.input_noise_std = input_noise_std
        self.seed = seed
        self.training = dict(training or {})
        self.trainer_config = TrainerConfig(seed=seed, **self.training)
        self.network: VelocityMLP | None = None
        self.latent_mean: np.ndarray | None = None  # (r,)
        self.latent_std: np.ndarray | None = None  # (r,)
        self.step_rms: np.ndarray | None = None  # (r,)

    @property
    def requirements(self) -> tuple[str, ...]:
        # optax through the training module this one imports, as for neural.py.
        return ("equinox", "optax", *super().requirements)

    def _new_network(self, size: int) -> VelocityMLP:
        return VelocityMLP(size, self.window, self.width, self.depth, key=jax.random.key(self.seed))

    def _scales(self) -> LatentScales:
        return LatentScales(
            jnp.asarray(self.latent_mean), jnp.asarray(self.latent_std), jnp.asarray(self.step_rms)
        )

    # -- the latent dynamics ---------------------------------------------------

    def fit_latents(self, latents: Mapping[str, np.ndarray], hooks: FitHooks | None = None) -> None:
        hooks = hooks or FitHooks()
        arrays = [np.asarray(z, dtype=np.float32).reshape(len(z), -1) for z in latents.values()]
        if not arrays or min(len(a) for a in arrays) <= self.window:
            raise ValueError(
                f"LatentFlow.fit needs datasets of more than window={self.window} frames"
            )
        self.latent_mean, self.latent_std, self.step_rms = fit_scales(arrays)
        self.network = self._new_network(arrays[0].shape[1])
        log.info("%s: %d parameters", self.name, self.latent_parameter_count())

        scales, window, noise_std = self._scales(), self.window, self.input_noise_std

        def loss_fn(network, batch, key):
            return latent_flow_loss(network, scales, batch, window, key, noise_std)

        validate = None
        if hooks.validate is not None:

            def validate(network):
                self.network = network
                return hooks.validate(self)

        result = train(
            self.network,
            loss_fn,
            WindowSampler(arrays, window + 1),
            self.trainer_config,
            hooks,
            validate,
            fingerprint={
                "name": self.name,
                "codec": self.codec_reference,
                "window": self.window,
                **self.config(),
            },
        )
        self.network = result.network
        self.fit_info = result.info

    def sample_latents(
        self, window: np.ndarray, n_steps: int, seeds: Sequence[int], first_step: int = 0
    ) -> np.ndarray:
        if self.network is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        keys = jnp.stack([jax.random.key(seed) for seed in seeds])
        latents = _sample(
            self.network,
            self._scales(),
            jnp.asarray(window, dtype=jnp.float32).reshape(self.window, -1),
            keys,
            jnp.asarray(first_step),
            n_steps,
            self.sampling_steps,
        )
        return np.asarray(latents).reshape(len(seeds), n_steps, *window.shape[1:])

    def latent_parameter_count(self) -> int:
        """The network's weights; the latent scales are normalization."""
        return sum(x.size for x in jax.tree.leaves(eqx.filter(self.network, eqx.is_array)))

    # -- checkpoint ------------------------------------------------------------

    def config(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "depth": self.depth,
            "sampling_steps": self.sampling_steps,
            "input_noise_std": self.input_noise_std,
            "seed": self.seed,
            "training": self.training,
        }

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "latent_mean": self.latent_mean,
            "latent_std": self.latent_std,
            "step_rms": self.step_rms,
        }

    def set_arrays(self, arrays: Mapping[str, np.ndarray]) -> None:
        self.latent_mean = arrays["latent_mean"]
        self.latent_std = arrays["latent_std"]
        self.step_rms = arrays["step_rms"]

    def save(self, directory: Path) -> None:
        super().save(directory)
        eqx.tree_serialise_leaves(Path(directory) / NETWORK_FILE, self.network)

    @classmethod
    def load(cls, directory: Path) -> Self:
        model = super().load(directory)
        skeleton = model._new_network(len(model.latent_mean))
        model.network = eqx.tree_deserialise_leaves(Path(directory) / NETWORK_FILE, skeleton)
        return model
