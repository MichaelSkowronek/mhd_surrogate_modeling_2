"""Flow matching: the math behind the generative latent model.

Flow matching (Lipman et al. 2023, "Flow Matching for Generative Modeling";
Liu et al. 2023, "Flow Straight and Fast") learns to sample a distribution
by learning a velocity field that carries Gaussian noise to it. Here the
distribution is conditional -- the next step given a context c -- and the
path from a noise sample z ~ N(0, I) to a sample d of it is the straight
line

    x_t = (1 - t) z + t d,   t in [0, 1],

whose velocity is d - z. A network v(x, t, c) regressed onto d - z at
random points of random paths (`flow_matching_loss`) learns the expected
velocity E[d - z | x_t = x, c], the field whose flow carries N(0, I) to the
distribution of d given c; a sample is the flow's endpoint from a fresh z
(`euler_sample`).

- t is drawn uniformly. Every sample starts at t = 0, whatever the number
  of steps, and there x_0 = z says nothing about d, so the optimal velocity
  is E[d | c] - z and one Euler step lands on E[d | c], the conditional
  mean: with one sampling step the model is (as closely as its network
  learned that field) a deterministic one-step forecaster, the ablation
  that separates what sampling adds from what the network does. A schedule
  that rarely draws t near 0 (logit-normal, as for images) would leave
  exactly that point thinly trained.
- Euler with uniform steps. The sampler is inference only: how the scores
  change with the step count measures its error, and a higher-order one
  (Heun) could be tried on the same trained network.
- Noise per forecast step (`step_noise`): drawn from the member's key and
  the step's index, not split off a key carried along the rollout, so a
  forecast streamed in blocks, or a member sampled in a batch, draws the
  same noise as one call for its seed alone.

Functions take the velocity field as `velocity(x, t, condition)` on one
sample (an Equinox network, or any function): x of shape (r,), t a scalar.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

Velocity = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


def interpolate(noise: jax.Array, target: jax.Array, t: jax.Array) -> jax.Array:
    """The point x_t = (1 - t) z + t d on the path from `noise` to `target`."""
    return (1.0 - t) * noise + t * target


def flow_matching_loss(
    velocity: Velocity, condition: jax.Array, target: jax.Array, key: jax.Array
) -> jax.Array:
    """The mean squared error of `velocity` against the path velocity d - z
    over a batch of (`condition`, `target`) pairs, shapes (B, ...) and
    (B, r): one t ~ U[0, 1) and one z ~ N(0, I) drawn per pair from `key`.
    Averaged over the batch and the r components, so a zero field scores
    E[d^2] + 1 per component."""
    t_key, noise_key = jax.random.split(key)
    t = jax.random.uniform(t_key, (len(target),))
    noise = jax.random.normal(noise_key, target.shape)
    x = interpolate(noise, target, t[:, None])
    predicted = jax.vmap(velocity)(x, t, condition)
    return jnp.mean((predicted - (target - noise)) ** 2)


def euler_sample(
    velocity: Velocity, condition: jax.Array, noise: jax.Array, steps: int
) -> jax.Array:
    """The flow's endpoint at t = 1 from `noise` at t = 0: `steps` Euler
    steps of 1 / `steps`, the velocity taken at t = 0, 1 / steps, ..."""
    dt = 1.0 / steps

    def advance(k, x):
        return x + dt * velocity(x, k * dt, condition)

    return jax.lax.fori_loop(0, steps, advance, noise)


def step_noise(key: jax.Array, step: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """The noise forecast step `step` (0 for the first forecast frame) of
    the member with `key` starts its sample from."""
    return jax.random.normal(jax.random.fold_in(key, step), shape)
