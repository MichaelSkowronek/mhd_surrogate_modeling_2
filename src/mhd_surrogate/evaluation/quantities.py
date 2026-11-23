"""Per-step domain-mean quantities of a velocity series.

Shared by the long-horizon diagnostics (`diagnostics.py`, scored once after
training) and the stability check (`stability.py`, part of the selection
score that early stopping uses every epoch). Kept in a module of their own
so that training -- and the DVC `train` stage's dependencies -- only reach
what early stopping needs, not the diagnostics.

Each comes twice: in NumPy on the host (`kinetic_energy`, `enstrophy`, in
float64 by their callers) and in JAX on the device
(`device_energy_and_enstrophy`). The scoring reduces a forecast on the
device, a block of frames at a time as the model produces it, so the frames
never reach the host (an 837-step forecast is ~1 GB; copying and scoring it
there in float64 took most of a validation's time). The device versions run
in float32, ~1e-7 relative to the float64 ones on the validation data, and
the NumPy versions are the reference they are tested against.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from mhd_surrogate.analysis.fields import vorticity

# Steps processed at a time, so a long series is never in float64 at once,
# nor a forecast's frames on the device.
DEFAULT_CHUNK_T = 32


def kinetic_energy(block: np.ndarray) -> np.ndarray:
    """Spatial mean of 0.5 |u|^2 per step, shape (t,)."""
    return 0.5 * (block**2).sum(axis=1).mean(axis=(1, 2))


def enstrophy(block: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Spatial mean of 0.5 w^2 per step, shape (t,)."""
    return 0.5 * (vorticity(block, dx, dy) ** 2).mean(axis=(1, 2))


def to_device(frames) -> jax.Array:
    """`frames` as a float32 array on the device (a no-op for one there)."""
    return jnp.asarray(frames, dtype=jnp.float32)


def device_gradient(f: jax.Array, spacing: float, axis: int) -> jax.Array:
    """`np.gradient(f, spacing, axis=axis)` for a uniform `spacing`:
    second-order central differences, one-sided at the boundaries."""
    f = jnp.moveaxis(f, axis, -1)
    first = (f[..., 1:2] - f[..., :1]) / spacing
    interior = (f[..., 2:] - f[..., :-2]) / (2 * spacing)
    last = (f[..., -1:] - f[..., -2:-1]) / spacing
    return jnp.moveaxis(jnp.concatenate([first, interior, last], axis=-1), -1, axis)


def device_vorticity(frames: jax.Array, dx: float, dy: float) -> jax.Array:
    """`analysis.fields.vorticity` of (t, 2, Nx, Ny) frames, (t, Nx, Ny)."""
    return device_gradient(frames[:, 1], dx, 1) - device_gradient(frames[:, 0], dy, 2)


@partial(jax.jit, static_argnames=("dx", "dy"))
def device_energy_and_enstrophy(
    frames: jax.Array, dx: float, dy: float
) -> tuple[jax.Array, jax.Array]:
    """`kinetic_energy` and `enstrophy` of (t, 2, Nx, Ny) frames on the
    device, in float32: (energy, enstrophy), shape (t,) each."""
    frames = frames.astype(jnp.float32)
    energy = 0.5 * jnp.mean(jnp.sum(frames**2, axis=1), axis=(1, 2))
    ens = 0.5 * jnp.mean(device_vorticity(frames, dx, dy) ** 2, axis=(1, 2))
    return energy, ens
