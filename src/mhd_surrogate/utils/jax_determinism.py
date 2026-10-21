"""Deterministic GPU kernels for JAX entry points.

On the GPU, XLA autotunes every convolution and matmul when it compiles a
program: it times candidate kernels and keeps the fastest. The candidates
don't all round the same way, and which one wins depends on timing noise
and free GPU memory, so two cold compiles of the same program can produce
different bits. For the U-Net that's enough to change a forecast from its
first step (RMSE at lead 10 0.4154-0.4170 over four cold compiles of one
checkpoint) and, through training, which epoch early stopping keeps. A
persistent compilation cache hides it on one machine (a cache hit reuses
the kernels it compiled with), but not across machines, cache evictions or
jax upgrades.

`--xla_gpu_deterministic_ops=true` makes XLA pick only deterministic
kernels, so every compile on a given GPU and software stack produces the
same bits: measured, identical checkpoints and forecasts across cold
compiles, for ~50% more time per U-Net training epoch (47 -> 71 s) and ~8%
per forecast. Bit-identical results across *different* GPU models remain
out of reach either way. XLA reads its flags when JAX initializes its
backend, so call `enable_deterministic_ops` before anything touches a device;
the flag is part of the persistent cache's key, so deterministic and
autotuned compiles never share a cache entry.

XLA's per-fusion autotuning cache, which JAX keeps in the same directory
(`utils/jax_cache.py`), isn't keyed on the flag, though: a deterministic
compile reused the kernel an autotuned compile of the same fusion had
stored there. After an exploration run of the POD codec, which compiles
the same Gram matrix matmul, the DVC `train` stage's Hankel DMD came out
~1e-6 off the canonical checkpoint; with an empty cache, bit for bit on it.
So deterministic runs turn XLA's caches off; the compilation cache itself
stays on.
"""

from __future__ import annotations

import logging
import os

import jax
from jax._src import xla_bridge

log = logging.getLogger(__name__)

FLAG = "--xla_gpu_deterministic_ops=true"
# `jax_persistent_cache_enable_xla_caches` names the XLA caches to enable;
# a value naming none of them (rather than None, "unset") enables none.
XLA_CACHES_OFF = "none"


def enable_deterministic_ops() -> None:
    """Add `FLAG` to `XLA_FLAGS` (keeping any flags already set) and turn
    off XLA's caches beside JAX's compilation cache. Raises if JAX's backend
    is already initialized: XLA would ignore the flag."""
    if xla_bridge.backends_are_initialized():
        raise RuntimeError(
            "JAX's backend is already initialized, so XLA would ignore "
            f"{FLAG}: enable deterministic ops before anything touches a device"
        )
    flags = os.environ.get("XLA_FLAGS", "").split()
    if FLAG not in flags:
        os.environ["XLA_FLAGS"] = " ".join([*flags, FLAG])
    jax.config.update("jax_persistent_cache_enable_xla_caches", XLA_CACHES_OFF)
    log.info("XLA deterministic GPU ops enabled (XLA_FLAGS=%s)", os.environ["XLA_FLAGS"])
