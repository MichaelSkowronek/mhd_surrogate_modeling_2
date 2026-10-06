"""Persistent XLA compilation cache for JAX entry points.

JAX compiles every jitted function on first call in each process, and on
the GPU that includes autotuning: DMD's Gram matmul alone takes ~45 s to
compile, every run. With a persistent cache, a compiled executable (and
XLA's per-fusion autotuning results) is written to disk and reused by later
processes that compile the same program: same function, shapes, dtypes,
device, compile flags and jax/jaxlib version, all part of the cache key, so
a stale or foreign entry is never used, only missed.

A cache hit reuses the executable a cold compile produced, so it doesn't
change results; the cache is a performance knob, not a DVC dependency.

JAX initializes the cache on its first compile and ignores later changes to
its config, so call `enable_compilation_cache` before anything is jitted.
"""

from __future__ import annotations

import logging
from pathlib import Path

import jax

log = logging.getLogger(__name__)

# Repo-relative, like data/ and outputs/: scripts run from the repo root.
DEFAULT_CACHE_DIR = Path(".jax_cache")


def enable_compilation_cache(
    cache_dir: str | Path,
    max_size_gb: float = 2.0,
    min_compile_time_secs: float = 1.0,
) -> Path:
    """Point JAX's persistent compilation cache at `cache_dir` (created if
    missing) and return its absolute path.

    `max_size_gb` caps the cache, least recently used entries evicted first;
    `min_compile_time_secs` skips caching programs that compile faster than
    that (JAX's default 1 s), whose disk round trip wouldn't pay off.
    """
    path = Path(cache_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_compilation_cache_max_size", int(max_size_gb * 1024**3))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_time_secs)
    log.info("JAX compilation cache: %s (max %.1f GB)", path, max_size_gb)
    return path
