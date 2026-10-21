"""What the Hydra entry points do before their own work: configure JAX,
record the data version a tracked run starts against, and open the data.

Shared by training (`run.py`), the scoring of a stored checkpoint
(`scoring.py`) and its gate (`gating.py`), so they configure the process,
record provenance and check the data the same way.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import zarr
from omegaconf import DictConfig

import mlflow
from mhd_surrogate.data.normalization import NormalizationStats
from mhd_surrogate.data.versioning import data_provenance
from mhd_surrogate.utils.jax_cache import enable_compilation_cache
from mhd_surrogate.utils.jax_determinism import enable_deterministic_ops

log = logging.getLogger(__name__)


def configure_process(cfg: DictConfig) -> None:
    """JAX's persistent compilation cache and deterministic GPU ops per
    `cfg.jax`, before anything is jitted and the backend starts (JAX reads
    the cache config on its first compile, XLA its flags at start-up)."""
    if cfg.jax.compilation_cache_dir:
        enable_compilation_cache(cfg.jax.compilation_cache_dir, cfg.jax.compilation_cache_max_gb)
    if cfg.jax.get("deterministic_ops", False):
        enable_deterministic_ops()
    # Silences mlflow's "load this tracing skill" hint on every call, which
    # is unrelated to this project's plain params/metrics logging.
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def log_data_version(stats_path: str | Path) -> None:
    """Which data/stats version the active run started against: the hashes
    dvc.lock pins, as params (searchable in the UI), plus the lock and the
    stats as artifacts. Keeping disk in sync with the lock is `dvc repro`'s
    job, not checked here."""
    mlflow.log_params({f"data_version.{k}": v for k, v in data_provenance().items()})
    mlflow.log_artifact("dvc.lock")
    mlflow.log_artifact(str(stats_path))


def open_data(cfg: DictConfig) -> tuple[zarr.Group, NormalizationStats]:
    """The zarr store and the normalization statistics, checked against it
    and against the configured split. The test dataset is never opened."""
    root = zarr.open_group(store=cfg.data.zarr_store, mode="r")
    stats_path = Path(cfg.normalization.stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} not found; run scripts/data/compute_stats.py first")
    stats = NormalizationStats.load(stats_path)
    stats.check_compatible(
        root.attrs["channel_names"], cfg.data.train_datasets, cfg.data.test_dataset
    )
    log.info("test_dataset %s: configured, deliberately not opened", cfg.data.test_dataset)
    return root, stats
