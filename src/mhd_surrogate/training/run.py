"""One training run: fit the configured model, then score it on validation.

The body of `scripts/training/train.py`, in the package so that the
hyperparameter search (`scripts/training/tune.py`) runs exactly the same
thing in each Ray Tune trial (code run on Ray workers has to be importable
from `src/`). See train.py's docstring for what a run does; the entry points
only differ in where the output directory, run name and tags come from.
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import numpy as np
import zarr
from omegaconf import DictConfig, OmegaConf

import mlflow
from mhd_surrogate.data.grid import grid_spacing
from mhd_surrogate.data.normalization import NormalizationStats
from mhd_surrogate.data.versioning import data_provenance
from mhd_surrogate.evaluation.selection import selection_scores
from mhd_surrogate.models.base import FitHooks, fit_model
from mhd_surrogate.models.registry import build_model
from mhd_surrogate.training.export import checkpoint_digest, write_metrics
from mhd_surrogate.training.mlflow_model import log_surrogate
from mhd_surrogate.training.tracking import resumed_run_id, save_run_record, tracked_run
from mhd_surrogate.utils.jax_cache import enable_compilation_cache
from mhd_surrogate.utils.jax_determinism import enable_deterministic_ops

log = logging.getLogger(__name__)

# Called with a validation's scores and the MLflow run id (e.g. to report
# them to Ray Tune).
OnValidation = Callable[[dict[str, float], str], None]


def run_training(
    cfg: DictConfig,
    output_dir: Path,
    log_file: Path | None,
    run_name: str,
    *,
    tags: dict[str, Any] | None = None,
    on_validation: OnValidation | None = None,
) -> dict[str, dict[str, float]]:
    """Fit and score `cfg.model` in an MLflow run; returns the scores by
    prefix (`val`, `train.<dataset>`), none with `evaluation.final=false`
    (the DVC `train` stage, whose `evaluate` stage scores the checkpoint).

    `output_dir` gets the checkpoint (unless `cfg.export.dir` is set) and,
    for an iterative model, `training_state/`; with `cfg.resume` set, the run
    recorded there is continued. `tags` are set on the MLflow run;
    `on_validation` is called after every validation during training.
    """
    log.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))
    # Before anything is jitted: JAX reads the cache config on first compile.
    if cfg.jax.compilation_cache_dir:
        enable_compilation_cache(cfg.jax.compilation_cache_dir, cfg.jax.compilation_cache_max_gb)
    # Before the backend starts: XLA reads its flags then.
    if cfg.jax.get("deterministic_ops", False):
        enable_deterministic_ops()

    # Silences mlflow's "load this tracing skill" hint on every call, which
    # is unrelated to this project's plain params/metrics logging.
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

    # `resume` only says where to continue; it isn't part of what's trained
    # (and MLflow wouldn't let a resumed run's params change).
    resolved = OmegaConf.to_container(cfg, resolve=True)
    resolved.pop("resume", None)
    state_dir = output_dir / "training_state"
    run_id = resumed_run_id(state_dir, resolved) if cfg.get("resume") else None
    with tracked_run(
        cfg.mlflow.tracking_uri,
        cfg.mlflow.experiment_name,
        resolved,
        log_file=log_file,
        run_name=run_name,
        run_id=run_id,
    ) as run:
        if tags:
            mlflow.set_tags(tags)
        # Which data/stats version this run was started against: the hashes
        # dvc.lock pins, as params (searchable in the UI), plus the lock and
        # stats as artifacts. Keeping disk in sync with the lock is `dvc
        # repro`'s job, not checked here.
        provenance = {f"data_version.{k}": v for k, v in data_provenance().items()}
        mlflow.log_params(provenance)
        mlflow.log_artifact("dvc.lock")
        mlflow.log_artifact(cfg.normalization.stats_path)
        mlflow.log_param("jax_backend", jax.default_backend())
        log.info("jax devices: %s", jax.devices())

        root = zarr.open_group(store=cfg.data.zarr_store, mode="r")
        stats_path = Path(cfg.normalization.stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(
                f"{stats_path} not found; run scripts/data/compute_stats.py first"
            )
        stats = NormalizationStats.load(stats_path)
        stats.check_compatible(
            root.attrs["channel_names"], cfg.data.train_datasets, cfg.data.test_dataset
        )
        log.info("test_dataset %s: configured, deliberately not opened", cfg.data.test_dataset)

        model = build_model(OmegaConf.to_container(cfg.model, resolve=True))
        train = {name: root[name] for name in cfg.data.train_datasets}
        scale = np.asarray(stats.effective_std(cfg.normalization.std_mode))
        hooks = FitHooks()
        if getattr(model, "iterative", False):
            if run_id is None:
                save_run_record(state_dir, run.info.run_id, resolved)
            validate = ValidationMonitor(root[cfg.data.val_dataset], scale, cfg)
            if on_validation is not None:
                validate = _reporting(validate, on_validation, run.info.run_id)
            hooks = FitHooks(
                validate=validate,
                log_metrics=lambda metrics, step: mlflow.log_metrics(metrics, step=step),
                state_dir=state_dir,
            )
        elif run_id is not None:
            raise ValueError(f"resume: {model.name} isn't trained iteratively")
        start = time.perf_counter()
        fit_model(model, train, hooks)
        fit_seconds = time.perf_counter() - start
        mlflow.log_metric("fit_seconds", fit_seconds)
        # Model-specific fit diagnostics (e.g. DMD's explained variance), if any.
        fit_info = getattr(model, "fit_info", {})
        if fit_info:
            mlflow.log_metrics({f"fit.{k}": v for k, v in fit_info.items()})
        log.info("fitted %s in %.1f s", model.name, fit_seconds)

        # The DVC `train` stage sets export.dir so the checkpoint and metrics
        # land at a fixed path (dvc.yaml outs); otherwise the run's directory.
        export_dir = Path(cfg.export.dir) if cfg.export.dir else None
        checkpoint = (export_dir or output_dir) / "model"
        model.save(checkpoint)
        frame_shape = tuple(next(iter(train.values())).shape[1:])
        model_info = log_surrogate(model, checkpoint, frame_shape, params=resolved["model"])
        # The checkpoint's content hash ties a later evaluation of it (the DVC
        # `evaluate` stage) back to this run's logged model.
        digest = checkpoint_digest(checkpoint)
        mlflow.set_tag("checkpoint_sha256", digest)
        mlflow.set_logged_model_tags(model_info.model_id, {"checkpoint_sha256": digest})
        log.info("checkpoint saved to %s, logged as model %s", checkpoint, model_info.model_id)

        scores: dict[str, dict[str, float]] = {}
        if cfg.evaluation.final:
            # By name: see training/scoring.py for why.
            scoring = importlib.import_module("mhd_surrogate.training.scoring")
            scores = scoring.score_and_log(model, root, cfg, scale, model_info.model_id)
        if export_dir is not None and scores:
            write_metrics(export_dir / "metrics.json", scores)
        if export_dir is not None:
            log.info("exported to %s", export_dir)
        return scores


def _reporting(monitor, on_validation: OnValidation, run_id: str):
    def validate(model) -> dict[str, float]:
        scores = monitor(model)
        on_validation(scores, run_id)
        return scores

    return validate


class ValidationMonitor:
    """`FitHooks.validate` for an iterative model: its selection scores on
    the validation dataset (`evaluation.selection_scores`), read into memory
    on first use (~1 GB) instead of from zarr at every evaluation."""

    def __init__(self, series, scale: np.ndarray, cfg: DictConfig) -> None:
        self.series = series
        self.frames: np.ndarray | None = None
        self.scale = scale
        self.dx, self.dy = grid_spacing(series.shape[2], series.shape[3])
        self.context_steps = cfg.data.context_steps
        self.skill_threshold = cfg.evaluation.skill_threshold
        self.tie_break_lead = cfg.evaluation.tie_break_lead
        self.stability = cfg.evaluation.stability

    def __call__(self, model) -> dict[str, float]:
        if self.frames is None:
            self.frames = np.asarray(self.series)
        return selection_scores(
            model,
            self.frames,
            self.context_steps,
            self.scale,
            self.skill_threshold,
            self.tie_break_lead,
            self.dx,
            self.dy,
            self.stability.block_steps,
            self.stability.max_ratio,
        )
