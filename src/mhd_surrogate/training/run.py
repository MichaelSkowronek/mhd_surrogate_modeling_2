"""One training run: fit the configured model, then score it on validation.

The body of `scripts/training/train.py`, in the package so that the
hyperparameter search (`scripts/training/tune.py`) runs exactly the same
thing in each Ray Tune trial (code run on Ray workers has to be importable
from `src/`). See train.py's docstring for what a run does; the entry points
only differ in where the output directory, run name and tags come from.
"""

from __future__ import annotations

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
from mhd_surrogate.evaluation.evaluate import evaluate, selection_scores, train_eval_datasets
from mhd_surrogate.models.base import FitHooks, fit_model
from mhd_surrogate.models.registry import build_model
from mhd_surrogate.training.export import write_metrics
from mhd_surrogate.training.mlflow_model import log_surrogate
from mhd_surrogate.training.mlflow_utils import finite_metrics, log_metric_series
from mhd_surrogate.training.tracking import resumed_run_id, save_run_record, tracked_run
from mhd_surrogate.utils.jax_cache import enable_compilation_cache

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
    prefix (`val`, `train.<dataset>`).

    `output_dir` gets the checkpoint (unless `cfg.export.dir` is set) and,
    for an iterative model, `training_state/`; with `cfg.resume` set, the run
    recorded there is continued. `tags` are set on the MLflow run;
    `on_validation` is called after every validation during training.
    """
    log.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))
    # Before anything is jitted: JAX reads the cache config on first compile.
    if cfg.jax.compilation_cache_dir:
        enable_compilation_cache(cfg.jax.compilation_cache_dir, cfg.jax.compilation_cache_max_gb)

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
        log.info("checkpoint saved to %s, logged as model %s", checkpoint, model_info.model_id)

        # Validation is what decisions are made on; the training dataset(s)
        # are scored the same way as a sanity check (can the model fit at all,
        # and how big is the train/val gap).
        targets = [(cfg.data.val_dataset, "val")] + [
            (name, f"train.{name}")
            for name in train_eval_datasets(
                list(cfg.evaluation.train_datasets), list(cfg.data.train_datasets)
            )
        ]
        scores = {
            prefix: _score_and_log(model, root[name], name, prefix, scale, cfg, model_info.model_id)
            for name, prefix in targets
        }
        if export_dir is not None:
            write_metrics(export_dir / "metrics.json", scores)
            log.info("exported checkpoint and metrics to %s", export_dir)
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
        self.context_steps = cfg.data.context_steps
        self.skill_threshold = cfg.evaluation.skill_threshold
        self.tie_break_lead = cfg.evaluation.tie_break_lead

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
        )


def _score_and_log(
    model, series, name: str, prefix: str, scale: np.ndarray, cfg, model_id: str
) -> dict[str, float]:
    """Score `model` on one dataset under the forecast protocol and log the
    scalar scores as `<prefix>.*` metrics and the RMSE curve as `<prefix>.rmse`,
    on the run and linked to the logged model `model_id`. Returns the scores."""
    dx, dy = grid_spacing(series.shape[2], series.shape[3])
    start = time.perf_counter()
    result = evaluate(
        model,
        series,
        cfg.data.context_steps,
        dx,
        dy,
        scale=scale,
        skill_threshold=cfg.evaluation.skill_threshold,
        report_leads=list(cfg.evaluation.report_leads),
    )
    eval_seconds = time.perf_counter() - start
    log.info("evaluated on %s in %.1f s", name, eval_seconds)

    finite, undefined = finite_metrics(result.scores)
    mlflow.log_metrics({f"{prefix}.{k}": v for k, v in finite.items()}, model_id=model_id)
    # Timings are logged to MLflow only, not returned with the scores: the
    # scores go to the DVC-tracked metrics.json, which must not change from
    # one `dvc repro` to the next.
    mlflow.log_metrics(
        {
            f"{prefix}.eval_seconds": eval_seconds,
            f"{prefix}.forecast_seconds_per_frame": result.seconds_per_frame,
        },
        model_id=model_id,
    )
    if undefined:
        log.info("%s: undefined for this model, not logged: %s", prefix, ", ".join(undefined))
    log_metric_series(f"{prefix}.rmse", result.rmse, start_step=1, model_id=model_id)
    log.info(
        "%s scores:\n%s",
        prefix,
        "\n".join(f"  {k}: {v:.4g}" for k, v in sorted(result.scores.items())),
    )
    return result.scores
