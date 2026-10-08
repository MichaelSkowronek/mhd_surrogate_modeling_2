"""Score a fitted model in full and log the scores.

The end of a training run (`run.py`, unless `evaluation.final=false`) and
the whole job of the DVC `evaluate` stage, which scores the `train` stage's
stored checkpoint (`scripts/evaluation/evaluate_checkpoint.py`). `run.py`
imports this module by name, only when the run scores itself, so the
`train` stage -- which doesn't -- doesn't depend on the evaluation code
beyond what early stopping uses (`evaluation/selection.py`).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from omegaconf import DictConfig, OmegaConf

import mlflow
from mhd_surrogate.data.grid import grid_spacing
from mhd_surrogate.data.normalization import NormalizationStats
from mhd_surrogate.data.versioning import data_provenance
from mhd_surrogate.evaluation.evaluate import evaluate, train_eval_datasets
from mhd_surrogate.models.registry import load_model
from mhd_surrogate.training.export import checkpoint_digest, write_metrics
from mhd_surrogate.training.mlflow_utils import finite_metrics, log_metric_series
from mhd_surrogate.training.tracking import tracked_run
from mhd_surrogate.utils.jax_cache import enable_compilation_cache
from mhd_surrogate.utils.jax_determinism import enable_deterministic_ops

log = logging.getLogger(__name__)


# The config groups that decide a checkpoint's scores: logged as the
# evaluation run's params (the model's own config is in the checkpoint).
EVALUATION_CONFIG = ("data", "normalization", "evaluation", "jax", "checkpoint")


def run_evaluation(cfg: DictConfig, log_file: Path | None) -> dict[str, dict[str, float]]:
    """Score the checkpoint `cfg.checkpoint` in an MLflow run of its own: the
    body of `scripts/evaluation/evaluate_checkpoint.py`, the DVC `evaluate`
    stage. Writes `<cfg.export.dir>/metrics.json` if `export.dir` is set and
    returns the scores by prefix.

    The scores are also linked to the logged model of the run that trained
    the checkpoint, found by the checkpoint's content hash
    (`checkpoint_sha256`, tagged by `run.py`); a checkpoint from elsewhere
    (e.g. `dvc pull`ed without our tracking store) is scored all the same.
    """
    if not cfg.checkpoint:
        raise ValueError("checkpoint=<checkpoint directory> is required")
    # Before anything is jitted / the backend starts, as in run.py.
    if cfg.jax.compilation_cache_dir:
        enable_compilation_cache(cfg.jax.compilation_cache_dir, cfg.jax.compilation_cache_max_gb)
    if cfg.jax.get("deterministic_ops", False):
        enable_deterministic_ops()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

    checkpoint = Path(cfg.checkpoint)
    model = load_model(checkpoint)
    digest = checkpoint_digest(checkpoint)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    params = {key: resolved[key] for key in EVALUATION_CONFIG}
    with tracked_run(
        cfg.mlflow.tracking_uri,
        cfg.mlflow.experiment_name,
        params,
        log_file=log_file,
        run_name=f"evaluate {model.name}",
    ) as run:
        mlflow.set_tags({"stage": "evaluate", "checkpoint_sha256": digest})
        mlflow.log_params({f"data_version.{k}": v for k, v in data_provenance().items()})
        mlflow.log_artifact("dvc.lock")
        mlflow.log_artifact(cfg.normalization.stats_path)

        root = zarr.open_group(store=cfg.data.zarr_store, mode="r")
        stats = NormalizationStats.load(Path(cfg.normalization.stats_path))
        stats.check_compatible(
            root.attrs["channel_names"], cfg.data.train_datasets, cfg.data.test_dataset
        )
        log.info("test_dataset %s: configured, deliberately not opened", cfg.data.test_dataset)
        scale = np.asarray(stats.effective_std(cfg.normalization.std_mode))

        model_id = trained_model_id(run.info.experiment_id, digest)
        if model_id is None:
            log.warning("no logged model with checkpoint_sha256 %s: scores not linked", digest)
        else:
            mlflow.set_tag("evaluates_model_id", model_id)
        scores = score_and_log(model, root, cfg, scale, model_id)
        if cfg.export.dir:
            write_metrics(Path(cfg.export.dir) / "metrics.json", scores)
            log.info("wrote %s", Path(cfg.export.dir) / "metrics.json")
        return scores


def trained_model_id(experiment_id: str, digest: str) -> str | None:
    """The newest logged model in the experiment whose checkpoint has
    content hash `digest`, if any."""
    models = mlflow.search_logged_models(
        experiment_ids=[experiment_id],
        filter_string=f"tags.checkpoint_sha256 = '{digest}'",
        order_by=[{"field_name": "creation_time", "ascending": False}],
        max_results=1,
        output_format="list",
    )
    return models[0].model_id if models else None


def score_and_log(
    model, root, cfg, scale: np.ndarray, model_id: str | None
) -> dict[str, dict[str, float]]:
    """Score `model` on the validation dataset (`val`) and on
    `cfg.evaluation.train_datasets` (`train.<name>`) and log the scores to
    the active MLflow run, linked to the logged model `model_id` if given.
    Returns the scores by prefix.

    Validation is what decisions are made on; the training dataset(s) are
    scored the same way as a sanity check (can the model fit at all, and how
    big is the train/val gap)."""
    targets = [(cfg.data.val_dataset, "val")] + [
        (name, f"train.{name}")
        for name in train_eval_datasets(
            list(cfg.evaluation.train_datasets), list(cfg.data.train_datasets)
        )
    ]
    return {
        prefix: _score_one(model, root[name], name, prefix, scale, cfg, model_id)
        for name, prefix in targets
    }


def _score_one(
    model, series, name: str, prefix: str, scale: np.ndarray, cfg: Any, model_id: str | None
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
        block_steps=cfg.evaluation.stability.block_steps,
        max_ratio=cfg.evaluation.stability.max_ratio,
        n_members=cfg.evaluation.ensemble_size,
        member_batch=cfg.evaluation.member_batch,
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
    log_metric_series(f"{prefix}.crps", result.ensemble.crps, start_step=1, model_id=model_id)
    if result.ensemble.size > 1:
        for name in ("spread", "ensemble_mean_rmse"):
            curve = getattr(result.ensemble, name)
            log_metric_series(f"{prefix}.{name}", curve, start_step=1, model_id=model_id)
    log.info(
        "%s scores:\n%s",
        prefix,
        "\n".join(f"  {k}: {v:.4g}" for k, v in sorted(result.scores.items())),
    )
    return result.scores
