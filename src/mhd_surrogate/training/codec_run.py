"""One codec training run: fit the configured codec, then score its
reconstruction on validation.

The body of `scripts/training/train_codec.py` (see its docstring), in the
package as `run.py` is for train.py. A codec is fitted, logged and
referenced on its own, not inside a latent model's run (`models/codec.py`).
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

import mlflow
from mhd_surrogate.data.grid import grid_spacing
from mhd_surrogate.evaluation.evaluate import train_eval_datasets
from mhd_surrogate.evaluation.reconstruction import reconstruction_scores
from mhd_surrogate.models.codec import build_codec
from mhd_surrogate.training.mlflow_codec import log_codec
from mhd_surrogate.training.mlflow_utils import finite_metrics, log_checkpoint_once
from mhd_surrogate.training.startup import configure_process, log_data_version, open_data
from mhd_surrogate.training.tracking import tracked_run

log = logging.getLogger(__name__)

# The config groups that decide a codec run: logged as its params (the
# forecasting-only groups the root config also composes are left out).
CODEC_CONFIG = ("data", "normalization", "codec", "evaluation", "jax", "seed")


def run_codec_training(
    cfg: DictConfig,
    output_dir: Path,
    log_file: Path | None,
    run_name: str,
    *,
    tags: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Fit `cfg.codec` in an MLflow run, save its checkpoint to
    `<output_dir>/codec` and log it as a model; returns its reconstruction
    scores by prefix (`val`, `train.<dataset>`)."""
    log.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))
    configure_process(cfg)

    resolved = OmegaConf.to_container(cfg, resolve=True)
    params = {key: resolved[key] for key in CODEC_CONFIG}
    with tracked_run(
        cfg.mlflow.tracking_uri,
        cfg.mlflow.experiment_name,
        params,
        log_file=log_file,
        run_name=run_name,
    ):
        if tags:
            mlflow.set_tags(tags)
        log_data_version(cfg.normalization.stats_path)
        mlflow.log_param("jax_backend", jax.default_backend())

        root, stats = open_data(cfg)
        codec = build_codec(resolved["codec"])
        train = {name: root[name] for name in cfg.data.train_datasets}
        start = time.perf_counter()
        codec.fit(train)
        fit_seconds = time.perf_counter() - start
        n_parameters = codec.parameter_count()
        mlflow.log_metrics(
            {
                "fit_seconds": fit_seconds,
                "n_parameters": n_parameters,
                "latent_size": math.prod(codec.latent_shape),
                **{f"fit.{k}": v for k, v in getattr(codec, "fit_info", {}).items()},
            }
        )
        log.info(
            "fitted codec %s (latent shape %s, %d parameters) in %.1f s",
            codec.name,
            codec.latent_shape,
            n_parameters,
            fit_seconds,
        )

        checkpoint = output_dir / "codec"
        codec.save(checkpoint)
        frame_shape = tuple(next(iter(train.values())).shape[1:])
        model_id = log_checkpoint_once(
            checkpoint, lambda: log_codec(codec, checkpoint, frame_shape, params=resolved["codec"])
        )
        log.info("checkpoint saved to %s, logged as model %s", checkpoint, model_id)

        scale = np.asarray(stats.effective_std(cfg.normalization.std_mode))
        targets = [(cfg.data.val_dataset, "val")] + [
            (name, f"train.{name}")
            for name in train_eval_datasets(
                list(cfg.evaluation.train_datasets), list(cfg.data.train_datasets)
            )
        ]
        return {
            prefix: _score_one(codec, root[name], prefix, scale, model_id)
            for name, prefix in targets
        }


def _score_one(codec, series, prefix: str, scale: np.ndarray, model_id: str) -> dict[str, float]:
    """Score `codec`'s reconstruction of one dataset and log the scores as
    `<prefix>.*` metrics, on the run and linked to the logged model."""
    dx, dy = grid_spacing(series.shape[2], series.shape[3])
    start = time.perf_counter()
    scores = reconstruction_scores(codec, series, scale, dx, dy)
    eval_seconds = time.perf_counter() - start
    finite, undefined = finite_metrics(scores)
    mlflow.log_metrics({f"{prefix}.{k}": v for k, v in finite.items()}, model_id=model_id)
    mlflow.log_metric(f"{prefix}.eval_seconds", eval_seconds, model_id=model_id)
    if undefined:
        log.info("%s: undefined for this codec, not logged: %s", prefix, ", ".join(undefined))
    log.info(
        "%s reconstruction scores (%.1f s):\n%s",
        prefix,
        eval_seconds,
        "\n".join(f"  {k}: {v:.4g}" for k, v in sorted(scores.items())),
    )
    return scores
