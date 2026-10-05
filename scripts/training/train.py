"""Training entry point: fit the configured model, then score it on the
validation dataset.

1. Logs the data version the run started against (`dvc.lock` hashes as
   params, `dvc.lock` and the normalization stats as artifacts).
2. Fits `cfg.model` on the train datasets and saves the checkpoint into the
   run's output directory, logged to MLflow as the `model` artifact.
3. Scores it on `cfg.data.val_dataset` under the forecast protocol
   (`evaluation/protocol.py`): scalar scores as `val.*` metrics and the
   per-lead-time RMSE as the `val.rmse` history (step = lead time).
4. Scores it the same way on `cfg.evaluation.train_datasets` (a subset of the
   train datasets), as `train.<dataset>.*`: a sanity check of the fit and the
   train/val gap, not something to make decisions on.

Deliberately never opens `cfg.data.test_dataset`, not even to check its
shape -- see configs/data/re16k.yaml's docstring: it's the dataset-level
held-out test set and must never be read by any script, including this
one, until final evaluation.

Usage:
    uv run scripts/training/train.py                       # model=mean_field
    uv run scripts/training/train.py model=persistence
    uv run scripts/training/train.py mlflow=server  # Docker stack, see README
    uv run mlflow ui --backend-store-uri sqlite:///mlruns.db  # view runs
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import hydra
import jax
import numpy as np
import zarr
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import mlflow
from mhd_surrogate.data.grid import grid_spacing
from mhd_surrogate.data.normalization import NormalizationStats
from mhd_surrogate.data.versioning import data_provenance
from mhd_surrogate.evaluation.evaluate import evaluate, train_eval_datasets
from mhd_surrogate.models.registry import build_model
from mhd_surrogate.training.mlflow_utils import finite_metrics, log_metric_series
from mhd_surrogate.training.tracking import tracked_run

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Logging is Hydra's job_logging config (configs/hydra/job_logging/
    # project.yaml); this is the DEBUG file it writes in the run's output
    # directory, which tracked_run uploads to MLflow.
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    log_file = output_dir / f"{hydra_cfg.job.name}.log"
    log.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))

    # Silences mlflow's "load this tracing skill" hint on every call, which
    # is unrelated to this project's plain params/metrics logging.
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

    resolved = OmegaConf.to_container(cfg, resolve=True)
    with tracked_run(
        cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name, resolved, log_file=log_file
    ):
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
        start = time.perf_counter()
        model.fit(train)
        fit_seconds = time.perf_counter() - start
        mlflow.log_metric("fit_seconds", fit_seconds)
        log.info("fitted %s in %.1f s", model.name, fit_seconds)

        checkpoint = output_dir / "model"
        model.save(checkpoint)
        mlflow.log_artifacts(str(checkpoint), artifact_path="model")
        log.info("checkpoint saved to %s", checkpoint)

        # Validation is what decisions are made on; the training dataset(s)
        # are scored the same way as a sanity check (can the model fit at all,
        # and how big is the train/val gap).
        targets = [(cfg.data.val_dataset, "val")] + [
            (name, f"train.{name}")
            for name in train_eval_datasets(
                list(cfg.evaluation.train_datasets), list(cfg.data.train_datasets)
            )
        ]
        scale = np.asarray(stats.effective_std(cfg.normalization.std_mode))
        for name, prefix in targets:
            _score_and_log(model, root[name], name, prefix, scale, cfg)


def _score_and_log(model, series, name: str, prefix: str, scale: np.ndarray, cfg) -> None:
    """Score `model` on one dataset under the forecast protocol and log the
    scalar scores as `<prefix>.*` metrics and the RMSE curve as `<prefix>.rmse`."""
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
    log.info("evaluated on %s in %.1f s", name, time.perf_counter() - start)

    finite, undefined = finite_metrics(result.scores)
    mlflow.log_metrics({f"{prefix}.{k}": v for k, v in finite.items()})
    if undefined:
        log.info("%s: undefined for this model, not logged: %s", prefix, ", ".join(undefined))
    log_metric_series(f"{prefix}.rmse", result.rmse, start_step=1)
    log.info(
        "%s scores:\n%s",
        prefix,
        "\n".join(f"  {k}: {v:.4g}" for k, v in sorted(result.scores.items())),
    )


if __name__ == "__main__":
    main()
