"""Training entry point: fit the configured model, then score it on the
validation dataset.

1. Logs the data version the run started against (`dvc.lock` hashes as
   params, `dvc.lock` and the normalization stats as artifacts).
2. Fits `cfg.model` on the train datasets, saves the checkpoint into the
   run's output directory and logs it as an MLflow model named after the
   model (`training/mlflow_model.py`); the run is named after it too. An
   iteratively trained model (a neural network) also gets `FitHooks`
   (`models/base.py`): validation scores to early-stop on, MLflow for its
   training curves, and `training_state/` in the output directory, from
   which `resume=<output dir>` continues an interrupted run.
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
    uv run scripts/training/train.py -m model=dmd model.rank=50,100,200  # a sweep
    uv run scripts/training/train.py model=unet
    uv run scripts/training/train.py model=unet resume=outputs/<date>/<time>  # continue it
    uv run dvc repro train   # the canonical model, as a DVC pipeline stage
    uv run scripts/training/train.py mlflow=server  # Docker stack, see README
    uv run mlflow ui --backend-store-uri sqlite:///mlruns.db  # view runs
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

# JAX preallocates 75% of GPU memory at start-up. Under WSL, with the
# desktop on the same GPU, far less is free, so that fails and retries
# noisily; allocate on demand instead (the usual setting for a shared GPU).
# Must be set before jax is imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from hydra.types import RunMode  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import mhd_surrogate.utils.hydra_resolvers  # noqa: E402, F401  (before @hydra.main resolves)
import mlflow  # noqa: E402
from mhd_surrogate.data.grid import grid_spacing  # noqa: E402
from mhd_surrogate.data.normalization import NormalizationStats  # noqa: E402
from mhd_surrogate.data.versioning import data_provenance  # noqa: E402
from mhd_surrogate.evaluation.evaluate import (  # noqa: E402
    evaluate,
    selection_scores,
    train_eval_datasets,
)
from mhd_surrogate.models.base import FitHooks, fit_model  # noqa: E402
from mhd_surrogate.models.registry import build_model  # noqa: E402
from mhd_surrogate.training.export import write_metrics  # noqa: E402
from mhd_surrogate.training.mlflow_model import log_surrogate  # noqa: E402
from mhd_surrogate.training.mlflow_utils import finite_metrics, log_metric_series  # noqa: E402
from mhd_surrogate.training.tracking import (  # noqa: E402
    resumed_run_id,
    save_run_record,
    tracked_run,
)
from mhd_surrogate.utils.jax_cache import enable_compilation_cache  # noqa: E402

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
    # Before anything is jitted: JAX reads the cache config on first compile.
    if cfg.jax.compilation_cache_dir:
        enable_compilation_cache(cfg.jax.compilation_cache_dir, cfg.jax.compilation_cache_max_gb)

    # Silences mlflow's "load this tracing skill" hint on every call, which
    # is unrelated to this project's plain params/metrics logging.
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

    # `resume` only says where to continue; it isn't part of what's trained
    # (and MLflow wouldn't let a resumed run's params change).
    resolved = OmegaConf.to_container(cfg, resolve=True)
    resolved.pop("resume")
    state_dir = output_dir / "training_state"
    run_id = resumed_run_id(state_dir, resolved) if cfg.resume else None
    with tracked_run(
        cfg.mlflow.tracking_uri,
        cfg.mlflow.experiment_name,
        resolved,
        log_file=log_file,
        run_name=run_name(cfg.model.name, list(hydra_cfg.overrides.task)),
        run_id=run_id,
    ) as run:
        if hydra_cfg.mode == RunMode.MULTIRUN:
            # Groups a sweep's runs in the MLflow UI (filter: tags.sweep = '...').
            mlflow.set_tags(
                {"sweep": Path(hydra_cfg.sweep.dir).name, "sweep_job": hydra_cfg.job.num}
            )
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
            hooks = FitHooks(
                validate=ValidationMonitor(root[cfg.data.val_dataset], scale, cfg),
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


def run_name(model_name: str, overrides: list[str]) -> str:
    """The model's name plus the command-line overrides that set its
    hyperparameters (e.g. `dmd model.rank=50`), so a sweep's runs are told
    apart in the runs table; overrides that pick the model or the tracking
    backend, or touch Hydra itself, are left out."""
    skipped = ("model=", "mlflow", "hydra.", "~", "+mlflow", "resume=")
    shown = [o for o in overrides if not o.startswith(skipped)]
    return " ".join([model_name, *shown])


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


if __name__ == "__main__":
    main()
