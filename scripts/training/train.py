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

import os
from pathlib import Path

# JAX preallocates 75% of GPU memory at start-up. Under WSL, with the
# desktop on the same GPU, far less is free, so that fails and retries
# noisily; allocate on demand instead (the usual setting for a shared GPU).
# Must be set before jax is imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from hydra.types import RunMode  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

import mhd_surrogate.utils.hydra_resolvers  # noqa: E402, F401  (before @hydra.main resolves)
from mhd_surrogate.training.run import run_training  # noqa: E402


@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Logging is Hydra's job_logging config (configs/hydra/job_logging/
    # project.yaml); this is the DEBUG file it writes in the run's output
    # directory, which tracked_run uploads to MLflow.
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    tags = None
    if hydra_cfg.mode == RunMode.MULTIRUN:
        # Groups a sweep's runs in the MLflow UI (filter: tags.sweep = '...').
        tags = {"sweep": Path(hydra_cfg.sweep.dir).name, "sweep_job": hydra_cfg.job.num}
    run_training(
        cfg,
        output_dir,
        output_dir / f"{hydra_cfg.job.name}.log",
        run_name(cfg.model.name, list(hydra_cfg.overrides.task)),
        tags=tags,
    )


def run_name(model_name: str, overrides: list[str]) -> str:
    """The model's name plus the command-line overrides that set its
    hyperparameters (e.g. `dmd model.rank=50`), so a sweep's runs are told
    apart in the runs table; overrides that pick the model or the tracking
    backend, or touch Hydra itself, are left out."""
    skipped = ("model=", "mlflow", "hydra.", "~", "+mlflow", "resume=")
    shown = [o for o in overrides if not o.startswith(skipped)]
    return " ".join([model_name, *shown])


if __name__ == "__main__":
    main()
