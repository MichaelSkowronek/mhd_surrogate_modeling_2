"""Codec training entry point: fit the configured codec, then score its
reconstruction on the validation dataset.

1. Logs the data version the run started against, as train.py does.
2. Fits `cfg.codec` (`models/codec.py`) on the train datasets, saves the
   checkpoint into the run's output directory (`codec/`) and logs it as an
   MLflow model named after the codec (`training/mlflow_codec.py`); a
   latent model references it as `models:/<model id>`.
3. Scores its reconstruction (`evaluation/reconstruction.py`) on
   `cfg.data.val_dataset` and `cfg.evaluation.train_datasets`, as `val.*`
   and `train.<dataset>.*` metrics: `reconstruction_rmse` and the physics
   scores.

Codec runs are tracked in their own MLflow experiment, `mhd-codecs`
(configs/train_codec.yaml). Like train.py, never opens
`cfg.data.test_dataset`.

Usage:
    uv run --extra gpu scripts/training/train_codec.py                  # codec=pod, rank 750
    uv run --extra gpu scripts/training/train_codec.py -m codec.rank=250,1500  # a sweep
"""

from __future__ import annotations

import os
from pathlib import Path

# As in train.py: allocate GPU memory on demand. Before jax is imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from hydra.types import RunMode  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

import mhd_surrogate.utils.hydra_resolvers  # noqa: E402, F401  (before @hydra.main resolves)
from mhd_surrogate.training.codec_run import run_codec_training  # noqa: E402


@hydra.main(config_path="../../configs", config_name="train_codec")
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    tags = None
    if hydra_cfg.mode == RunMode.MULTIRUN:
        tags = {"sweep": Path(hydra_cfg.sweep.dir).name, "sweep_job": hydra_cfg.job.num}
    run_codec_training(
        cfg,
        output_dir,
        output_dir / f"{hydra_cfg.job.name}.log",
        run_name(cfg.codec.name, list(hydra_cfg.overrides.task)),
        tags=tags,
    )


def run_name(codec_name: str, overrides: list[str]) -> str:
    """The codec's name plus the command-line overrides that set its
    hyperparameters (e.g. `pod codec.rank=250`); overrides that pick the
    codec or the tracking backend, or touch Hydra itself, are left out."""
    skipped = ("codec=", "mlflow", "hydra.", "~", "+mlflow")
    shown = [o for o in overrides if not o.startswith(skipped)]
    return " ".join([codec_name, *shown])


if __name__ == "__main__":
    main()
