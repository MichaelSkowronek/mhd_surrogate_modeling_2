"""Score a stored checkpoint in full: the DVC `evaluate` stage.

The `train` stage fits the canonical model and stores its checkpoint
without scoring it (`evaluation.final=false`); this scores that checkpoint
on the validation dataset (and `evaluation.train_datasets` as a sanity
check) exactly as a training run scores itself at the end
(`training/scoring.py`), logs the scores to an MLflow run of its own, linked
to the training run's logged model by the checkpoint's content hash, and
writes `<export.dir>/metrics.json`. Splitting the two means a change to the
evaluation code re-scores the stored model in a minute instead of
retraining it.

The test dataset is never opened (see configs/data/re16k.yaml).

Usage:
    uv run dvc repro evaluate
    uv run --extra gpu scripts/evaluation/evaluate_checkpoint.py checkpoint=models/unet/model
"""

from __future__ import annotations

import os
from pathlib import Path

# As in train.py: allocate GPU memory on demand (must be set before jax is
# imported).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

import mhd_surrogate.utils.hydra_resolvers  # noqa: E402, F401  (before @hydra.main resolves)
from mhd_surrogate.training.scoring import run_evaluation  # noqa: E402


@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    run_evaluation(cfg, output_dir / f"{hydra_cfg.job.name}.log")


if __name__ == "__main__":
    main()
