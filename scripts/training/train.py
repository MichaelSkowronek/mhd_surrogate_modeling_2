"""Training entry point.

No model exists yet, so this currently just resolves the Hydra config,
confirms the configured train/val datasets are reachable, and logs the
config and dataset info to MLflow, as a smoke test of the plumbing this
will grow into the real training loop on top of.

Deliberately never opens `cfg.data.test_dataset`, not even to check its
shape -- see configs/data/re16k.yaml's docstring: it's the dataset-level
held-out test set and must never be read by any script, including this
one, until final evaluation.

Usage:
    uv run scripts/training/train.py
    uv run scripts/training/train.py data=re16k seed=123
    uv run scripts/training/train.py mlflow=server  # Docker stack, see README
    uv run mlflow ui --backend-store-uri sqlite:///mlruns.db  # view runs
"""

from __future__ import annotations

import logging
import os

import hydra
import zarr
from omegaconf import DictConfig, OmegaConf

import mlflow
from mhd_surrogate.data.dataset import load_full_dataset
from mhd_surrogate.training.mlflow_utils import flatten_for_mlflow

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))

    # Silences mlflow's "load this tracing skill" hint on every call, which
    # is unrelated to this project's plain params/metrics logging.
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run() as run:
        log.info("mlflow run: %s (experiment: %s)", run.info.run_id, cfg.mlflow.experiment_name)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_params(flatten_for_mlflow(resolved))
        mlflow.log_dict(resolved, "config.json")

        root = zarr.open_group(store=cfg.data.zarr_store, mode="r")
        log.info("test_dataset %s: configured, deliberately not opened", cfg.data.test_dataset)

        for name in [*cfg.data.train_datasets, cfg.data.val_dataset]:
            arr = root[name]
            ds = load_full_dataset(
                cfg.data.zarr_store,
                name,
                cfg.dataset.window,
                cfg.dataset.horizon,
                cfg.dataset.stride,
            )
            x, y = ds[0]
            role = "val" if name == cfg.data.val_dataset else "train"
            log.info(
                "%s (%s): shape=%s dtype=%s, %d windowed samples, sample shapes x=%s y=%s",
                name,
                role,
                arr.shape,
                arr.dtype,
                len(ds),
                x.shape,
                y.shape,
            )
            mlflow.log_params(
                {
                    f"{role}.{name}.shape": str(arr.shape),
                    f"{role}.{name}.n_samples": len(ds),
                }
            )


if __name__ == "__main__":
    main()
