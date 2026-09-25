"""Training entry point.

No model exists yet, so this currently just resolves the Hydra config,
confirms the configured dataset is reachable, and logs the config and
dataset info to MLflow, as a smoke test of the plumbing this will grow into
the real training loop on top of.

Usage:
    uv run scripts/training/train.py
    uv run scripts/training/train.py data=re16k seed=123
    uv run mlflow ui --backend-store-uri sqlite:///mlruns.db  # view runs
"""

from __future__ import annotations

import logging
import os

import hydra
import mlflow
import zarr
from omegaconf import DictConfig, OmegaConf

from mhd_surrogate.data.dataset import load_dataset
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
        mlflow.log_params(flatten_for_mlflow(OmegaConf.to_container(cfg, resolve=True)))

        root = zarr.open_group(store=cfg.data.zarr_store, mode="r")
        arr = root[cfg.data.dataset]
        log.info("dataset %s: shape=%s dtype=%s", cfg.data.dataset, arr.shape, arr.dtype)
        mlflow.log_params({"data.shape": str(arr.shape), "data.dtype": str(arr.dtype)})

        for split in ("train", "test"):
            ds = load_dataset(
                cfg.dataset.manifest,
                cfg.data.dataset,
                split,
                cfg.dataset.window,
                cfg.dataset.horizon,
                cfg.dataset.stride,
            )
            x, y = ds[0]
            log.info("%s: %d samples, sample shapes x=%s y=%s", split, len(ds), x.shape, y.shape)
            mlflow.log_params(
                {f"{split}.n_samples": len(ds), f"{split}.sample_shape_x": str(x.shape)}
            )


if __name__ == "__main__":
    main()
