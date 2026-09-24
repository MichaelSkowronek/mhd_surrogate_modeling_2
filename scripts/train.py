"""Training entry point.

No model exists yet, so this currently just resolves the Hydra config and
confirms the configured dataset is reachable, as a smoke test of the config
plumbing this will grow into the real training loop on top of.

Usage:
    uv run scripts/train.py
    uv run scripts/train.py data=re16k seed=123
"""

from __future__ import annotations

import logging

import hydra
import zarr
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))

    root = zarr.open_group(store=cfg.data.zarr_store, mode="r")
    arr = root[cfg.data.dataset]
    log.info("dataset %s: shape=%s dtype=%s", cfg.data.dataset, arr.shape, arr.dtype)


if __name__ == "__main__":
    main()
