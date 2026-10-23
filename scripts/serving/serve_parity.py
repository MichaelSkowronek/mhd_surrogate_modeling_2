"""Check that the forecast service serves the canonical model's own forecast:
the DVC `serve_parity` stage.

Runs the checkpoint's validation forecast through the service in process
and compares it with the forecast under the evaluation protocol
(src/mhd_surrogate/serving/parity.py), on the CPU, as the deployed pod
forecasts. Writes the verdict to `<export.dir>/serve_parity.json` either
way and fails (exit status 1, so `dvc repro` stops) if they differ by more
than `serving.parity_tolerance` of the truth's standard deviation. The test
dataset is never opened.

Usage:
    uv run dvc repro serve_parity
    JAX_PLATFORMS=cpu uv run scripts/serving/serve_parity.py \\
        checkpoint=models/hankel_dmd/model export.dir=models/hankel_dmd
"""

from __future__ import annotations

import logging
import sys

import hydra
from omegaconf import DictConfig

import mhd_surrogate.utils.hydra_resolvers  # noqa: F401  (before @hydra.main resolves)
from mhd_surrogate.serving.parity import run_parity

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    result = run_parity(cfg)
    print(
        f"served vs offline forecast, {result.steps} steps: max difference "
        f"{result.max_difference:.3g} of the truth's std (tolerance {result.tolerance:g})"
    )
    if not result.passed:
        print("serve parity failed: the service doesn't serve the model's own forecast")
        sys.exit(1)
    print("serve parity passed")


if __name__ == "__main__":
    main()
