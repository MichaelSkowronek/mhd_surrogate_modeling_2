"""Gate a stored checkpoint: the DVC `gate` stage.

Runs after the `evaluate` stage and fails (exit status 1, so `dvc repro`
stops and leaves the stage stale) unless the checkpoint still meets the bar
that made it canonical: stable over the whole validation forecast and over
an `evaluation.gate.rollout_steps` rollout from the validation context, and
its validation scores within `evaluation.gate.max_scores` (see
src/mhd_surrogate/evaluation/gate.py). Writes the verdict to
`<export.dir>/gate.json` either way.

The test dataset is never opened (see configs/data/re16k.yaml).

Usage:
    uv run dvc repro gate
    uv run --extra gpu scripts/evaluation/gate_checkpoint.py checkpoint=models/unet/model \\
        export.dir=models/unet
"""

from __future__ import annotations

import logging
import os
import sys

# As in train.py: allocate GPU memory on demand (must be set before jax is
# imported).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

import mhd_surrogate.utils.hydra_resolvers  # noqa: E402, F401  (before @hydra.main resolves)
from mhd_surrogate.training.gating import run_gate  # noqa: E402

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    result = run_gate(cfg)
    scores = result.rollout.scores()
    print(
        f"{result.rollout.n_steps}-step rollout: stable for {scores['stable_steps']:g} steps, "
        f"peak energy {scores['energy_peak_ratio']:.3g}x, "
        f"peak enstrophy {scores['enstrophy_peak_ratio']:.3g}x the truth's"
    )
    if result.passed:
        print("gate passed")
        return
    for failure in result.failures:
        print(f"gate failed: {failure}")
    sys.exit(1)


if __name__ == "__main__":
    main()
