"""Register a tiny model for a smoke deploy.

Fits a mean field (src/mhd_surrogate/models/baselines.py) on random frames
of a small shape, logs it in a throwaway experiment store and promotes it to
the registered model `--name` on `--tracking-uri`
(src/mhd_surrogate/serving/promotion.py), with a validated horizon of
`--steps` and the checkpoint hash `--checkpoint-md5` the CI overlay pins
(deploy/k8s/overlays/ci), as the canonical model is promoted, and points
the `champion` alias at it, which the CI overlay fetches. CI's
deploy-smoke job serves it from a throwaway kind cluster, having neither
the real model nor its data.

Usage:
    python scripts/serving/smoke_model.py --tracking-uri http://localhost:5000
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

import mlflow
from mhd_surrogate.models.baselines import MeanField
from mhd_surrogate.serving.promotion import promote
from mhd_surrogate.training.mlflow_model import log_surrogate
from mhd_surrogate.training.mlflow_utils import log_checkpoint_once
from mhd_surrogate.utils.logging_config import add_log_level_arg, setup_logging

log = logging.getLogger(__name__)

EXPERIMENT = "smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--name", default="mhd-surrogate-smoke")
    parser.add_argument("--frame", type=int, nargs=3, default=[2, 16, 8], metavar=("C", "NX", "NY"))
    parser.add_argument("--steps", type=int, default=100, help="the validated horizon")
    parser.add_argument("--checkpoint-md5", default="smoke-checkpoint")
    add_log_level_arg(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    frame = tuple(args.frame)
    model = MeanField()
    model.fit({"smoke": np.random.default_rng(0).normal(size=(8, *frame)).astype(np.float32)})
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "model"
        model.save(checkpoint)
        source = f"sqlite:///{tmp}/mlruns.db"
        mlflow.set_tracking_uri(source)
        mlflow.MlflowClient().create_experiment(EXPERIMENT, artifact_location=f"{tmp}/artifacts")
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run():
            log_checkpoint_once(checkpoint, lambda: log_surrogate(model, checkpoint, frame))
        promotion = promote(
            checkpoint,
            source,
            EXPERIMENT,
            args.tracking_uri,
            args.name,
            f"{args.name}-promotions",
            alias="champion",
            scores={"gate.rollout_stable_steps": float(args.steps)},
            checkpoint_md5=args.checkpoint_md5,
        )
    print(f"{promotion.name} version {promotion.version} ({promotion.model_id}) is @champion")


if __name__ == "__main__":
    main()
