"""Promote the canonical model to the serving registry.

Copies the DVC pipeline's gated checkpoint (`<model-dir>/model`) from the
experiment store (`--source`, the local sqlite store) into the MLflow Model
Registry on the tracking server (`--target`) and points the `champion` alias
at it (src/mhd_surrogate/serving/promotion.py). Refuses unless the
checkpoint is the gated one: the pipeline up to the `gate` stage up to date
(`dvc status`), dvc.lock and the model's metrics committed, at a commit on
`main`, and the gate passed. Promoting a registered checkpoint again only
moves the alias back to its version.

Usage:
    docker compose up -d     # the tracking server holding the registry
    uv run scripts/serving/promote.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

from mhd_surrogate.data.versioning import data_provenance
from mhd_surrogate.serving.promotion import gate_passed, promote
from mhd_surrogate.utils.logging_config import add_log_level_arg, setup_logging

log = logging.getLogger(__name__)

DVC_LOCK = Path("dvc.lock")

# The tracking server advertises presigned multipart transfers for artifacts
# of 500 MB or more, signed for its object store's address inside the
# compose network, which this host can't resolve: send them through the
# server like every other artifact (README, "Tracking stack").
os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "false")
os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=Path("models/hankel_dmd"))
    parser.add_argument("--stage", default="gate", help="the DVC stage that gates the model")
    parser.add_argument("--name", default="mhd-surrogate", help="registered model name")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--experiment", default="mhd-surrogate-promotions")
    parser.add_argument("--source", type=Path, default=Path("configs/mlflow/local.yaml"))
    parser.add_argument("--target", type=Path, default=Path("configs/mlflow/server.yaml"))
    parser.add_argument(
        "--allow-unmerged",
        action="store_true",
        help="allow a commit not on origin/main (smoke runs only, to a smoke registered model)",
    )
    add_log_level_arg(parser)
    return parser.parse_args()


def check_pipeline(stage: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dvc", "status", "--json", "--with-deps", stage],
        capture_output=True,
        text=True,
        check=True,
    )
    status = json.loads(result.stdout)
    if status:
        sys.exit(f"the pipeline up to `{stage}` isn't up to date (run `dvc repro`): {status}")


def check_committed(paths: list[Path], allow_unmerged: bool) -> str:
    """The commit promoted from: HEAD, which must hold `paths` as they are
    and (unless `allow_unmerged`) be on origin/main."""
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, paths)]).returncode:
        sys.exit(f"commit {', '.join(map(str, paths))} first")
    on_main = subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", "origin/main"])
    if on_main.returncode and not allow_unmerged:
        sys.exit("HEAD isn't on origin/main: promote from main (or fetch it first)")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    metrics, gate = args.model_dir / "metrics.json", args.model_dir / "gate.json"

    check_pipeline(args.stage)
    commit = check_committed([DVC_LOCK, metrics, gate], args.allow_unmerged)
    if not gate_passed(gate):
        sys.exit(f"the gate didn't pass: {gate.read_text()}")

    source = yaml.safe_load(args.source.read_text())
    target = yaml.safe_load(args.target.read_text())
    scores = {
        **{f"val.{k}": v for k, v in json.loads(metrics.read_text())["val"].items()},
        **{f"gate.{k}": v for k, v in json.loads(gate.read_text())["val"].items()},
    }
    params = {f"data_version.{k}": v for k, v in data_provenance(DVC_LOCK).items()}
    checkpoint = args.model_dir / "model"
    log.info("promoting %s to %s on %s", checkpoint, args.name, target["tracking_uri"])
    promotion = promote(
        checkpoint,
        source["tracking_uri"],
        source["experiment_name"],
        target["tracking_uri"],
        args.name,
        args.alias,
        args.experiment,
        scores=scores,
        params={**params, "dvc_stage": args.stage, "model_dir": str(args.model_dir)},
        records=[DVC_LOCK, metrics, gate],
        run_tags={
            "mlflow.source.git.commit": commit,
            "mlflow.source.name": "scripts/serving/promote.py",
        },
    )

    how = "copied and registered" if promotion.copied else "already registered, alias moved"
    print(
        f"{promotion.name} version {promotion.version} ({promotion.model_id}) "
        f"is now @{promotion.alias} ({how})"
    )


if __name__ == "__main__":
    main()
