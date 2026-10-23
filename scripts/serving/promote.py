"""Promote the canonical model to the serving registry and pin it.

Part of the pull request that makes a model canonical or retrains it
(README, "Continuous deployment"): after `dvc repro` has trained, scored
and gated it, this copies the gated checkpoint (`<model-dir>/model`) from
the experiment store (`--source`, the local sqlite store) into the MLflow
Model Registry on the tracking server (`--target`) as a version of
`--name` (src/mhd_surrogate/serving/promotion.py), and pins that version,
with the checkpoint's hash as dvc.lock records it, in the overlay Argo CD
deploys (`--pin`, deploy/k8s/overlays/kind/kustomization.yaml). Commit the
pin with the rest: CI checks that the pinned checkpoint is dvc.lock's, so
the merge deploys the model its own commit gated. The `champion` alias
isn't moved here: the deployment moves it once the version serves
(deploy/k8s/smoke-check).

Refuses unless the checkpoint is the gated one: the pipeline up to the
`gate` stage up to date (`dvc status`), dvc.lock and the model's metrics
committed, and the gate passed. A checkpoint registered already keeps its
version. `--no-pin` (and a smoke `--name`) for a smoke run.

Usage:
    docker compose up -d     # the tracking server holding the registry
    uv run dvc repro && git commit ...
    uv run scripts/serving/promote.py && git commit deploy/k8s/overlays/kind/kustomization.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from mhd_surrogate.data.versioning import data_provenance
from mhd_surrogate.serving.promotion import gate_passed, promote
from mhd_surrogate.utils.logging_config import add_log_level_arg, setup_logging

log = logging.getLogger(__name__)

DVC_LOCK = Path("dvc.lock")
# The training data's per-channel statistics (the DVC compute_stats stage),
# the reference for input drift.
STATS = Path("data/processed/normalization_stats.json")
KIND_OVERLAY = Path("deploy/k8s/overlays/kind/kustomization.yaml")

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
    parser.add_argument("--experiment", default="mhd-surrogate-promotions")
    parser.add_argument("--source", type=Path, default=Path("configs/mlflow/local.yaml"))
    parser.add_argument("--target", type=Path, default=Path("configs/mlflow/server.yaml"))
    parser.add_argument(
        "--pin",
        type=Path,
        default=KIND_OVERLAY,
        help="the Kustomize overlay to pin the promoted version in (default: %(default)s)",
    )
    parser.add_argument("--no-pin", dest="pin", action="store_const", const=None)
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


def check_committed(paths: list[Path]) -> tuple[str, str]:
    """The commit promoted from and its tree: HEAD, which must hold `paths`
    as they are. On a pull request it is the head commit, whose tree is the
    one its squash commit on main gets (branch protection keeps the branch
    up to date)."""
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, paths)]).returncode:
        sys.exit(f"commit {', '.join(map(str, paths))} first")
    return tuple(
        subprocess.run(
            ["git", "rev-parse", ref], capture_output=True, text=True, check=True
        ).stdout.strip()
        for ref in ("HEAD", "HEAD^{tree}")
    )


def checkpoint_md5(lock_text: str, checkpoint: Path | str) -> str:
    """The hash dvc.lock (its `lock_text`) records for the stage output
    `checkpoint`."""
    for stage in yaml.safe_load(lock_text)["stages"].values():
        for out in stage.get("outs", []):
            if out["path"] == str(checkpoint):
                return out["md5"]
    raise ValueError(f"dvc.lock has no output {checkpoint}")


URI_PIN = re.compile(r"(MHD_MODEL_URI=models:/)([\w.-]+)/(\w+)")
MD5_PIN = re.compile(r"(MHD_MODEL_CHECKPOINT_MD5=)(\S*)")


def pin_model(text: str, name: str, version: str, md5: str) -> str:
    """A Kustomize overlay's `text` with the model the deployment fetches
    pinned: version `version` of `name` (`MHD_MODEL_URI`) and the
    checkpoint's dvc.lock hash `md5` (`MHD_MODEL_CHECKPOINT_MD5`)."""
    uri = URI_PIN.search(text)
    if uri is None or uri.group(2) != name:
        raise ValueError(f"the overlay doesn't pin a version of {name}")
    if MD5_PIN.search(text) is None:
        raise ValueError("the overlay doesn't pin a checkpoint (MHD_MODEL_CHECKPOINT_MD5)")
    text = URI_PIN.sub(lambda match: f"{match.group(1)}{name}/{version}", text, count=1)
    return MD5_PIN.sub(lambda match: f"{match.group(1)}{md5}", text, count=1)


def pinned_md5(text: str) -> str:
    """The checkpoint hash a Kustomize overlay's `text` pins."""
    match = MD5_PIN.search(text)
    if match is None:
        raise ValueError("the overlay doesn't pin a checkpoint (MHD_MODEL_CHECKPOINT_MD5)")
    return match.group(2)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    metrics, gate = args.model_dir / "metrics.json", args.model_dir / "gate.json"

    check_pipeline(args.stage)
    commit, tree = check_committed([DVC_LOCK, metrics, gate])
    stats = json.loads(STATS.read_text())
    reference = {key: stats[key] for key in ("channel_names", "mean", "std")}
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
    md5 = checkpoint_md5(DVC_LOCK.read_text(), checkpoint)
    log.info("promoting %s to %s on %s", checkpoint, args.name, target["tracking_uri"])
    promotion = promote(
        checkpoint,
        source["tracking_uri"],
        source["experiment_name"],
        target["tracking_uri"],
        args.name,
        args.experiment,
        scores=scores,
        params={**params, "dvc_stage": args.stage, "model_dir": str(args.model_dir)},
        records=[DVC_LOCK, metrics, gate, STATS],
        run_tags={
            "mlflow.source.git.commit": commit,
            "git_tree": tree,
            "mlflow.source.name": "scripts/serving/promote.py",
        },
        reference=reference,
        checkpoint_md5=md5,
    )

    how = "copied and registered" if promotion.copied else "registered already"
    print(f"{promotion.name} version {promotion.version} ({promotion.model_id}): {how}")
    if args.pin is not None:
        text = args.pin.read_text()
        pinned = pin_model(text, promotion.name, promotion.version, md5)
        if pinned == text:
            print(f"{args.pin} pins it already")
        else:
            args.pin.write_text(pinned)
            print(f"pinned {args.pin}: commit it with this pull request")


if __name__ == "__main__":
    main()
