"""Compare runs' multi-start skill with a paired block bootstrap over the
starts.

A run's final scoring logs every start's CRPS curve as the artifact
`multistart/<dataset>.json` (src/mhd_surrogate/training/scoring.py). For
finalists whose horizons are closer than their intervals, this resamples
the same starts for every run (src/mhd_surrogate/evaluation/bootstrap.py)
and prints how often each ranks first by the selection rule's skill: the
capped horizon of the CRPS averaged over the starts, then its tie-break
(`metrics.capped_skill`). The threshold, tie-break lead and bootstrap come
from configs/evaluation/default.yaml, as the scoring's did.

Usage:
    uv run scripts/evaluation/compare_multistart.py RUN_ID RUN_ID [RUN_ID ...]
    uv run scripts/evaluation/compare_multistart.py RUN_ID RUN_ID --dataset train.re16k_t400_0
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

import mlflow
from mhd_surrogate.evaluation.bootstrap import BlockBootstrap
from mhd_surrogate.evaluation.metrics import capped_skill
from mhd_surrogate.utils.logging_config import add_log_level_arg, setup_logging
from mlflow import MlflowClient

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="MLflow run ids (at least two)")
    parser.add_argument("--dataset", default="val", help="val or train.<dataset>")
    parser.add_argument("--tracking-uri", default="sqlite:///mlruns.db")
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/default.yaml"))
    add_log_level_arg(parser)
    args = parser.parse_args()
    setup_logging(args.log_level)
    if len(args.runs) < 2:
        sys.exit("compare at least two runs")

    config = yaml.safe_load(args.config.read_text())
    threshold, tie_break_lead = config["skill_threshold"], config["tie_break_lead"]
    bootstrap = BlockBootstrap(**config["multistart"]["bootstrap"])
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(args.tracking_uri)
    names, curves, starts = [], [], None
    for run_id in args.runs:
        path = f"multistart/{args.dataset}.json"
        log.info("reading %s of run %s", path, run_id)
        record = mlflow.artifacts.load_dict(f"runs:/{run_id}/{path}")
        if starts is not None and record["starts"] != starts:
            sys.exit(f"run {run_id} was scored from other starts: {record['starts']}")
        starts = record["starts"]
        names.append(f"{client.get_run(run_id).info.run_name} ({run_id[:8]})")
        curves.append(np.asarray(record["crps"]))

    print(
        f"{args.dataset}: {len(starts)} starts ({starts[0]}-{starts[-1]}), "
        f"{curves[0].shape[1]} leads; {bootstrap.resamples} resamples of runs of "
        f"{bootstrap.block} starts"
    )
    for name, curve in zip(names, curves):
        horizon, tie_break = capped_skill(curve.mean(axis=0), threshold, tie_break_lead)
        low, high = bootstrap.interval(curve, threshold)
        print(f"  {name}: horizon {horizon} [{low:g}, {high:g}], tie-break CRPS {tie_break:.4f}")
    for (a, curve_a), (b, curve_b) in itertools.combinations(zip(names, curves), 2):
        result = bootstrap.compare(curve_a, curve_b, threshold, tie_break_lead)
        low, high = result.difference
        print(
            f"  {a} vs {b}: first in {result.a_first:.0%} / {result.b_first:.0%} of the "
            f"resamples, horizon difference [{low:g}, {high:g}]"
        )


if __name__ == "__main__":
    main()
