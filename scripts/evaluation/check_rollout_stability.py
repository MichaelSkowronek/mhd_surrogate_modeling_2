"""Roll a model out far past a dataset's length: does it stay bounded?

The selection rule's stability check (`evaluation/stability.py`) only sees
the scored forecast, 837 steps of the validation dataset. A surrogate is
meant to be rolled out for as long as one likes, so this check rolls the
model out for `--steps` steps from the dataset's context (as the forecast
protocol does), one `--block-steps` block at a time, and prints each block's
mean kinetic energy and enstrophy against the truth's: the ratio of the
block mean to the truth's largest block mean over the scored steps, the
same reference the selection rule uses, so the first blocks reproduce its
`stable_steps`. A block over `--max-ratio` is marked; the summary gives the
first one. A stochastic model is rolled out once per member (`--members`,
seeds 0, 1, ...), and the selection rule needs at least
`evaluation.gate.min_member_fraction` of them (0.8, 7 of 8) stable here and
passing its other absolute guardrails too; the blocks are printed for a
single member, a summary per member for several.

The model comes from MLflow (--model-id) or a checkpoint directory
(--checkpoint). The dataset defaults to the validation one; the test
dataset is refused.

Usage:
    uv run scripts/evaluation/check_rollout_stability.py --checkpoint models/hankel_dmd/model
    uv run --extra gpu scripts/evaluation/check_rollout_stability.py --model-id m-... \\
        --steps 5000 --block-steps 500
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import yaml
import zarr

from mhd_surrogate.data.grid import grid_spacing
from mhd_surrogate.evaluation.metrics import member_median
from mhd_surrogate.evaluation.protocol import (
    check_readable,
    check_window,
    ensemble_size,
    split_context,
)
from mhd_surrogate.evaluation.rollout import rollout_stability
from mhd_surrogate.evaluation.stability import QUANTITIES
from mhd_surrogate.models.registry import load_model
from mhd_surrogate.utils.jax_cache import DEFAULT_CACHE_DIR, enable_compilation_cache
from mhd_surrogate.utils.logging_config import add_log_level_arg, setup_logging

log = logging.getLogger(__name__)

DEFAULT_DATA_CONFIG = Path("configs/data/re16k.yaml")
DEFAULT_EVAL_CONFIG = Path("configs/evaluation/default.yaml")
DEFAULT_TRACKING_URI = "sqlite:///mlruns.db"


def parse_args() -> argparse.Namespace:
    evaluation = yaml.safe_load(DEFAULT_EVAL_CONFIG.read_text())
    stability = evaluation["stability"]
    floor = evaluation["gate"]["min_member_fraction"]
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-id", help="MLflow logged model id (m-...)")
    source.add_argument("--checkpoint", type=Path, help="Checkpoint directory (model.json)")
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help="MLflow tracking URI for --model-id (default: $MLFLOW_TRACKING_URI or %(default)s)",
    )
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--dataset", default=None, help="Default: the validation dataset")
    parser.add_argument("--steps", type=int, default=1500, help="Rollout length")
    parser.add_argument(
        "--block-steps",
        type=int,
        default=stability["block_steps"],
        help="Steps per block (default: the selection rule's, %(default)s)",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=stability["max_ratio"],
        help="Block mean over the truth's largest that counts as unbounded "
        "(default: the selection rule's, %(default)s)",
    )
    parser.add_argument(
        "--members",
        type=int,
        default=evaluation["ensemble_size"],
        help="Members a stochastic model is rolled out for, seeds 0, 1, ...; a deterministic "
        "model has one (default: the evaluation's ensemble size, %(default)s)",
    )
    parser.add_argument(
        "--jax-cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Persistent JAX compilation cache (default: %(default)s)",
    )
    add_log_level_arg(parser)
    args = parser.parse_args()
    args.min_member_fraction = floor
    return args


def load_forecast_model(args: argparse.Namespace):
    if args.checkpoint is not None:
        return load_model(args.checkpoint)
    import mlflow
    from mhd_surrogate.training.mlflow_model import load_logged_surrogate

    mlflow.set_tracking_uri(args.tracking_uri)
    return load_logged_surrogate(args.model_id)[0]


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    enable_compilation_cache(args.jax_cache_dir)
    data_config = yaml.safe_load(args.data_config.read_text())
    dataset = args.dataset or data_config["val_dataset"]
    check_readable(dataset, data_config)
    model = load_forecast_model(args)

    series = zarr.open_group(store=data_config["zarr_store"], mode="r")[dataset]
    dx, dy = grid_spacing(series.shape[2], series.shape[3])
    context_steps = data_config["context_steps"]
    check_window(model.window, context_steps)
    context, targets = split_context(series, context_steps)

    seeds = range(ensemble_size(model, args.members))
    context = np.asarray(context)
    rollouts = []
    for seed in seeds:
        log.info(
            "rolling %s out for %d steps on %s (seed %d)", model.name, args.steps, dataset, seed
        )
        start = time.perf_counter()
        rollouts.append(
            rollout_stability(
                model,
                context,
                targets,
                args.steps,
                args.block_steps,
                args.max_ratio,
                dx,
                dy,
                seed=seed,
            )
        )
        log.info("rollout took %.1f s", time.perf_counter() - start)
    reference = rollouts[0].reference

    print(f"{model.name} on {dataset}: {args.steps}-step rollout from the context")
    for q in QUANTITIES:
        print(
            f"  true {q}, {args.block_steps}-step block means over the scored "
            f"{targets.shape[0]} steps: {reference[q].min():.4g} - {reference[q].max():.4g}"
        )
    if len(rollouts) > 1:
        print(f"{'member':>6} {'stable steps':>12} {'peak energy':>12} {'peak enstrophy':>15}")
        for seed, rollout in zip(seeds, rollouts):
            scores = rollout.scores()
            print(
                f"{seed:>6} {scores['stable_steps']:>12g} {scores['energy_peak_ratio']:>12.3g}"
                f" {scores['enstrophy_peak_ratio']:>15.3g}"
            )
        stable = sum(rollout.stable_steps == args.steps for rollout in rollouts)
        share = stable / len(rollouts)
        print(
            f"{stable} of {len(rollouts)} members stable for all {args.steps} steps ({share:.0%},"
            f" median stable steps {member_median([r.stable_steps for r in rollouts]):g});"
            f" the rule needs {args.min_member_fraction:.0%} of the members stable here and"
            " passing its other absolute guardrails: "
            + ("enough here" if share >= args.min_member_fraction else "too few already")
        )
        return
    print_blocks(rollouts[0], args.steps, args.block_steps, args.max_ratio)
    stable = rollouts[0].stable_steps
    if stable == args.steps:
        print(f"stable for all {args.steps} steps (no block over {args.max_ratio:g}x)")
    else:
        print(
            f"stable for {stable} steps: the block from step {stable} is over {args.max_ratio:g}x"
        )


def print_blocks(rollout, n_steps: int, block_steps: int, max_ratio: float) -> None:
    """One rollout's block means and their ratios to the truth's largest."""
    blocks, ratios = rollout.blocks, rollout.ratios
    print(f"{'steps':>12} {'energy':>10} {'ratio':>8} {'enstrophy':>10} {'ratio':>8}")
    for i, first in enumerate(range(0, n_steps, block_steps)):
        last = min(first + block_steps, n_steps)
        over = any(not ratios[q][i] <= max_ratio for q in QUANTITIES)
        print(
            f"{first:>5} - {last:<5} {blocks['energy'][i]:>10.4g} {ratios['energy'][i]:>8.3g}"
            f" {blocks['enstrophy'][i]:>10.4g} {ratios['enstrophy'][i]:>8.3g}"
            + (f"  > {max_ratio:g}x" if over else "")
        )


if __name__ == "__main__":
    main()
