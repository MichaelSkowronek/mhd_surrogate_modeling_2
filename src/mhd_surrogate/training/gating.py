"""Run the canonical model's gate (`evaluation/gate.py`) on a stored
checkpoint: the body of `scripts/evaluation/gate_checkpoint.py`, the DVC
`gate` stage.

It reads the `evaluate` stage's `<export.dir>/metrics.json` (with a
stochastic model's members' own scores), rolls the checkpoint out from the
validation context (each of a stochastic model's `evaluation.ensemble_size`
members) and writes the verdict to `<export.dir>/gate.json`, failing or
not, so a failed gate leaves its numbers behind for a look. The test
dataset is never opened.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from omegaconf import DictConfig, OmegaConf

from mhd_surrogate.data.grid import grid_spacing
from mhd_surrogate.evaluation.gate import gate_failures, member_scores
from mhd_surrogate.evaluation.metrics import member_median
from mhd_surrogate.evaluation.protocol import (
    check_readable,
    check_window,
    ensemble_size,
    split_context,
)
from mhd_surrogate.evaluation.rollout import RolloutStability, rollout_stability
from mhd_surrogate.models.registry import load_model
from mhd_surrogate.training.export import write_metrics
from mhd_surrogate.training.startup import configure_process

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateResult:
    """The rollout of each member (one for a deterministic model), their
    median scores (`RolloutStability.scores`' keys, plus `stable_fraction`,
    the share of members stable over it) and why the gate failed."""

    rollouts: list[RolloutStability]
    rollout_scores: dict[str, float]
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


def run_gate(cfg: DictConfig) -> GateResult:
    """Gate the checkpoint `cfg.checkpoint` against `cfg.evaluation.gate`
    (see `evaluation/gate.py`) and write `<cfg.export.dir>/gate.json`."""
    if not cfg.checkpoint or not cfg.export.dir:
        raise ValueError("checkpoint=<checkpoint directory> and export.dir=<dir> are required")
    configure_process(cfg)

    export = Path(cfg.export.dir)
    metrics = json.loads((export / "metrics.json").read_text())
    val_scores = metrics["val"]
    dataset = cfg.data.val_dataset
    check_readable(dataset, OmegaConf.to_container(cfg.data, resolve=True))
    model = load_model(Path(cfg.checkpoint))
    series = zarr.open_group(store=cfg.data.zarr_store, mode="r")[dataset]
    check_window(model.window, cfg.data.context_steps)
    context, targets = split_context(series, cfg.data.context_steps)
    dx, dy = grid_spacing(series.shape[2], series.shape[3])

    gate, stability = cfg.evaluation.gate, cfg.evaluation.stability
    seeds = range(ensemble_size(model, cfg.evaluation.ensemble_size))
    log.info(
        "rolling %s out for %d steps on %s, %d member(s)",
        model.name,
        gate.rollout_steps,
        dataset,
        len(seeds),
    )
    context = np.asarray(context)
    rollouts = [
        rollout_stability(
            model,
            context,
            targets,
            gate.rollout_steps,
            stability.block_steps,
            stability.max_ratio,
            dx,
            dy,
            seed=seed,
        )
        for seed in seeds
    ]
    scores = [rollout.scores() for rollout in rollouts]
    median = {key: member_median([s[key] for s in scores]) for key in scores[0]}
    stable_steps = [s["stable_steps"] for s in scores]
    median["stable_fraction"] = float(np.mean([s >= gate.rollout_steps for s in stable_steps]))
    failures = gate_failures(
        val_scores,
        member_scores(metrics) or [val_scores],
        targets.shape[0],
        stable_steps,
        gate.rollout_steps,
        dict(gate.max_scores),
        dict(gate.member_max_scores),
        dict(gate.min_scores),
        gate.min_member_fraction,
    )
    verdict = {
        "passed": float(not failures),
        "rollout_steps": float(gate.rollout_steps),
        **{f"rollout_{key}": value for key, value in median.items()},
    }
    write_metrics(export / "gate.json", {"val": verdict})
    log.info("wrote %s", export / "gate.json")
    return GateResult(rollouts, median, failures)
