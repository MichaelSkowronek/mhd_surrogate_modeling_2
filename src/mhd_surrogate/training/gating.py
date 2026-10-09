"""Run the canonical model's gate (`evaluation/gate.py`) on a stored
checkpoint: the body of `scripts/evaluation/gate_checkpoint.py`, the DVC
`gate` stage.

It reads the `evaluate` stage's `<export.dir>/metrics.json`, rolls the
checkpoint out from the validation context and writes the verdict to
`<export.dir>/gate.json`, failing or not, so a failed gate leaves its
numbers behind for a look. The test dataset is never opened.
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
from mhd_surrogate.evaluation.gate import gate_failures
from mhd_surrogate.evaluation.protocol import check_readable, check_window, split_context
from mhd_surrogate.evaluation.rollout import RolloutStability, rollout_stability
from mhd_surrogate.models.registry import load_model
from mhd_surrogate.training.export import write_metrics
from mhd_surrogate.utils.jax_cache import enable_compilation_cache
from mhd_surrogate.utils.jax_determinism import enable_deterministic_ops

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateResult:
    rollout: RolloutStability
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


def run_gate(cfg: DictConfig) -> GateResult:
    """Gate the checkpoint `cfg.checkpoint` against `cfg.evaluation.gate`
    (see `evaluation/gate.py`) and write `<cfg.export.dir>/gate.json`."""
    if not cfg.checkpoint or not cfg.export.dir:
        raise ValueError("checkpoint=<checkpoint directory> and export.dir=<dir> are required")
    # Before anything is jitted / the backend starts, as in run.py.
    if cfg.jax.compilation_cache_dir:
        enable_compilation_cache(cfg.jax.compilation_cache_dir, cfg.jax.compilation_cache_max_gb)
    if cfg.jax.get("deterministic_ops", False):
        enable_deterministic_ops()

    export = Path(cfg.export.dir)
    val_scores = json.loads((export / "metrics.json").read_text())["val"]
    dataset = cfg.data.val_dataset
    check_readable(dataset, OmegaConf.to_container(cfg.data, resolve=True))
    model = load_model(Path(cfg.checkpoint))
    series = zarr.open_group(store=cfg.data.zarr_store, mode="r")[dataset]
    check_window(model.window, cfg.data.context_steps)
    context, targets = split_context(series, cfg.data.context_steps)
    dx, dy = grid_spacing(series.shape[2], series.shape[3])

    gate, stability = cfg.evaluation.gate, cfg.evaluation.stability
    log.info("rolling %s out for %d steps on %s", model.name, gate.rollout_steps, dataset)
    rollout = rollout_stability(
        model,
        np.asarray(context),
        targets,
        gate.rollout_steps,
        stability.block_steps,
        stability.max_ratio,
        dx,
        dy,
    )
    failures = gate_failures(
        val_scores,
        targets.shape[0],
        rollout.stable_steps,
        gate.rollout_steps,
        dict(gate.max_scores),
    )
    verdict = {
        "passed": float(not failures),
        "rollout_steps": float(gate.rollout_steps),
        **{f"rollout_{key}": value for key, value in rollout.scores().items()},
    }
    write_metrics(export / "gate.json", {"val": verdict})
    log.info("wrote %s", export / "gate.json")
    return GateResult(rollout, failures)
