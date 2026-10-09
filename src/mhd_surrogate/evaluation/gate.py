"""The canonical model's gate: does a retrain still meet the bar?

The DVC pipeline retrains the canonical model whenever its code or params
change, and a retrain is a new draw: the same recipe can come out unstable
(a deterministic retrain of the noise-0.1 U-Net did, 2026-10-08). The `gate`
stage (`scripts/evaluation/gate_checkpoint.py`, after `evaluate`) fails
`dvc repro` unless the stored checkpoint passes what made it canonical
(CLAUDE.md, "Data analysis"):

- stable over the whole validation forecast (its `stable_steps`),
- stable over a long rollout from the validation context
  (`rollout.rollout_stability`, the same blocks and threshold),
- every score in `max_scores` at or below its limit.

The limits are frozen when a model becomes canonical: its validation scores
then plus the margins a replacement may be worse by (0.03 on the energy and
enstrophy errors, 0.05 on the spectrum distances), and the absolute limits
on the domain-mean `u_y` drift. Not the latest
`metrics.json`: limits that followed every retrain would let each one get a
margin worse than the last. Replacing the canonical model stays a human
decision; this checks that a retrain of it still meets the bar that decision
set.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


def gate_failures(
    val_scores: Mapping[str, float],
    scored_steps: int,
    rollout_stable_steps: int,
    rollout_steps: int,
    max_scores: Mapping[str, float],
) -> list[str]:
    """Why the model fails the gate, one line per failed check; empty if it
    passes. `val_scores` are its validation scores (`metrics.json`'s `val`),
    over a forecast of `scored_steps` steps. A score that is missing or NaN
    (undefined, e.g. left out of `metrics.json`) fails its check."""
    failures = []
    stable = val_scores.get("stable_steps", math.nan)
    if not stable >= scored_steps:
        failures.append(f"validation forecast stable for {stable:g} of its {scored_steps} steps")
    if not rollout_stable_steps >= rollout_steps:
        failures.append(f"rollout stable for {rollout_stable_steps} of its {rollout_steps} steps")
    for key, limit in max_scores.items():
        value = val_scores.get(key, math.nan)
        if not value <= limit:
            failures.append(f"{key} {value:.4g} over its limit {limit:g}")
    return failures
