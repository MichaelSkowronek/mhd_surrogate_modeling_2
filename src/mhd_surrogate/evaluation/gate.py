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
- every score in `member_max_scores` (the domain-mean drift band) at or
  below its limit,
- every score in `max_scores` at or below its limit, and every score in
  `min_scores` at or above its limit.

A stochastic model is checked as the ensemble it is scored as. The first
three are absolute checks of a single forecast -- a member that blows up or
drifts is a broken forecast, served to whoever asks for its seed -- so at
least `min_member_fraction` of the members (0.8: 7 of 8) must pass all
three. A member failing any of them is broken, whichever: what counts is
how many seeds give a usable forecast (checked one by one, a member blowing
up and another drifting would leave 6 of 8; the median member would let 3
of 8 fail). The `max_scores` limits are
relative to the canonical model's scores, a typical member's against it, so
they're checked on the median over the members (`metrics.json`'s `val`),
like the `min_scores` (the stable fraction, already a share). A
deterministic model is a one-member ensemble: each check is its forecast's.

The evaluate stage writes each member's own scores to `metrics.json` too,
in groups `val.member_<seed>` (`member_group`), for an ensemble of more
than one.

The limits are frozen when a model becomes canonical: its validation scores
then plus the margins a replacement may be worse by (0.03 on the energy and
enstrophy errors, 0.05 on the spectrum distances), and the absolute limits
on the domain-mean velocity drift and the stable fraction. Not the latest
`metrics.json`: limits that followed every retrain would let each one get a
margin worse than the last. Replacing the canonical model stays a human
decision; this checks that a retrain of it still meets the bar that decision
set.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def member_group(prefix: str, seed: int) -> str:
    """The `metrics.json` group of an ensemble member's own scores."""
    return f"{prefix}.member_{seed}"


def member_scores(metrics: Mapping[str, Mapping[str, float]], prefix: str = "val") -> list:
    """The members' own scores in `metrics` (`member_group`), by seed; empty
    for a model scored as one forecast."""
    members = []
    while member_group(prefix, len(members)) in metrics:
        members.append(metrics[member_group(prefix, len(members))])
    return members


def gate_failures(
    val_scores: Mapping[str, float],
    members: Sequence[Mapping[str, float]],
    scored_steps: int,
    rollout_stable_steps: Sequence[float],
    rollout_steps: int,
    max_scores: Mapping[str, float],
    member_max_scores: Mapping[str, float] | None = None,
    min_scores: Mapping[str, float] | None = None,
    min_member_fraction: float = 0.8,
) -> list[str]:
    """Why the model fails the gate, one line per failed check; empty if it
    passes. `val_scores` are its validation scores (`metrics.json`'s `val`,
    for a stochastic model the median over its members) and `members` each
    member's own, by seed (`[val_scores]` for a deterministic model), over a
    forecast of `scored_steps` steps; `rollout_stable_steps` are each
    member's over its `rollout_steps`-step rollout. A score that is missing
    or NaN (undefined, e.g. left out of `metrics.json`) fails its check."""
    if len(rollout_stable_steps) != len(members):
        raise ValueError(
            f"{len(members)} members' validation scores but {len(rollout_stable_steps)} rollouts"
        )
    # Why each member is broken (empty: usable), by seed.
    broken = [
        _stable(member.get("stable_steps", math.nan), scored_steps, "validation forecast")
        + _stable(steps, rollout_steps, "rollout")
        + _over(member, member_max_scores or {})
        for member, steps in zip(members, rollout_stable_steps)
    ]
    usable = sum(not reasons for reasons in broken)
    failures: list[str] = []
    if usable < min_member_fraction * len(members):
        if len(members) == 1:
            failures.extend(broken[0])
        else:
            failures.append(
                f"{usable} of {len(members)} members pass every absolute check, under "
                f"{min_member_fraction:.0%} of them: "
                + "; ".join(
                    f"seed {seed}: " + ", ".join(reasons)
                    for seed, reasons in enumerate(broken)
                    if reasons
                )
            )
    failures.extend(_over(val_scores, max_scores))
    for key, limit in (min_scores or {}).items():
        value = val_scores.get(key, math.nan)
        if not value >= limit:
            failures.append(f"{key} {value:.4g} under its limit {limit:g}")
    return failures


def _stable(steps: float, n_steps: int, forecast: str) -> list[str]:
    return [] if steps >= n_steps else [f"{forecast} stable for {steps:g} of its {n_steps} steps"]


def _over(scores: Mapping[str, float], limits: Mapping[str, float]) -> list[str]:
    failures = []
    for key, limit in limits.items():
        value = scores.get(key, math.nan)
        if not value <= limit:
            failures.append(f"{key} {value:.4g} over its limit {limit:g}")
    return failures
