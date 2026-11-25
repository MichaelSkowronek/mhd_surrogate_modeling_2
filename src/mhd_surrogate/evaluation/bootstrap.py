"""How much of the multi-start skill is noise: a block bootstrap over the
starts.

The multi-start skill (`selection.start_ensembles`) averages a CRPS curve
over a dataset's starts, every 20 frames of one trajectory of the flow.
Neighbouring starts share its events, so their curves aren't independent
draws; a circular block bootstrap (Politis & Romano 1992) resamples them in
runs of `block` consecutive starts, wrapping around the last, which keeps
that dependence: a resample is ceil(S / block) such runs, cut to the S
starts. Seeded, so a score computed from it reproduces.

- `BlockBootstrap.interval`: the spread of the capped skill horizon over
  the resamples (the final scoring's `multistart_crps_skill_horizon_p05`
  and `_p95`). Reported, not selected on.
- `BlockBootstrap.compare`: two models' per-start curves resampled
  together, the same starts in every resample, and how often each ranks
  first by the selection rule's skill (`metrics.capped_skill`): for
  finalists closer than their intervals
  (`scripts/evaluation/compare_multistart.py`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mhd_surrogate.evaluation.metrics import capped_skill, skill_horizon


@dataclass(frozen=True)
class BlockBootstrap:
    """`resamples` circular block resamples of runs of `block` consecutive
    starts, drawn from `seed` (the defaults: configs/evaluation's
    `multistart.bootstrap`)."""

    block: int = 8
    resamples: int = 1000
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("block", "resamples"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")

    def indices(self, n_starts: int) -> np.ndarray:
        """The starts of every resample, (resamples, n_starts) indices."""
        if n_starts < 1:
            raise ValueError(f"no starts to resample, got {n_starts}")
        runs = -(-n_starts // self.block)
        rng = np.random.default_rng(self.seed)
        first = rng.integers(0, n_starts, size=(self.resamples, runs, 1))
        indices = (first + np.arange(self.block)) % n_starts
        return indices.reshape(self.resamples, -1)[:, :n_starts]

    def averaged(self, curves: np.ndarray) -> np.ndarray:
        """Each resample's curve averaged over its starts, (resamples, L),
        from per-start `curves`, (S, L)."""
        curves = np.asarray(curves, dtype=np.float64)
        return curves[self.indices(len(curves))].mean(axis=1)

    def interval(
        self, curves: np.ndarray, threshold: float, percentiles: tuple[float, float] = (5, 95)
    ) -> tuple[float, float]:
        """The `percentiles` of the skill horizon of the averaged curve, capped
        at its length, over the resamples of the per-start `curves`."""
        horizons = [skill_horizon(curve, threshold) for curve in self.averaged(curves)]
        low, high = np.percentile(horizons, percentiles)
        return float(low), float(high)

    def compare(
        self,
        curves_a: np.ndarray,
        curves_b: np.ndarray,
        threshold: float,
        tie_break_lead: int,
        percentiles: tuple[float, float] = (5, 95),
    ) -> Comparison:
        """How two models' multi-start skills compare over the same
        resamples of their per-start `curves`, (S, L) each, from the same
        starts and leads (`Comparison`)."""
        curves_a, curves_b = (np.asarray(c, dtype=np.float64) for c in (curves_a, curves_b))
        if curves_a.shape != curves_b.shape:
            raise ValueError(
                f"curves of shapes {curves_a.shape} and {curves_b.shape}: not the same starts"
            )
        indices = self.indices(len(curves_a))
        ranks, differences = [], []
        for starts in indices:
            a = capped_skill(curves_a[starts].mean(axis=0), threshold, tie_break_lead)
            b = capped_skill(curves_b[starts].mean(axis=0), threshold, tie_break_lead)
            ranks.append(np.sign(_order(*a) - _order(*b)))
            differences.append(a[0] - b[0])
        ranks = np.asarray(ranks)
        low, high = np.percentile(differences, percentiles)
        return Comparison(
            a_first=float(np.mean(ranks > 0)),
            b_first=float(np.mean(ranks < 0)),
            difference=(float(low), float(high)),
        )


@dataclass(frozen=True)
class Comparison:
    """Of the resamples: the share where model a ranks first by the
    selection rule's skill, where b does (the rest tie), and the
    `percentiles` of a's skill horizon minus b's."""

    a_first: float
    b_first: float
    difference: tuple[float, float]


def _order(horizon: int, tie_break: float) -> float:
    """A number that orders skills like `metrics.selection_score`: by
    horizon, then by the tie-break CRPS, lower first, whose term lies in
    (-1, 0], so it never outweighs a lead of horizon; a NaN CRPS (a blown-up
    forecast) is the worst of all."""
    if math.isnan(tie_break):
        return -math.inf
    return float(horizon) - (1.0 if math.isinf(tie_break) else tie_break / (1.0 + tie_break))
