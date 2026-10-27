"""Probabilistic scores of an ensemble forecast, streamed one member at a time.

A generative surrogate (one that samples, `stochastic = True`) forecasts a
distribution, and its forecast is judged as one: is the truth a plausible
member of the ensemble? Two standard measures, per lead time:

- CRPS (continuous ranked probability score, `metrics.CrpsAccumulator`):
  E|X - y| - E|X - X'| / 2 for members X, X' and truth y, averaged over
  channels and pixels. A proper score, and for a single member the absolute
  error, so deterministic models get a comparable number. The selection
  rule's skill horizon is measured on it, so it lives with the selection
  scores in `metrics.py`.
- Spread vs error: the ensemble's spread (RMS of its per-pixel std) against
  the RMSE of the ensemble mean. For a calibrated ensemble of M members,
  sqrt((M + 1) / M) * spread matches that RMSE (Fortin et al. 2014, "Why
  should ensemble spread match the RMSE of the ensemble mean?"), so the
  corrected ratio is ~1; below 1 the ensemble is overconfident. Reported
  only, so it stays out of what early stopping imports.

Members of an 837-step validation forecast are ~1 GB each, so they're never
all held at once: the pairwise terms use the cycle of members, consecutive
pairs plus the one closing it (see `metrics.CrpsAccumulator`), M pairs of
independent members, an unbiased estimate of E|X - X'| and of
E(X - X')^2 = 2 Var(X) as all M (M - 1) / 2 pairs would be, at the cost of
some variance. Memory is the targets, a running sum (for the ensemble mean)
and three members.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mhd_surrogate.evaluation.metrics import AXES, CrpsAccumulator
from mhd_surrogate.evaluation.quantities import DEFAULT_CHUNK_T


@dataclass(frozen=True)
class EnsembleScores:
    """Per-lead curves, shape (T,) each, index 0 = lead time 1, in units of
    `scale` like the RMSE. `spread` is all zeros for a single member."""

    size: int
    crps: np.ndarray
    spread: np.ndarray
    ensemble_mean_rmse: np.ndarray

    @property
    def spread_skill(self) -> np.ndarray:
        """sqrt((M + 1) / M) * spread / RMSE of the ensemble mean: ~1 for a
        calibrated ensemble (undefined, NaN, for a single member)."""
        if self.size < 2:
            return np.full_like(self.crps, np.nan)
        return np.sqrt((self.size + 1) / self.size) * self.spread / self.ensemble_mean_rmse


class EnsembleAccumulator(CrpsAccumulator):
    """Feed it the members of one forecast with `add`, then read `scores()`.

    `targets` and every member are (T, C, Nx, Ny); `scale` is a per-channel
    (C,) divisor, as for `metrics.rmse_per_step`. Work is done `chunk_t`
    lead times at a time, so float64 temporaries stay small.
    """

    PAIR_TERMS = (*CrpsAccumulator.PAIR_TERMS, "squared")

    def __init__(self, targets: np.ndarray, scale: np.ndarray, chunk_t: int = DEFAULT_CHUNK_T):
        super().__init__(targets, scale, chunk_t)
        self.total: np.ndarray | None = None

    def _pair_terms(self, difference: np.ndarray) -> dict[str, np.ndarray]:
        return {**super()._pair_terms(difference), "squared": (difference**2).mean(axis=AXES)}

    def add(self, member: np.ndarray) -> None:
        super().add(member)
        if self.total is None:
            self.total = np.asarray(member).astype(np.float32, copy=True)
        else:
            self.total += member

    def scores(self) -> EnsembleScores:
        sums, pairs = self._cycle()
        m = self.size
        spread = np.sqrt(0.5 * sums["squared"] / pairs)
        mean_squared = np.zeros(len(self.targets))
        for t in self._chunks():
            error = self._scaled(self.total, t) / m - self._scaled(self.targets, t)
            mean_squared[t : t + len(error)] = (error**2).mean(axis=AXES)
        return EnsembleScores(
            size=m,
            crps=self._crps(sums, pairs),
            spread=spread,
            ensemble_mean_rmse=np.sqrt(mean_squared),
        )
