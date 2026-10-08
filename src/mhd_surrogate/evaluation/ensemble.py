"""Probabilistic scores of an ensemble forecast, streamed one member at a time.

A generative surrogate (one that samples, `stochastic = True`) forecasts a
distribution, and its forecast is judged as one: is the truth a plausible
member of the ensemble? Two standard measures, per lead time:

- CRPS (continuous ranked probability score): E|X - y| - E|X - X'| / 2 for
  members X, X' and truth y, averaged over channels and pixels. A proper
  score: it is minimized by forecasting the true distribution, so unlike the
  RMSE it doesn't reward collapsing toward the mean. For a single member it
  *is* the absolute error, so deterministic models get a comparable number.
- Spread vs error: the ensemble's spread (RMS of its per-pixel std) against
  the RMSE of the ensemble mean. For a calibrated ensemble of M members,
  sqrt((M + 1) / M) * spread matches that RMSE (Fortin et al. 2014, "Why
  should ensemble spread match the RMSE of the ensemble mean?"), so the
  corrected ratio is ~1; below 1 the ensemble is overconfident.

Members of an 837-step validation forecast are ~1 GB each, so they're never
all held at once: the pairwise terms use consecutive members only, (X_k,
X_{k-1}). Every such pair is a pair of independent members, so the M - 1 of
them are an unbiased estimate of E|X - X'| and of E(X - X')^2 = 2 Var(X), as
all M (M - 1) / 2 pairs would be, at the cost of some variance. Memory is
the targets, a running sum (for the ensemble mean) and two members.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


class EnsembleAccumulator:
    """Feed it the members of one forecast with `add`, then read `scores()`.

    `targets` and every member are (T, C, Nx, Ny); `scale` is a per-channel
    (C,) divisor, as for `metrics.rmse_per_step`. Work is done `chunk_t`
    lead times at a time, so float64 temporaries stay small.
    """

    def __init__(self, targets: np.ndarray, scale: np.ndarray, chunk_t: int = DEFAULT_CHUNK_T):
        self.targets = targets
        self.scale = np.asarray(scale, dtype=np.float64).reshape(1, -1, 1, 1)
        self.chunk_t = chunk_t
        n = len(targets)
        self.size = 0
        self.abs_error = np.zeros(n)
        self.pair_abs = np.zeros(n)
        self.pair_squared = np.zeros(n)
        self.total: np.ndarray | None = None
        self.previous: np.ndarray | None = None

    def _chunks(self):
        return range(0, len(self.targets), self.chunk_t)

    def _scaled(self, array: np.ndarray, start: int) -> np.ndarray:
        return np.asarray(array[start : start + self.chunk_t], dtype=np.float64) / self.scale

    def add(self, member: np.ndarray) -> None:
        member = np.asarray(member)
        if member.shape != self.targets.shape:
            raise ValueError(f"member shape {member.shape}, expected {self.targets.shape}")
        axes = (1, 2, 3)
        for t in self._chunks():
            x = self._scaled(member, t)
            leads = slice(t, t + len(x))
            self.abs_error[leads] += np.abs(x - self._scaled(self.targets, t)).mean(axis=axes)
            if self.previous is not None:
                difference = x - self._scaled(self.previous, t)
                self.pair_abs[leads] += np.abs(difference).mean(axis=axes)
                self.pair_squared[leads] += (difference**2).mean(axis=axes)
        if self.total is None:
            self.total = member.astype(np.float32, copy=True)
        else:
            self.total += member
        self.previous = member
        self.size += 1

    def scores(self) -> EnsembleScores:
        if self.size == 0:
            raise ValueError("no members added")
        m = self.size
        pairs = max(m - 1, 1)
        crps = self.abs_error / m - 0.5 * self.pair_abs / pairs
        spread = np.sqrt(0.5 * self.pair_squared / pairs)
        mean_squared = np.zeros(len(self.targets))
        for t in self._chunks():
            error = self._scaled(self.total, t) / m - self._scaled(self.targets, t)
            mean_squared[t : t + len(error)] = (error**2).mean(axis=(1, 2, 3))
        return EnsembleScores(
            size=m, crps=crps, spread=spread, ensemble_mean_rmse=np.sqrt(mean_squared)
        )
