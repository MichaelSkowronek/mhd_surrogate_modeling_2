"""Whether a rollout is usable (CLAUDE.md, "Data analysis"): stable over its
whole length and, over its first `window_steps` steps, in the
climatological drift band.

- Stable: no `block_steps`-step block whose mean energy or enstrophy
  exceeds `max_ratio` times the validation truth's largest block mean
  (`stability.py`'s blocks and threshold, now over a rollout longer than
  the truth).
- In the band: each channel's domain-mean velocity
  (`quantities.device_domain_mean`) over the rollout's first
  `window_steps` steps is at most `max_offset` units from the real flow's
  level and its std over time at most `max_std_ratio` units, the
  climatology's (`data.climatology.Climatology`, from the training
  datasets' windows of that length).

The reference is the climatology rather than the validation truth because
the truth covers frames 80-916 only: a rollout from the end of the context
spans those frames in its first 837 steps, one from a later start doesn't,
so a fixed truth window would judge later starts against frames they don't
cover, while the real bulk flow declines through every record (by ~0.6
units per 1000 frames). The climatology judges every start's rollout alike.
The first `window_steps` steps only, because they cover times since the
records' start that the records still span; whether the decline goes on
past 1248 frames can't be checked.

Also scored, logged but neither ranked nor gated (decided at the canonical
switch): the band over the windows that cover the whole rollout, the
slow-variability ratio (how much the bulk flow wanders on slow time scales
against the real flow) and the energy and enstrophy errors over the first
window against the validation truth.

Early stopping imports this module (`selection.py`), so the DVC `train`
stage depends on it: NumPy and the device reductions only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

import jax
import numpy as np

from mhd_surrogate.data.climatology import Climatology, block_std
from mhd_surrogate.evaluation.quantities import device_domain_mean, device_energy_and_enstrophy
from mhd_surrogate.evaluation.stability import QUANTITIES, block_means, first_failing_step


@dataclass(frozen=True)
class UsableCheck:
    """What makes a rollout usable: `steps` steps stable against `max_ratio`
    times the truth's largest `block_steps`-step block mean, and the first
    `climatology.window_steps` in the band (`max_offset`, `max_std_ratio`
    units of the `climatology`). `slow_from` is where the slow-variability
    ratio's windows start (past the rollout's initial transient)."""

    climatology: Climatology
    steps: int = 3000
    block_steps: int = 100
    max_ratio: float = 2.0
    max_offset: float = 1.0
    max_std_ratio: float = 2.0
    slow_from: int = 300

    def __post_init__(self) -> None:
        if self.steps < self.climatology.window_steps:
            raise ValueError(
                f"a rollout of {self.steps} steps is shorter than the band's "
                f"{self.climatology.window_steps}-step window"
            )
        for name in ("block_steps", "max_ratio", "max_offset", "max_std_ratio"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.slow_from < 0:
            raise ValueError(f"slow_from must be >= 0, got {self.slow_from}")


@partial(jax.jit, static_argnames=("dx", "dy"))
def _reduced(frames: jax.Array, dx: float, dy: float) -> tuple[jax.Array, ...]:
    energy, ens = device_energy_and_enstrophy(frames, dx, dy)
    return (energy, ens, *device_domain_mean(frames))


class DeviceRolloutQuantities:
    """A rollout's per-step energy, enstrophy and domain-mean velocity, fed
    a block of frames on the device at a time, in order (`add`), each
    reduced there as it comes; `series()` returns them on the host in
    float64: `energy` and `enstrophy` (T,), `mean_velocity` (T, C)."""

    def __init__(self, dx: float, dy: float) -> None:
        self.dx, self.dy = dx, dy
        self._blocks: list = []

    def add(self, frames) -> None:
        self._blocks.append(_reduced(frames, self.dx, self.dy))

    def series(self) -> dict[str, np.ndarray]:
        if not self._blocks:
            raise ValueError("no frames added")
        blocks = [[np.asarray(part, np.float64) for part in block] for block in self._blocks]
        return {
            "energy": np.concatenate([b[0] for b in blocks]),
            "enstrophy": np.concatenate([b[1] for b in blocks]),
            "mean_velocity": np.concatenate([b[2] + b[3] for b in blocks]),
        }


def band_scores(mean_velocity: np.ndarray, climatology: Climatology) -> dict[str, float]:
    """Over a (T, C) domain-mean series: per channel `<name>_clim_offset`,
    |its mean - the climatology's level| in the climatology's units, and
    `<name>_clim_std_ratio`, its std over time in those units (NaN for a
    rollout that blew up)."""
    mean_velocity = np.asarray(mean_velocity, dtype=np.float64)
    level, unit = np.array(climatology.level), np.array(climatology.unit)
    offset = np.abs(mean_velocity.mean(axis=0) - level) / unit
    ratio = mean_velocity.std(axis=0) / unit
    scores = {}
    for c, name in enumerate(climatology.channel_names):
        scores[f"{name}_clim_offset"] = float(offset[c])
        scores[f"{name}_clim_std_ratio"] = float(ratio[c])
    return scores


def in_band(scores: Mapping[str, float], check: UsableCheck) -> bool:
    """Whether `band_scores` are within the check's limits for every channel
    (a NaN is outside)."""
    return all(
        scores[f"{name}_clim_offset"] <= check.max_offset
        and scores[f"{name}_clim_std_ratio"] <= check.max_std_ratio
        for name in check.climatology.channel_names
    )


def covering_windows(n_steps: int, window_steps: int) -> list[int]:
    """The first steps of consecutive `window_steps`-step windows covering
    `n_steps` steps, the last one ending with them (overlapping the one
    before): 0, 837, 1674 and 2163 for 3000 steps of 837."""
    if not 1 <= window_steps <= n_steps:
        raise ValueError(f"no {window_steps}-step window fits {n_steps} steps")
    starts = list(range(0, n_steps - window_steps + 1, window_steps))
    if starts[-1] + window_steps < n_steps:
        starts.append(n_steps - window_steps)
    return starts


def slow_std_ratios(
    mean_velocity: np.ndarray, climatology: Climatology, first: int
) -> dict[str, float]:
    """Per channel `<name>_slow_std_ratio`: how much the domain mean wanders
    on slow time scales against the real flow -- the std of its
    `block_steps`-step block means within each consecutive
    `window_steps`-step window from step `first` on, averaged over the
    windows, over the climatology's `slow_std` (the same statistic of the
    real windows, so both sides span the same time). NaN if no window fits
    after `first`."""
    window, block = climatology.window_steps, climatology.block_steps
    starts = range(first, len(mean_velocity) - window + 1, window)
    names = climatology.channel_names
    if not starts:
        return {f"{name}_slow_std_ratio": float("nan") for name in names}
    stds = np.mean([block_std(mean_velocity[s : s + window], block) for s in starts], axis=0)
    return {
        f"{name}_slow_std_ratio": float(std / reference)
        for name, std, reference in zip(names, stds, climatology.slow_std)
    }


def rollout_scores(
    rollout: Mapping[str, np.ndarray], truth: Mapping[str, np.ndarray], check: UsableCheck
) -> dict[str, float]:
    """A rollout's scores (see the module docstring): `rollout` as
    `DeviceRolloutQuantities.series` returns it for its `check.steps`
    steps, `truth` the validation truth's per-step `QUANTITIES` over its
    scored frames.

    - `stable_steps`, `energy_peak_ratio`, `enstrophy_peak_ratio`: as
      `stability.stability_scores` defines them, over the whole rollout.
    - The `band_scores` of its first window and `in_band`, then `usable`
      (stable over all `check.steps` and in the band), 1.0 or 0.0.
    - Logged only: `windows_in_band`, the share of the `covering_windows`
      in the band; `<channel>_largest_clim_offset` over them; the
      `slow_std_ratios`; `energy_rel_error` and `enstrophy_rel_error` of
      the first window's mean against the truth's.
    """
    n = len(rollout["energy"])
    if n != check.steps:
        raise ValueError(f"a rollout of {n} steps, expected {check.steps}")
    ratios = {
        q: block_means(rollout[q], check.block_steps)
        / block_means(truth[q], check.block_steps).max()
        for q in QUANTITIES
    }
    stable = first_failing_step(ratios, check.block_steps, check.max_ratio, n)
    climatology, window = check.climatology, check.climatology.window_steps
    mean_velocity = rollout["mean_velocity"]
    band = band_scores(mean_velocity[:window], climatology)
    banded = in_band(band, check)
    windows = [
        band_scores(mean_velocity[s : s + window], climatology) for s in covering_windows(n, window)
    ]
    scores = {
        "stable_steps": float(stable),
        **{f"{q}_peak_ratio": float(np.max(r)) for q, r in ratios.items()},
        **band,
        "in_band": float(banded),
        "usable": float(banded and stable >= n),
        "windows_in_band": float(np.mean([in_band(w, check) for w in windows])),
        **{
            f"{name}_largest_clim_offset": float(
                np.max([w[f"{name}_clim_offset"] for w in windows])
            )
            for name in climatology.channel_names
        },
        **slow_std_ratios(mean_velocity, climatology, check.slow_from),
    }
    for q in QUANTITIES:
        true_mean = np.mean(truth[q])
        scores[f"{q}_rel_error"] = float(abs(np.mean(rollout[q][:window]) - true_mean) / true_mean)
    return scores
