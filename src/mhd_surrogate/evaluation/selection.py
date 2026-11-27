"""The selection rule's scores, cheap enough to run every epoch.

Early stopping and the hyperparameter search score a model in training by
`selection_scores`, and choose between a run's validations with
`RunSelection`; `evaluate.evaluate` scores the final model in full, its
multi-start skill by the same `start_ensembles`. In a module of its own
(with `metrics`, `protocol`, `stability`, `usable` and `quantities`), so
that training -- and the DVC `train` stage's dependencies -- reach only
this part of the evaluation code: a change to the diagnostics or the
ensemble's spread re-runs the `evaluate` stage on the stored checkpoint
instead of retraining the model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np

from mhd_surrogate.evaluation.metrics import (
    DeviceCrpsAccumulator,
    capped_skill,
    member_median,
    selection_score,
    skill_horizon,
)
from mhd_surrogate.evaluation.protocol import (
    ForecastModel,
    StartBlocks,
    ensemble_size,
    rollout_seed,
    skill_starts,
    split_context,
    start_seeds,
)
from mhd_surrogate.evaluation.quantities import DEFAULT_CHUNK_T, to_device
from mhd_surrogate.evaluation.stability import DeviceQuantities
from mhd_surrogate.evaluation.usable import DeviceRolloutQuantities, UsableCheck, rollout_scores

# A rollout's scores counted rather than averaged over a validation's rollouts.
_COUNTED = ("stable_steps", "in_band", "usable")
# What RunSelection needs of a validation to judge it with its window.
_WINDOW_KEYS = (
    "multistart_crps_skill_horizon",
    "selection_tie_break",
    "rollout_steps",
    "rollouts",
    "usable_rollouts",
)


def selection_scores(
    model: ForecastModel,
    series,
    context_steps: int,
    scale: np.ndarray,
    skill_threshold: float,
    tie_break_lead: int,
    dx: float,
    dy: float,
    stride: int,
    leads: int,
    check: UsableCheck,
    rollout_starts: Sequence[int] = (0,),
    validation: int = 0,
    threshold: float = 0.8,
    members: int = 1,
    member_batch: int = 1,
    chunk_t: int = DEFAULT_CHUNK_T,
) -> dict[str, float]:
    """The scores the selection rule ranks by, without the relative physics
    guardrails: the multi-start skill, the rollouts' stability and drift
    band, their combination `selection_score` (of this validation judged
    alone, by its own rollouts' usable fraction against `threshold`;
    `RunSelection` judges it with its neighbors' rollouts too) and what
    `RunSelection` needs. Cheap enough to run during training (early stopping, the
    hyperparameter search); `evaluate` scores the final model in full.
    Every forecast is scored on the device as it streams, `chunk_t` lead
    times at a time, so its frames never reach the host.

    The skill comes from many starts (CLAUDE.md, "Data analysis"): every
    `stride` frames from the context's end, as many as fit `leads` lead
    times (`protocol.skill_starts`), each forecast for those leads and
    scored against the truth at the same frames (`start_ensembles`). Its
    CRPS averaged per lead over the starts gives
    `multistart_crps_skill_horizon` (capped at `leads`) and the tie-break
    (`metrics.capped_skill`): `multistart_crps_lead_<tie_break_lead>` below
    the cap, `multistart_crps_mean` at it (`selection_tie_break`, whichever
    applies). The first start is the context's
    end: its own skill is reported under the old names
    (`crps_skill_horizon`, `crps_lead_<tie_break_lead>`).

    The stability comes from rollouts of `check.steps` steps
    (`monitor_rollouts`), one from each of `rollout_starts` frames after
    the context's end, each scored by `usable.rollout_scores`:
    `rollout_stable_steps` and `rollout_peak_ratio` (the larger of a
    rollout's two, NaN if either is) are the median over them
    (`metrics.member_median`), the rest of their scores logged as
    `rollout_<name>` medians, and `rollouts`, `usable_rollouts` and
    `in_band_rollouts` count them; each one's `rollout_<k>_stable_steps`
    and `rollout_<k>_peak_ratio` are kept for `RunSelection`, which pools
    them over a window.

    A stochastic model is scored as an ensemble of `members` at every start
    (the validations' seeds, `protocol.start_seeds`: each start's own, never
    the scoring's, the same at every validation; a deterministic model has
    one member, whatever `members`), and rolled out with a new seed at each
    `validation` (`protocol.rollout_seed`), sampled `member_batch` per call
    where the model can (several starts' at once by a model with
    `predict_device_batch`)."""
    starts = skill_starts(series.shape[0], context_steps, stride, leads)
    if not 1 <= tie_break_lead <= leads:
        raise ValueError(f"tie_break_lead {tie_break_lead} is not one of the {leads} skill leads")
    size = ensemble_size(model, members)
    seeds = start_seeds(len(starts), size, monitor=True)
    curves = [
        start.crps()
        for start in start_ensembles(
            model, series, starts, leads, seeds, scale, DeviceCrpsAccumulator, member_batch, chunk_t
        )
    ]
    averaged = np.mean(curves, axis=0)
    skill, tie_break = capped_skill(averaged, skill_threshold, tie_break_lead)
    rollouts = monitor_rollouts(
        model,
        series,
        context_steps,
        stride,
        rollout_starts,
        validation,
        size,
        check,
        dx,
        dy,
        member_batch,
        chunk_t,
    )
    peaks = [float(np.max([r["energy_peak_ratio"], r["enstrophy_peak_ratio"]])) for r in rollouts]
    stable = member_median([r["stable_steps"] for r in rollouts])
    peak = member_median(peaks)
    usable = sum(r["usable"] for r in rollouts)
    logged = {
        f"rollout_{key}": member_median([r[key] for r in rollouts])
        for key in rollouts[0]
        if key not in _COUNTED
    }
    return {
        "multistart_crps_skill_horizon": float(skill),
        f"multistart_crps_lead_{tie_break_lead}": float(averaged[tie_break_lead - 1]),
        "multistart_crps_mean": float(averaged.mean()),
        "crps_skill_horizon": float(skill_horizon(curves[0], skill_threshold)),
        f"crps_lead_{tie_break_lead}": float(curves[0][tie_break_lead - 1]),
        **logged,
        "rollout_stable_steps": float(stable),
        "rollout_peak_ratio": float(peak),
        "rollout_steps": float(check.steps),
        **{f"rollout_{k}_stable_steps": r["stable_steps"] for k, r in enumerate(rollouts)},
        **{f"rollout_{k}_peak_ratio": value for k, value in enumerate(peaks)},
        "rollouts": float(len(rollouts)),
        "in_band_rollouts": float(sum(r["in_band"] for r in rollouts)),
        "usable_rollouts": float(usable),
        "selection_tie_break": float(tie_break),
        "selection_score": selection_score(
            skill,
            tie_break,
            stable,
            check.steps,
            peak,
            usable / len(rollouts),
            threshold,
            n_forecasts=len(rollouts),
        ),
        "members": float(size),
    }


def monitor_rollouts(
    model: ForecastModel,
    series,
    context_steps: int,
    stride: int,
    offsets: Sequence[int],
    validation: int,
    members: int,
    check: UsableCheck,
    dx: float,
    dy: float,
    batch_size: int = 1,
    chunk_t: int = DEFAULT_CHUNK_T,
) -> list[dict[str, float]]:
    """A validation's rollouts: one of `check.steps` steps from each of
    `offsets` frames after the context's end of `series` (multiples of
    `stride`, so each starts at the k-th multi-start start and draws from
    its seeds: `protocol.rollout_seed` for `validation`, past the `members`
    skill members), streamed on the device (`protocol.StartBlocks`,
    `batch_size` per call where the model can) and scored against the
    truth's scored frames (`usable.rollout_scores`)."""
    if not offsets:
        raise ValueError("no rollout starts")
    starts, seeds = [], []
    for offset in offsets:
        if offset < 0 or offset % stride:
            raise ValueError(
                f"a rollout {offset} frames after the context's end starts at no multi-start "
                f"start (every {stride} frames)"
            )
        starts.append(context_steps + offset)
        seeds.append([rollout_seed(offset // stride, members, validation)])
    _, targets = split_context(series, context_steps)
    truth = DeviceQuantities(dx, dy)
    for first in range(0, targets.shape[0], chunk_t):
        truth.add(to_device(targets[first : first + chunk_t]))
    truth = truth.series()
    quantities = [DeviceRolloutQuantities(dx, dy) for _ in starts]
    for block in StartBlocks(model, series, starts, check.steps, seeds, chunk_t, batch_size):
        for k, ensemble in block.ensembles.items():
            for frames in ensemble:
                quantities[k].add(frames)
    return [rollout_scores(q.series(), truth, check) for q in quantities]


def start_ensembles(
    model: ForecastModel,
    frames,
    starts: Sequence[int],
    leads: int,
    seeds: Iterable[Iterable[int]],
    scale: np.ndarray,
    accumulator: Callable = DeviceCrpsAccumulator,
    batch_size: int = 1,
    chunk_t: int = DEFAULT_CHUNK_T,
) -> list:
    """The multi-start skill's forecasts, scored: an `accumulator(leads,
    scale)` per start of `starts` (`metrics.DeviceCrpsAccumulator`, or
    `ensemble.DeviceEnsembleAccumulator` for the spread too), fed that
    start's members for its own `seeds` (a list per start,
    `protocol.start_seeds`) over `leads` lead times against the truth
    at the same frames of `frames`, as they stream on the device
    (`protocol.StartBlocks`, `batch_size` trajectories per call where the
    model can)."""
    late = [s for s in starts if s + leads > frames.shape[0]]
    if late:
        raise ValueError(
            f"starts {late} have no truth for {leads} leads in {frames.shape[0]} frames"
        )
    stream = StartBlocks(model, frames, starts, leads, seeds, chunk_t, batch_size)
    accumulators = [accumulator(leads, scale) for _ in stream.starts]
    # Every start's truth on the device at once (837 frames on the
    # validation dataset, ~1 GB): copied a block per start instead, the
    # starts' overlapping leads would move ~4.7 GB per validation (~1 s).
    first = min(starts)
    truth = to_device(frames[first : max(starts) + leads])
    for block in stream:
        for k, members in block.ensembles.items():
            lead = stream.starts[k] - first + block.first
            accumulators[k].add_block(block.first, truth[lead : lead + block.steps], members)
    return accumulators


class RunSelection:
    """The selection rule over one training run's validations: which one's
    network early stopping keeps, and the run's score in the search.

    A network whose rollouts are usable at one epoch and not at the next is
    a lucky draw: a refit for the same number of epochs (the final train +
    val refit, a retrain) is likely to land on an unusable one. And each
    validation rolls out with new seeds, so its own two rollouts are
    largely luck too (both survive with probability p^2 for a per-rollout
    survival p: 27% for run 1). So a validation is judged with the
    `2 * neighbors + 1` validations nearest it (itself included), shifted
    inward at the ends of the run so that every validation is judged with
    as many (fewer only while the run has fewer), their rollouts pooled:
    its `usable_fraction` is the share of them usable (stable over their
    whole length and in the drift band, `usable.rollout_scores`), and its
    score is `metrics.selection_score` with that fraction against
    `threshold` as the first key, then the median of those rollouts' stable
    steps (and, for an unstable median, of their peak ratios), then the
    validation's own skill -- its members' seeds are the same at every
    validation, so on skill epochs compare like with like. A validation
    none of whose own rollouts is usable doesn't count as reaching the
    threshold (`metrics.usable_fraction_bonus`): its window's fraction
    speaks for its neighbors' networks, and the best skill among the
    windows reaching it is where a less stable network would win. It is
    kept only if no validation reaches the threshold with a usable rollout
    of its own.

    Each validation's scores need `selection_score` and, to be judged with
    a window (`neighbors` above 0), `multistart_crps_skill_horizon`,
    `selection_tie_break`, `rollout_steps`, `rollouts`, `usable_rollouts`
    and every rollout's `rollout_<k>_stable_steps` and
    `rollout_<k>_peak_ratio` (`selection_scores` gives them all). With
    `neighbors` 0 a validation is judged alone: its `selection_score`.

    Validations with the same window (the first and the last `neighbors +
    1` of a run share theirs) are ordered by their scores alone (`rank`).

    A validation's window settles once `neighbors` more validations follow
    it (and the run has a full window); until then a later validation can
    still change its score, so the best validation so far is provisional:
    the one early stopping would keep if the run ended now.
    """

    def __init__(
        self,
        neighbors: int,
        history: Iterable[Mapping[str, float]] = (),
        threshold: float = 0.8,
    ) -> None:
        if neighbors < 0:
            raise ValueError(f"neighbors must be >= 0, got {neighbors}")
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self.neighbors = neighbors
        self.threshold = threshold
        self.history = [dict(scores) for scores in history]

    def __len__(self) -> int:
        return len(self.history)

    def add(self, scores: Mapping[str, float]) -> None:
        """Record the next validation's scores."""
        self.history.append(dict(scores))

    @property
    def size(self) -> int:
        """How many validations a full window holds."""
        return 2 * self.neighbors + 1

    def window(self, index: int) -> range:
        """The validations `index` is judged with, as they stand."""
        m = min(self.size, len(self))
        start = min(max(index - self.neighbors, 0), len(self) - m)
        return range(start, start + m)

    def settled(self, index: int) -> bool:
        """Whether `index`'s window (so its score) is final."""
        return len(self) >= max(self.size, index + self.neighbors + 1)

    def rollouts(self, index: int) -> tuple[float, float]:
        """(usable rollouts, rollouts) of validation `index`."""
        scores = self.history[index]
        return scores["usable_rollouts"], scores["rollouts"]

    def usable_fraction(self, index: int) -> float:
        counts = [self.rollouts(i) for i in self.window(index)]
        return sum(usable for usable, _ in counts) / sum(total for _, total in counts)

    def score(self, index: int) -> float:
        """The validation's `selection_score`, judged with its window."""
        scores = self.history[index]
        if self.neighbors == 0:
            return scores["selection_score"]
        missing = [k for k in _WINDOW_KEYS if k not in scores]
        if missing:
            raise ValueError(f"validation {index} lacks {missing} to be judged with its window")
        window = self.window(index)
        stable, peaks = [], []
        for i in window:
            other = self.history[i]
            for k in range(int(other["rollouts"])):
                stable.append(other[f"rollout_{k}_stable_steps"])
                peaks.append(other[f"rollout_{k}_peak_ratio"])
        return selection_score(
            scores["multistart_crps_skill_horizon"],
            scores["selection_tie_break"],
            member_median(stable),
            int(scores["rollout_steps"]),
            member_median(peaks),
            self.usable_fraction(index),
            self.threshold,
            # Rollouts in a full window: the run's rollouts per validation
            # are fixed.
            n_forecasts=self.size * int(scores["rollouts"]),
            own_usable=scores["usable_rollouts"] > 0,
        )

    def rank(self, index: int) -> tuple[float, float]:
        """What the validations are ordered by: the `score` and, between
        validations judged on the same window (at the ends of a run the
        first and last `neighbors + 1` share theirs), the score of the
        validation alone, its own rollouts and skill -- only there do its
        own draws decide."""
        return self.score(index), self.history[index]["selection_score"]

    def best(self) -> int:
        """The best validation so far (the first, on a tie of `rank`)."""
        if not self.history:
            raise ValueError("no validations recorded")
        return max(range(len(self)), key=lambda i: (self.rank(i), -i))
