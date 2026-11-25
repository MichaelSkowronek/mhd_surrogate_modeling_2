"""The selection rule's scores, cheap enough to run every epoch.

Early stopping and the hyperparameter search score a model in training by
`selection_scores`, and choose between a run's validations with
`RunSelection`; `evaluate.evaluate` scores the final model in full, its
multi-start skill by the same `start_ensembles`. In a
module of its own (with `metrics`, `protocol`, `stability` and
`quantities`), so that training -- and the DVC `train` stage's
dependencies -- reach only this part of the evaluation code: a change to
the diagnostics or the ensemble's spread re-runs the `evaluate` stage on the
stored checkpoint instead of retraining the model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from mhd_surrogate.evaluation.metrics import (
    DeviceCrpsAccumulator,
    capped_skill,
    member_median,
    selection_score,
    skill_horizon,
    stable_fraction_bonus,
)
from mhd_surrogate.evaluation.protocol import (
    ForecastModel,
    MemberBlocks,
    StartBlocks,
    ensemble_size,
    skill_starts,
    split_context,
    start_seeds,
)
from mhd_surrogate.evaluation.quantities import DEFAULT_CHUNK_T, to_device
from mhd_surrogate.evaluation.stability import (
    QUANTITIES,
    DeviceQuantities,
    stability_scores,
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
    block_steps: int,
    max_ratio: float,
    stride: int,
    leads: int,
    members: int = 1,
    member_batch: int = 1,
    chunk_t: int = DEFAULT_CHUNK_T,
) -> dict[str, float]:
    """The scores the selection rule ranks by, without the physics
    guardrails: the multi-start skill, the stability scores (`stable_steps`
    and the peak ratios), their combination `selection_score` (of this
    validation alone; `RunSelection` adds the stable fraction) and the
    stability forecast's length, `forecast_steps`. Cheap enough to run
    during training (early stopping, the hyperparameter search); `evaluate`
    scores the final model in full. Every forecast is scored on the device
    as it streams, `chunk_t` lead times at a time, so its frames never
    reach the host.

    The skill comes from many starts (CLAUDE.md, "Data analysis"): every
    `stride` frames from the context's end, as many as fit `leads` lead
    times (`protocol.skill_starts`), each forecast for those leads and
    scored against the truth at the same frames (`start_ensembles`). Its
    CRPS averaged per lead over the starts gives
    `multistart_crps_skill_horizon` (capped at `leads`) and the tie-break
    (`metrics.capped_skill`): `multistart_crps_lead_<tie_break_lead>` below
    the cap, `multistart_crps_mean` at it. The stability scores are the
    forecast of the whole series from the context's end, as before; that
    one trajectory's skill is reported too, under its old names
    (`crps_skill_horizon`, `crps_lead_<tie_break_lead>`).

    A stochastic model is scored as an ensemble of `members` at every start
    (the validations' seeds, `protocol.start_seeds`: each start's own, never
    the scoring's, the same at every validation; a deterministic model has
    one member, whatever `members`), sampled `member_batch` per call
    where the model can (several starts' at once by a model with
    `predict_device_batch`): the skill on the ensembles' CRPS, each
    stability score the median over the members
    (`metrics.member_median`), and `members` and `stable_members` (how many
    are stable over the whole forecast), which `RunSelection` pools into
    the stable fraction."""
    starts = skill_starts(series.shape[0], context_steps, stride, leads)
    if not 1 <= tie_break_lead <= leads:
        raise ValueError(f"tie_break_lead {tie_break_lead} is not one of the {leads} skill leads")
    _, targets = split_context(series, context_steps)
    # Start 0 is the end of the context: the stability forecast's members are its.
    seeds = start_seeds(len(starts), ensemble_size(model, members), monitor=True)
    stream = MemberBlocks(model, series, context_steps, seeds[0], chunk_t, member_batch)
    crps = DeviceCrpsAccumulator(stream.n_steps, scale)
    truth = DeviceQuantities(dx, dy)
    quantities = [DeviceQuantities(dx, dy) for _ in seeds[0]]
    for block in stream:
        target = to_device(targets[block.first : block.first + block.steps])
        truth.add(target)
        crps.add_block(block.first, target, _tracked(block, quantities))
    truth_series = truth.series()
    stability = [
        stability_scores(q.series(), truth_series, block_steps, max_ratio) for q in quantities
    ]
    curve = crps.crps()
    n_steps = len(curve)
    averaged = np.mean(
        [
            start.crps()
            for start in start_ensembles(
                model,
                series,
                starts,
                leads,
                seeds,
                scale,
                DeviceCrpsAccumulator,
                member_batch,
                chunk_t,
            )
        ],
        axis=0,
    )
    skill, tie_break = capped_skill(averaged, skill_threshold, tie_break_lead)
    scores = {key: member_median([s[key] for s in stability]) for key in stability[0]}
    # A member's peak ratio is the larger of its two, NaN if either is
    # (np.max propagates NaN).
    peak = member_median([np.max([s[f"{q}_peak_ratio"] for q in QUANTITIES]) for s in stability])
    return {
        "multistart_crps_skill_horizon": float(skill),
        f"multistart_crps_lead_{tie_break_lead}": float(averaged[tie_break_lead - 1]),
        "multistart_crps_mean": float(averaged.mean()),
        "crps_skill_horizon": float(skill_horizon(curve, skill_threshold)),
        f"crps_lead_{tie_break_lead}": float(curve[tie_break_lead - 1]),
        **scores,
        "selection_score": selection_score(skill, tie_break, scores["stable_steps"], n_steps, peak),
        "forecast_steps": float(n_steps),
        "members": float(len(stability)),
        "stable_members": float(sum(s["stable_steps"] >= n_steps for s in stability)),
    }


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


def _tracked(members: Iterable, quantities: list[DeviceQuantities]) -> Iterator:
    """`members`' frames, each also added to its member's `quantities`."""
    for frames, tracked in zip(members, quantities, strict=True):
        tracked.add(frames)
        yield frames


class RunSelection:
    """The selection rule over one training run's validations: which one's
    network early stopping keeps, and the run's score in the search.

    A network that is stable on the validation forecast at one epoch and not
    at the next is a lucky draw: a refit for the same number of epochs (the
    final train + val refit, a retrain) is likely to land on an unstable
    one. So a validation is judged with the `2 * neighbors + 1` validations
    nearest it (itself included), shifted inward at the ends of the run so
    that every validation is judged with as many (fewer only while the run
    has fewer): its `stable_fraction` is the share of their forecasts stable
    over their whole length -- for a stochastic model the members of all of
    them, pooled (a stable sample is a lucky draw too) -- and its score is
    `metrics.selection_score` with that fraction, which ranks it right after
    the stable steps. With `neighbors` 0 a validation is judged alone, and
    only its `selection_score` is needed; otherwise each validation's scores
    also need `stable_steps`, `forecast_steps`, `members` and
    `stable_members` (`selection_scores` gives them all; a validation
    without the last two counts as one member, stable if its
    `stable_steps` are).

    A validation's window settles once `neighbors` more validations follow
    it (and the run has a full window); until then a later validation can
    still change its score, so the best validation so far is provisional:
    the one early stopping would keep if the run ended now.
    """

    def __init__(self, neighbors: int, history: Iterable[Mapping[str, float]] = ()) -> None:
        if neighbors < 0:
            raise ValueError(f"neighbors must be >= 0, got {neighbors}")
        self.neighbors = neighbors
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

    def members(self, index: int) -> tuple[float, float]:
        """(stable members, members) of validation `index`."""
        scores = self.history[index]
        if "members" in scores:
            return scores["stable_members"], scores["members"]
        return float(scores["stable_steps"] >= scores["forecast_steps"]), 1.0

    def stable_fraction(self, index: int) -> float:
        counts = [self.members(i) for i in self.window(index)]
        return sum(stable for stable, _ in counts) / sum(total for _, total in counts)

    def score(self, index: int) -> float:
        """The validation's `selection_score` with its stable fraction."""
        scores = self.history[index]
        if self.neighbors == 0:
            return scores["selection_score"]
        return scores["selection_score"] + stable_fraction_bonus(
            self.stable_fraction(index),
            scores["stable_steps"],
            int(scores["forecast_steps"]),
            # Forecasts in a full window: the run's members per validation
            # are fixed.
            self.size * int(self.members(index)[1]),
        )

    def best(self) -> int:
        """The best validation so far (the first, on a tie)."""
        if not self.history:
            raise ValueError("no validations recorded")
        return int(np.argmax([self.score(i) for i in range(len(self))]))
