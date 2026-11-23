"""Score one model on one held-out dataset: the protocol plus all metrics."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from mhd_surrogate.evaluation.diagnostics import (
    DEFAULT_CHUNK_T,
    NPERSEG,
    DeviceSeriesSummary,
    compare_summaries,
)
from mhd_surrogate.evaluation.ensemble import DeviceEnsembleAccumulator, EnsembleScores
from mhd_surrogate.evaluation.metrics import DeviceRmse, member_median, skill_horizon
from mhd_surrogate.evaluation.protocol import (
    EnsembleBlock,
    ForecastModel,
    MemberBlocks,
    ensemble_size,
    split_context,
)
from mhd_surrogate.evaluation.quantities import to_device
from mhd_surrogate.evaluation.stability import QUANTITIES, stability_scores


def candidate_stable_fraction(model: ForecastModel, evaluation: Evaluation) -> float | None:
    """The stable fraction the selection rule's check reads (CLAUDE.md,
    "Data analysis"), given the model's `evaluation` on the validation
    dataset. An iteratively trained model's is its training run's, around
    the epoch it kept (its `fit_info`, from
    `evaluation.selection.RunSelection`); None if its checkpoint predates
    it, which fails the check. A closed-form fit is the same model on
    every refit, so its forecast decides: the share of its members stable
    (1 or 0 for a deterministic model)."""
    if getattr(model, "iterative", False):
        fraction = getattr(model, "fit_info", {}).get("stable_fraction")
        return None if fraction is None else float(fraction)
    n_steps = len(evaluation.rmse)
    return float(np.mean([s["stable_steps"] >= n_steps for s in evaluation.members]))


def train_eval_datasets(requested: list[str], train_datasets: list[str]) -> list[str]:
    """The training datasets to also score a model on, checked to really be
    training datasets: scoring "train" on anything else would mislabel a
    held-out dataset's scores (and could read the test dataset)."""
    unknown = [name for name in requested if name not in train_datasets]
    if unknown:
        raise ValueError(f"evaluation.train_datasets {unknown} are not in data.train_datasets")
    return list(requested)


@dataclass(frozen=True)
class Evaluation:
    """`scores` are scalars (ready for `mlflow.log_metrics`); `rmse` is the
    per-step error curve (the median over the members), index 0 = lead time
    1, `members` each member's own scores (by seed) and `ensemble` the
    ensemble's per-step curves (CRPS, spread). `predict_seconds` is the wall
    time of one member's forecast alone (no data reading or scoring); it's
    kept out of `scores`, which must be reproducible."""

    scores: dict[str, float]
    rmse: np.ndarray
    members: list[dict[str, float]]
    ensemble: EnsembleScores
    predict_seconds: float

    @property
    def seconds_per_frame(self) -> float:
        """Forecast cost per predicted frame, comparable across datasets of
        different lengths (and with the solver's cost per time step)."""
        return self.predict_seconds / len(self.rmse)


def evaluate(
    model: ForecastModel,
    series,
    context_steps: int,
    dx: float,
    dy: float,
    scale: np.ndarray,
    skill_threshold: float,
    report_leads: list[int],
    block_steps: int,
    max_ratio: float,
    n_members: int = 1,
    member_batch: int = 1,
    chunk_t: int = DEFAULT_CHUNK_T,
    nperseg: int = NPERSEG,
) -> Evaluation:
    """Forecast `series` under the protocol and compute every metric.

    RMSE and CRPS are scaled per channel by `scale` (the training std), so
    1.0 means an error as large as the flow's own variability. The skill
    horizon is measured on the CRPS (`crps_skill_horizon`, at most
    `skill_threshold`; named for it, since until 2026-10-27 runs logged an
    RMSE one as `skill_horizon`). `report_leads`
    (1-based) are also reported as scalars, `rmse_lead_<n>` and
    `crps_lead_<n>`; leads past the forecast are skipped. `block_steps` and
    `max_ratio` configure the stability check (`stability.stability_scores`).

    A stochastic model forecasts an ensemble of `n_members` (seeds 0, 1,
    ...; a deterministic model has one member, whatever `n_members`),
    sampled `member_batch` per call where the model can, after seed 0
    alone, which is timed (`predict_seconds`). Every score of a trajectory
    -- the RMSE, the diagnostics, the stability scores -- is the median
    over the members (`metrics.member_median`; each member is one
    trajectory, as `predict` gives it: the ensemble *mean* would be smooth
    and fail the physics). The ensemble adds `crps_mean`, `crps_lead_<n>`,
    `ensemble_size` and, for more than one member, `spread_skill_lead_<n>`
    (`ensemble.EnsembleScores`). For one member the CRPS is the absolute
    error, so deterministic and generative models share that score, and
    the skill horizon.

    The members are scored on the device as they stream, `chunk_t` lead
    times at a time (`protocol.MemberBlocks`), each reduced there to its
    per-step quantities (`diagnostics.DeviceSeriesSummary`,
    `metrics.DeviceRmse`, `ensemble.DeviceEnsembleAccumulator`), which the
    scores are computed from on the host.
    """
    _, targets = split_context(series, context_steps)
    targets = np.asarray(targets)
    seeds = range(ensemble_size(model, n_members))
    stream = MemberBlocks(
        model, series, context_steps, seeds, chunk_t, member_batch, time_first=True
    )
    accumulator = DeviceEnsembleAccumulator(stream.n_steps, scale)
    truth = DeviceSeriesSummary(dx, dy)
    summaries = [DeviceSeriesSummary(dx, dy) for _ in seeds]
    errors = [DeviceRmse(stream.n_steps, scale) for _ in seeds]
    for block in stream:
        target = to_device(targets[block.first : block.first + block.steps])
        truth.add(target)
        accumulator.add_block(block.first, target, _tracked(block, target, summaries, errors))
    true = truth.summary()
    rmse_curves, members = [], []
    for summary, error in zip(summaries, errors):
        rmse = error.curve()
        scores = {"rmse_mean": float(rmse.mean())}
        for lead in report_leads:
            if 1 <= lead <= len(rmse):
                scores[f"rmse_lead_{lead}"] = float(rmse[lead - 1])
        pred = summary.summary()
        scores.update(compare_summaries(pred, true, nperseg))
        scores.update(
            stability_scores(
                {q: pred[q] for q in QUANTITIES},
                {q: true[q] for q in QUANTITIES},
                block_steps,
                max_ratio,
            )
        )
        rmse_curves.append(rmse)
        members.append(scores)
    ensemble = accumulator.scores()
    scores = {key: member_median([s[key] for s in members]) for key in members[0]}
    scores["crps_skill_horizon"] = float(skill_horizon(ensemble.crps, skill_threshold))
    scores["ensemble_size"] = float(ensemble.size)
    scores["crps_mean"] = float(ensemble.crps.mean())
    for lead in report_leads:
        if 1 <= lead <= len(ensemble.crps):
            scores[f"crps_lead_{lead}"] = float(ensemble.crps[lead - 1])
            if ensemble.size > 1:
                scores[f"spread_skill_lead_{lead}"] = float(ensemble.spread_skill[lead - 1])
    return Evaluation(
        scores=scores,
        rmse=member_median(rmse_curves),
        members=members,
        ensemble=ensemble,
        predict_seconds=stream.first_member_seconds,
    )


def _tracked(
    block: EnsembleBlock, target, summaries: list[DeviceSeriesSummary], errors: list[DeviceRmse]
) -> Iterator:
    """The members' frames of `block`, each also reduced into its member's
    summary and its error against `target`, the truth's frames."""
    for frames, summary, error in zip(block, summaries, errors, strict=True):
        summary.add(frames)
        error.add(block.first, frames, target)
        yield frames
