"""Score one model on one held-out dataset: the protocol plus all metrics."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from mhd_surrogate.evaluation.bootstrap import BlockBootstrap
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
    skill_starts,
    split_context,
    start_seeds,
)
from mhd_surrogate.evaluation.quantities import to_device
from mhd_surrogate.evaluation.selection import start_ensembles
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
class StartSkill:
    """The multi-start skill's ensembles on one dataset: from each of
    `starts` (frames), `size` members per start, drawn from each start's
    `seeds` (`protocol.start_seeds`), and per start (row) and lead time
    (column, index 0 = lead 1) the ensemble's CRPS, spread and
    ensemble-mean RMSE (`ensemble.EnsembleScores`), in units of the
    training std."""

    starts: list[int]
    size: int
    seeds: list[list[int]]
    crps: np.ndarray
    spread: np.ndarray
    ensemble_mean_rmse: np.ndarray

    @property
    def curve(self) -> np.ndarray:
        """The CRPS averaged over the starts, per lead: what the skill
        horizon is measured on."""
        return self.crps.mean(axis=0)

    @property
    def spread_skill(self) -> np.ndarray:
        """Calibration per lead over all the starts: sqrt((M + 1) / M) times
        the spread over the ensemble mean's RMSE, each the root of its mean
        square over the starts (the variance and the squared error pooled,
        as the ratio's expectation is), ~1 for a calibrated ensemble (NaN
        for one member)."""
        if self.size < 2:
            return np.full(self.crps.shape[1], np.nan)
        spread = np.sqrt((self.spread**2).mean(axis=0))
        error = np.sqrt((self.ensemble_mean_rmse**2).mean(axis=0))
        return np.sqrt((self.size + 1) / self.size) * spread / error

    def scores(
        self, threshold: float, report_leads: list[int], bootstrap: BlockBootstrap
    ) -> dict[str, float]:
        """The `multistart_*` scores: the skill horizon of the averaged
        curve (capped at its leads), its bootstrap interval over the starts
        (`_p05`, `_p95`), the averaged CRPS over every lead (the tie-break at
        the cap) and at the `report_leads` that fit, the calibration at them
        for more than one member, and how many starts there were."""
        curve = self.curve
        low, high = bootstrap.interval(self.crps, threshold)
        scores = {
            "multistart_starts": float(len(self.starts)),
            "multistart_crps_skill_horizon": float(skill_horizon(curve, threshold)),
            "multistart_crps_skill_horizon_p05": low,
            "multistart_crps_skill_horizon_p95": high,
            "multistart_crps_mean": float(curve.mean()),
        }
        for lead in report_leads:
            if 1 <= lead <= len(curve):
                scores[f"multistart_crps_lead_{lead}"] = float(curve[lead - 1])
                if self.size > 1:
                    scores[f"multistart_spread_skill_lead_{lead}"] = float(
                        self.spread_skill[lead - 1]
                    )
        return scores

    def as_dict(self) -> dict:
        """Everything, JSON-ready: the MLflow artifact that paired comparisons
        of runs resample (`bootstrap.BlockBootstrap.compare`)."""
        return {
            "starts": list(self.starts),
            "members": self.size,
            "seeds": [list(seeds) for seeds in self.seeds],
            **{
                name: getattr(self, name).tolist()
                for name in ("crps", "spread", "ensemble_mean_rmse")
            },
        }


@dataclass(frozen=True)
class Evaluation:
    """`scores` are scalars (ready for `mlflow.log_metrics`); `rmse` is the
    per-step error curve (the median over the members), index 0 = lead time
    1, `members` each member's own scores (by seed) and `ensemble` the
    ensemble's per-step curves (CRPS, spread), all of the forecast from the
    context's end; `multistart` is the multi-start skill's per-start curves.
    `predict_seconds` is the wall time of one member's forecast alone (no
    data reading or scoring); it's kept out of `scores`, which must be
    reproducible."""

    scores: dict[str, float]
    rmse: np.ndarray
    members: list[dict[str, float]]
    ensemble: EnsembleScores
    multistart: StartSkill
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
    stride: int,
    leads: int,
    n_members: int = 1,
    member_batch: int = 1,
    bootstrap: BlockBootstrap | None = None,
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
    Those are the scores of one forecast, from the context's end.

    The skill the selection rule ranks by comes from many starts, every
    `stride` frames from the context's end, `leads` lead times each
    (`protocol.skill_starts`, `selection.start_ensembles`): the
    `multistart_*` scores (`StartSkill.scores`), the horizon of the CRPS
    averaged over the starts, capped at `leads`, with its `bootstrap`
    interval over the starts (`bootstrap.BlockBootstrap`, its defaults if
    None), and the per-start curves (`Evaluation.multistart`).

    A stochastic model forecasts an ensemble of `n_members` (a deterministic
    model has one member, whatever `n_members`) from the context's end, seeds
    0, 1, ..., and from every start, each start's own seeds
    (`protocol.start_seeds`; the context's end is start 0), sampled
    `member_batch` per
    call where the model can, the first from the context's end alone,
    which is timed (`predict_seconds`). Every score of a trajectory
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
    frames = np.asarray(series[:])
    _, targets = split_context(frames, context_steps)
    starts = skill_starts(len(frames), context_steps, stride, leads)
    seeds = start_seeds(len(starts), ensemble_size(model, n_members))
    stream = MemberBlocks(
        model, frames, context_steps, seeds[0], chunk_t, member_batch, time_first=True
    )
    accumulator = DeviceEnsembleAccumulator(stream.n_steps, scale)
    truth = DeviceSeriesSummary(dx, dy)
    summaries = [DeviceSeriesSummary(dx, dy) for _ in seeds[0]]
    errors = [DeviceRmse(stream.n_steps, scale) for _ in seeds[0]]
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
    ensembles = [
        accumulator.scores()
        for accumulator in start_ensembles(
            model,
            frames,
            starts,
            leads,
            seeds,
            scale,
            DeviceEnsembleAccumulator,
            member_batch,
            chunk_t,
        )
    ]
    multistart = StartSkill(
        starts=starts,
        size=ensemble.size,
        seeds=seeds,
        **{
            name: np.stack([getattr(e, name) for e in ensembles])
            for name in ("crps", "spread", "ensemble_mean_rmse")
        },
    )
    scores.update(multistart.scores(skill_threshold, report_leads, bootstrap or BlockBootstrap()))
    return Evaluation(
        scores=scores,
        rmse=member_median(rmse_curves),
        members=members,
        ensemble=ensemble,
        multistart=multistart,
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
