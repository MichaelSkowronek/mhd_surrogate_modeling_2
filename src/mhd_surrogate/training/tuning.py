"""Hyperparameter search with Ray Tune: Optuna proposes, ASHA prunes.

`scripts/training/tune.py` is the entry point; this module holds what runs
in the Ray workers (`run_trial`, importable from `src/`) and the pure pieces
around it.

- Every trial is an ordinary training run (`training/run.py`, as
  `train.py` does it) with the sampled hyperparameters applied to the
  config as dotted overrides (`model.training.learning_rate`, ...): its own
  MLflow run, nested under the search's parent run and tagged with the
  search's `sweep` name, with a checkpoint, logged model and full validation
  scores if it finishes.
- After every validation the trial reports its selection scores to Tune.
  The metric is `selection_score`: the selection rule (skill horizon,
  ties broken by the RMSE at lead 10) as one number, the same one early
  stopping maximizes, so the search optimizes what models are chosen by.
- Optuna's TPE sampler proposes the next configuration from the finished
  and running ones; ASHA (asynchronous successive halving) stops trials
  whose score after `grace_period`, `grace_period * reduction_factor`, ...
  validations is below the top 1/`reduction_factor` of the trials that got
  that far. Most of a search's budget goes to the promising configurations
  rather than to training every one of them to the end.
- A pruned trial's process is ended by Ray mid-training, so its MLflow run
  can't close itself: `close_pruned_runs` marks it KILLED, tags it
  `pruned=true` and uploads its log, from the driver.
"""

from __future__ import annotations

import copy
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

METRIC = "selection_score"
MODE = "max"
TRIAL_LOG = "train.log"


def search_space(space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Ray Tune sampling domains from the search config's `space`:
    {dotted.key: {type: loguniform|uniform|choice|randint, ...}}."""
    from ray import tune

    builders = {
        "loguniform": lambda s: tune.loguniform(s["low"], s["high"]),
        "uniform": lambda s: tune.uniform(s["low"], s["high"]),
        "randint": lambda s: tune.randint(s["low"], s["high"]),
        "choice": lambda s: tune.choice(list(s["values"])),
    }
    domains = {}
    for key, spec in space.items():
        if spec.get("type") not in builders:
            raise ValueError(
                f"{key}: unknown type {spec.get('type')!r}, expected {sorted(builders)}"
            )
        domains[key] = builders[spec["type"]](spec)
    return domains


def apply_overrides(config: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of the nested `config` with each dotted key in `params` set.
    Every key must already exist: a typo would otherwise tune nothing."""
    out = copy.deepcopy(dict(config))
    for dotted, value in params.items():
        *parents, leaf = dotted.split(".")
        node = out
        for part in parents:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"{dotted}: no config section {part!r}")
            node = node[part]
        if leaf not in node:
            raise KeyError(f"{dotted}: not a key of the config")
        node[leaf] = value
    return out


def trial_name(model_name: str, params: Mapping[str, Any]) -> str:
    """The run name: the model plus its sampled values, like train.py's
    overrides (`unet model.window=4 ...`); floats to 3 significant digits."""
    shown = [f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()]
    return " ".join([model_name, *shown])


def setup_trial_logging(log_file: Path) -> logging.Handler:
    """In a Ray worker there is no Hydra job logging: write this project's
    DEBUG records to the trial's own log file (uploaded by tracked_run)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    project = logging.getLogger("mhd_surrogate")
    project.setLevel(logging.DEBUG)
    project.addHandler(handler)
    return handler


def run_trial(
    params: dict[str, Any],
    *,
    base_config: dict[str, Any],
    root: str,
    output_dir: str,
    tags: dict[str, str],
) -> None:
    """One Ray Tune trial: a training run with `params` applied to
    `base_config`, reporting every validation to Tune.

    Runs in a Ray worker: works from the repo `root` (paths in the config
    are relative to it), and writes to `<output_dir>/<trial id>/`.
    """
    os.chdir(root)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from omegaconf import OmegaConf
    from ray import tune

    from mhd_surrogate.training.run import run_training

    context = tune.get_context()
    trial_dir = Path(output_dir) / context.get_trial_id()
    log_file = trial_dir / TRIAL_LOG
    setup_trial_logging(log_file)
    cfg = OmegaConf.create(apply_overrides(base_config, params))

    def report(scores: dict[str, float], run_id: str) -> None:
        tune.report({**scores, "mlflow_run_id": run_id, "trial_dir": str(trial_dir)})

    run_training(
        cfg,
        trial_dir,
        log_file,
        trial_name(cfg.model.name, params),
        tags={**tags, "trial_id": context.get_trial_id()},
        on_validation=report,
    )


def close_pruned_runs(results: Sequence[Any], tracking_uri: str) -> list[str]:
    """Mark the MLflow runs of trials ASHA stopped as KILLED, tagged
    `pruned=true`, with their log uploaded. A trial that finished (early
    stopping included) closed its own run; one that errored is left FAILED.
    `results` are Ray Tune `Result`s; returns the pruned run ids."""
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri)
    pruned = []
    for result in results:
        run_id = (result.metrics or {}).get("mlflow_run_id")
        if run_id is None or result.error is not None:
            continue
        run = client.get_run(run_id)
        if run.info.status == "FINISHED":
            continue
        log_file = Path(result.metrics["trial_dir"]) / TRIAL_LOG
        if log_file.is_file():
            client.log_artifact(run_id, str(log_file))
        client.set_tag(run_id, "pruned", "true")
        client.set_terminated(run_id, status="KILLED")
        pruned.append(run_id)
    return pruned


def summary_rows(results: Sequence[Any], params: Sequence[str]) -> list[dict[str, Any]]:
    """One row per trial, best first: its sampled `params`, how many
    validations it reported, its best and last selection score, and whether
    it ended with an error. Results without a reported score (failed before
    the first validation) come last."""
    rows = []
    for result in results:
        metrics = result.metrics or {}
        history = getattr(result, "metrics_dataframe", None)
        scores = list(history[METRIC]) if history is not None and METRIC in history else []
        rows.append(
            {
                **{p: (result.config or {}).get(p) for p in params},
                "validations": metrics.get("training_iteration", 0),
                "best_score": max(scores) if scores else None,
                "last_score": metrics.get(METRIC),
                "error": result.error is not None,
            }
        )
    rows.sort(
        key=lambda r: -float("inf") if r["best_score"] is None else r["best_score"], reverse=True
    )
    return rows
