"""Hyperparameter search: Ray Tune trials of train.py, Optuna + ASHA.

Each trial is a full training run (`training/run.py`, as train.py does it)
with hyperparameters sampled from `configs/search/<name>.yaml`'s space by
Optuna's TPE sampler, reporting its validation selection score after every
validation; ASHA stops the trials that fall behind (see
`training/tuning.py`). Everything is decided on the validation dataset; the
test dataset is never read.

MLflow: the search is a parent run (`<model> search`, holding the search
config and, at the end, the best trial's values and score), each trial a
nested child run tagged with the search's `sweep` name, like a Hydra
multirun's. Pruned trials' runs are marked KILLED and tagged `pruned`.

Usage:
    uv run --extra gpu --extra ray scripts/training/tune.py
    uv run --extra gpu --extra ray scripts/training/tune.py search.num_samples=8 \\
        model.training.max_epochs=30
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

# As in train.py: the trials share the GPU with the desktop (and each
# other), so JAX allocates on demand. Ray workers inherit the environment.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import hydra  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import mhd_surrogate.utils.hydra_resolvers  # noqa: E402, F401  (before @hydra.main resolves)
import mlflow  # noqa: E402
from mhd_surrogate.training.tracking import tracked_run  # noqa: E402
from mhd_surrogate.training.tuning import (  # noqa: E402
    METRIC,
    MODE,
    close_pruned_runs,
    run_trial,
    search_space,
    summary_rows,
)
from mhd_surrogate.utils.parallel import init_ray  # noqa: E402

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="tune")
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir).resolve()
    log_file = output_dir / f"{hydra_cfg.job.name}.log"
    search = cfg.search
    resolved = OmegaConf.to_container(cfg, resolve=True)
    resolved.pop("resume")
    # What each trial trains from: the training config without the search.
    base_config = {k: v for k, v in resolved.items() if k != "search"}
    sweep = f"tune-{output_dir.parent.name}-{output_dir.name}"

    with tracked_run(
        cfg.mlflow.tracking_uri,
        cfg.mlflow.experiment_name,
        resolved,
        log_file=log_file,
        run_name=f"{cfg.model.name} search",
    ) as parent:
        mlflow.set_tags({"sweep": sweep, "search": "true"})
        results = _search(cfg, base_config, output_dir, sweep, parent.info.run_id)

        pruned = close_pruned_runs(results, cfg.mlflow.tracking_uri)
        log.info("%d of %d trials pruned by ASHA", len(pruned), len(results))
        rows = summary_rows(results, list(search.space))
        _print_table(rows)
        scored = [r for r in results if (r.metrics or {}).get(METRIC) is not None]
        if not scored:
            log.warning("no trial reported a validation score")
            return
        best = results.get_best_result(METRIC, MODE, scope="all")
        mlflow.log_params({f"best.{k}": v for k, v in best.config.items()})
        mlflow.log_metric(f"best.{METRIC}", best.metrics_dataframe[METRIC].max())
        mlflow.set_tag("best_run_id", best.metrics["mlflow_run_id"])
        log.info("best trial: %s (MLflow run %s)", best.config, best.metrics["mlflow_run_id"])


def _search(cfg: DictConfig, base_config: dict, output_dir: Path, sweep: str, parent_id: str):
    search = cfg.search
    resources = search.resources_per_trial
    ray = init_ray()
    try:
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler
        from ray.tune.search.optuna import OptunaSearch

        trainable = tune.with_resources(
            tune.with_parameters(
                run_trial,
                base_config=base_config,
                root=str(Path.cwd()),
                output_dir=str(output_dir / "trials"),
                tags={"sweep": sweep, "mlflow.parentRunId": parent_id},
            ),
            {
                "cpu": resources.cpu,
                "gpu": resources.gpu,
                "memory": int(resources.memory_gb * 1024**3),
            },
        )
        training = cfg.model.training
        # One Tune iteration per validation. ASHA also stops a trial that
        # reaches max_t, which at the last validation would kill it before
        # its final checkpoint and scoring: one past the last one, it never
        # does (and the rungs are the same).
        max_t = math.ceil(training.max_epochs / training.eval_every) + 1
        tuner = tune.Tuner(
            trainable,
            param_space=search_space(OmegaConf.to_container(search.space)),
            tune_config=tune.TuneConfig(
                metric=METRIC,
                mode=MODE,
                search_alg=OptunaSearch(seed=search.seed),
                scheduler=ASHAScheduler(
                    time_attr="training_iteration",
                    max_t=max_t,
                    grace_period=min(search.grace_period, max_t - 1),
                    reduction_factor=search.reduction_factor,
                ),
                num_samples=search.num_samples,
                max_concurrent_trials=search.max_concurrent_trials,
            ),
            run_config=tune.RunConfig(storage_path=str(output_dir / "ray"), name="search"),
        )
        return tuner.fit()
    finally:
        ray.shutdown()


def _print_table(rows: list[dict]) -> None:
    """The search's result table, best first (report output, not logging)."""
    if not rows:
        return
    columns = list(rows[0])

    def fmt(value) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    widths = {c: max(len(c), *(len(fmt(r[c])) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for row in rows:
        print("  ".join(fmt(row[c]).ljust(widths[c]) for c in columns))


if __name__ == "__main__":
    main()
