"""Run lifecycle for MLflow tracking: params up front, log file at the end.

Wraps the bookkeeping every training run needs so entry points don't repeat
it: select the experiment, start the run, log the resolved config (as params
and as a `config.json` artifact), and -- whether the run finishes or
crashes -- upload the per-run log file, so a run that died overnight keeps
its post-mortem record next to its metrics. Metrics themselves are logged
directly with `mlflow.log_metrics(..., step=...)` by the training loop.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT

import mlflow
from mhd_surrogate.training.mlflow_utils import flatten_for_mlflow

log = logging.getLogger(__name__)

# The commit the container image was built from (the Dockerfile's
# GIT_COMMIT build arg). The image has no git or .git, so MLflow can't detect
# it, and reading a mounted .git would give the checkout's current commit,
# not the one baked into the image.
IMAGE_COMMIT_ENV = "MHD_GIT_COMMIT"


class DivergenceError(RuntimeError):
    """Raised by a training loop when the loss goes NaN/inf or blows up.

    `tracked_run` tags the run `diverged=true`, so these runs are filterable
    in the MLflow UI (MLflow itself only marks the run FAILED).
    """


@contextmanager
def tracked_run(
    tracking_uri: str,
    experiment_name: str,
    config: dict[str, Any],
    log_file: str | Path | None = None,
    run_name: str | None = None,
    run_id: str | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Start an MLflow run for `config`, and always upload `log_file` on exit.

    `config` is the resolved (nested) config; it is logged flattened as
    params and whole as `config.json`. On an exception the traceback is
    logged (so it lands in the log file), a `DivergenceError` additionally
    sets the `diverged` tag, and the exception propagates; MLflow marks the
    run FAILED. `run_name` is shown in the runs table instead of MLflow's
    random name (the training entry point passes the model's name).

    With `run_id`, that existing run is continued instead (a resumed
    training run, see `resumed_run_id`) and tagged `resumed`; the config
    must be the one it was started with, as MLflow refuses to change a
    logged param's value.

    A new run started in the container image is tagged with the image's
    commit (`IMAGE_COMMIT_ENV`) as `mlflow.source.git.commit`, the tag
    MLflow sets itself from git outside the container. Like MLflow's own
    tag, it's set only when a run is created, not on resume.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    image_commit = os.environ.get(IMAGE_COMMIT_ENV)
    tags = {MLFLOW_GIT_COMMIT: image_commit} if image_commit and run_id is None else None
    with mlflow.start_run(run_id=run_id, run_name=None if run_id else run_name, tags=tags) as run:
        log.info("mlflow run: %s (experiment: %s)", run.info.run_id, experiment_name)
        if run_id is not None:
            mlflow.set_tag("resumed", "true")
        mlflow.log_params(flatten_for_mlflow(config))
        mlflow.log_dict(config, "config.json")
        try:
            yield run
        except DivergenceError:
            mlflow.set_tag("diverged", "true")
            log.exception("run diverged")
            raise
        except Exception:
            log.exception("run failed")
            raise
        finally:
            _upload_log_file(log_file)


def _upload_log_file(log_file: str | Path | None) -> None:
    if log_file is None:
        return
    path = Path(log_file)
    for handler in logging.getLogger().handlers:
        handler.flush()
    if path.is_file():
        mlflow.log_artifact(str(path))
    else:
        log.warning("log file %s not found, not uploading it", path)


RUN_RECORD = "mlflow_run.json"


def save_run_record(state_dir: Path, run_id: str, config: dict[str, Any]) -> None:
    """Record which MLflow run, under which config, owns the training state
    in `state_dir`, so a resume can continue that run."""
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / RUN_RECORD).write_text(json.dumps({"run_id": run_id, "config": config}, indent=2))


def resumed_run_id(state_dir: Path, config: dict[str, Any]) -> str:
    """The MLflow run to continue from the training state in `state_dir`;
    raises unless `config` is the one that run was started with."""
    path = state_dir / RUN_RECORD
    if not path.exists():
        raise FileNotFoundError(f"no resumable training run in {state_dir} ({RUN_RECORD} missing)")
    record = json.loads(path.read_text())
    if record["config"] != json.loads(json.dumps(config)):
        changed = sorted(
            key
            for key in set(record["config"]) | set(config)
            if record["config"].get(key) != json.loads(json.dumps(config.get(key)))
        )
        raise ValueError(
            f"resume with the original run's config: {', '.join(changed)} differ from {path}"
        )
    return record["run_id"]
