"""Run lifecycle for MLflow tracking: params up front, log file at the end.

Wraps the bookkeeping every training run needs so entry points don't repeat
it: select the experiment, start the run, log the resolved config (as params
and as a `config.json` artifact), and -- whether the run finishes or
crashes -- upload the per-run log file, so a run that died overnight keeps
its post-mortem record next to its metrics. Metrics themselves are logged
directly with `mlflow.log_metrics(..., step=...)` by the training loop.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from mhd_surrogate.training.mlflow_utils import flatten_for_mlflow

log = logging.getLogger(__name__)


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
) -> Iterator[mlflow.ActiveRun]:
    """Start an MLflow run for `config`, and always upload `log_file` on exit.

    `config` is the resolved (nested) config; it is logged flattened as
    params and whole as `config.json`. On an exception the traceback is
    logged (so it lands in the log file), a `DivergenceError` additionally
    sets the `diverged` tag, and the exception propagates; MLflow marks the
    run FAILED.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        log.info("mlflow run: %s (experiment: %s)", run.info.run_id, experiment_name)
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
