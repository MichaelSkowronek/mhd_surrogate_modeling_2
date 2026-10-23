"""Promote the gated canonical model to the serving registry.

The DVC pipeline decides which checkpoint is canonical: `train` fits it,
`evaluate` scores it and `gate` checks it still meets the bar that made it
canonical (dvc.yaml). The experiment store it's logged in (the local sqlite
one, `configs/mlflow/local.yaml`) holds every sweep's models besides; what
gets served comes from a registry of its own, the MLflow Model Registry on
the tracking server (`configs/mlflow/server.yaml`). Promotion registers the
gated checkpoint there as a version of a registered model; the deployment
pins that version (README, "Continuous deployment"), and the deployment,
once it has checked the version serving, points the `champion` alias at
it, so the alias always names the version verified in service. (A caller
may still move an alias itself: CI's smoke model has no deployment to.)

Open-source MLflow can't copy a model between stores (`copy_model_version`
stays within one registry), so `promote` copies the logged model holding the
checkpoint -- the source experiment's one tagged with its content hash,
`checkpoint_sha256` (`training/mlflow_utils.py`) -- file by file into a new
logged model on the server. Only the identity fields of its MLmodel (model
id, run id, artifact location) are rewritten to the copy's; the checkpoint,
the code, the pinned requirements and the signature are the source's, byte
for byte, so what's served is the artifact that was trained and gated, not a
rebuild of it. The downloaded checkpoint is hashed again before anything is
uploaded.

The copy is logged by a promotion run, which records the lineage: the source
store, logged model and training run (tags of the copy and of the registered
version), the data version (params), the files that justified the promotion
(dvc.lock, the scores, the gate's verdict) and the validation scores, as
metrics linked to the copy, so its model page shows what it scored.

A checkpoint registered already (same `checkpoint_sha256`) isn't copied
again: its version is the one returned.

Two more tags of the version, set on a version registered already too:
the training data's per-channel statistics (`reference_stats`), the
reference the server measures input drift against (`serving/metrics.py`),
and the checkpoint's hash as `dvc.lock` records it (`checkpoint_dvc_md5`),
which the deployment pins next to the version and its fetch checks.

That the checkpoint is the gated one is the caller's to check
(`scripts/serving/promote.py`: the pipeline up to date by `dvc status`,
dvc.lock committed on `main`, the gate passed by `gate_passed`).
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlflow.entities import LoggedModel, LoggedModelOutput, LoggedModelStatus, Metric, Param
from mlflow.entities.model_registry import ModelVersion
from mlflow.models import Model

import mlflow
from mhd_surrogate.training.export import checkpoint_digest

log = logging.getLogger(__name__)

DIGEST_TAG = "checkpoint_sha256"
# The pyfunc artifact holding the checkpoint (training/mlflow_model.py).
CHECKPOINT_ARTIFACT = "checkpoint"
MLMODEL = "MLmodel"
MODELS_URI = "models:/"
REFERENCE_TAG = "reference_stats"
CHECKPOINT_MD5_TAG = "checkpoint_dvc_md5"


@dataclass(frozen=True)
class Promotion:
    """The registered version of the checkpoint; `copied` is False when it
    was registered already, `alias` the alias moved to it, if any."""

    name: str
    version: str
    model_id: str
    copied: bool
    alias: str | None = None


def gate_passed(path: Path | str) -> bool:
    """Whether a gate verdict (the DVC `gate` stage's `gate.json`) is a pass."""
    return json.loads(Path(path).read_text())["val"]["passed"] == 1


def promote(
    checkpoint: Path | str,
    source_uri: str,
    source_experiment: str,
    target_uri: str,
    name: str,
    experiment: str,
    *,
    alias: str | None = None,
    scores: Mapping[str, float] | None = None,
    params: Mapping[str, str] | None = None,
    records: Sequence[Path | str] = (),
    run_tags: Mapping[str, str] | None = None,
    reference: Mapping[str, Any] | None = None,
    checkpoint_md5: str | None = None,
) -> Promotion:
    """Register `checkpoint` as a version of the registered model `name` on
    the tracking server `target_uri`, copying its logged model there from the
    experiment `source_experiment` of the store `source_uri` unless it's
    registered already (see the module docstring), and point `alias` at it
    if given.

    The copy is logged by a run in `experiment`, tagged `run_tags`, with
    `params` as its params, `records` as its artifacts and `scores` as
    metrics linked to the copy. `reference` (the training data's
    per-channel `channel_names`, `mean` and `std`) becomes the version's
    `reference_stats` tag, `checkpoint_md5` its `checkpoint_dvc_md5`.
    """
    digest = checkpoint_digest(checkpoint)
    target = mlflow.MlflowClient(tracking_uri=target_uri, registry_uri=target_uri)
    version = registered_version(target, name, digest)
    copied = version is None
    if copied:
        source = mlflow.MlflowClient(tracking_uri=source_uri)
        logged = source_model(source, source_experiment, digest)
        lineage = {
            "promoted_from.tracking_uri": source_uri,
            "promoted_from.model_id": logged.model_id,
            "promoted_from.run_id": logged.source_run_id or "",
        }
        experiment_id = _experiment_id(target, experiment)
        run_id = target.create_run(
            experiment_id, run_name=f"promote {logged.name}", tags=dict(run_tags or {})
        ).info.run_id
        try:
            model_id = _copy(target, experiment_id, run_id, logged, source_uri, digest, lineage)
            _log_records(target, run_id, model_id, scores or {}, params or {}, records)
            version = _register(target, name, model_id, run_id, {DIGEST_TAG: digest, **lineage})
        except BaseException:
            target.set_terminated(run_id, "FAILED")
            raise
        target.set_terminated(run_id)
    else:
        log.info("checkpoint already registered as %s version %s", name, version.version)
    number = str(version.version)
    if reference is not None:
        target.set_model_version_tag(name, number, REFERENCE_TAG, json.dumps(dict(reference)))
    if checkpoint_md5 is not None:
        target.set_model_version_tag(name, number, CHECKPOINT_MD5_TAG, checkpoint_md5)
    if alias is not None:
        target.set_registered_model_alias(name, alias, number)
    return Promotion(name, number, registered_model_id(version), copied, alias)


def registered_version(client: mlflow.MlflowClient, name: str, digest: str) -> ModelVersion | None:
    """The newest version of the registered model `name` whose checkpoint has
    content hash `digest`, if any."""
    versions = client.search_model_versions(f"name = '{name}' and tags.{DIGEST_TAG} = '{digest}'")
    return max(versions, key=lambda v: int(v.version)) if versions else None


def registered_model_id(version: ModelVersion) -> str:
    """The logged model `version` was registered from: its `model_id`, which
    a database registry doesn't store, or else its `models:/<id>` source."""
    if version.model_id:
        return version.model_id
    if not version.source.startswith(MODELS_URI):
        raise ValueError(f"{version.name} version {version.version} has no logged model")
    return version.source.removeprefix(MODELS_URI)


def source_model(client: mlflow.MlflowClient, experiment_name: str, digest: str) -> LoggedModel:
    """The newest logged model of the experiment `experiment_name` whose
    checkpoint has content hash `digest`; raises if there is none."""
    experiment = client.get_experiment_by_name(experiment_name)
    models = []
    if experiment is not None:
        models = client.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.{DIGEST_TAG} = '{digest}'",
            order_by=[{"field_name": "creation_time", "ascending": False}],
            max_results=1,
        )
    if not models:
        raise LookupError(
            f"no logged model of checkpoint {digest} in experiment {experiment_name!r}"
        )
    return models[0]


def retarget_mlmodel(path: Path, model_id: str, run_id: str, artifact_location: str) -> None:
    """Rewrite the identity fields of the MLmodel file at `path` to those of
    the logged model `model_id` (logged by run `run_id`, stored at
    `artifact_location`), leaving the rest as it is."""
    model = Model.load(path)
    model.model_id = model.model_uuid = model_id
    model.run_id = run_id
    model.artifact_path = artifact_location
    model.save(path)


def _experiment_id(client: mlflow.MlflowClient, name: str) -> str:
    experiment = client.get_experiment_by_name(name)
    return experiment.experiment_id if experiment else client.create_experiment(name)


def _copy(
    target: mlflow.MlflowClient,
    experiment_id: str,
    run_id: str,
    logged: LoggedModel,
    source_uri: str,
    digest: str,
    lineage: Mapping[str, str],
) -> str:
    """Copy `logged` from the store `source_uri` into a new logged model of
    `target`, logged by `run_id`; returns its id."""
    copy = target.create_logged_model(
        experiment_id,
        name=logged.name,
        source_run_id=run_id,
        tags={**logged.tags, **lineage},
        params=logged.params,
        model_type=logged.model_type,
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            log.info("downloading logged model %s", logged.model_id)
            local = Path(
                mlflow.artifacts.download_artifacts(
                    artifact_uri=logged.artifact_location, dst_path=tmp, tracking_uri=source_uri
                )
            )
            _check_checkpoint(local, digest)
            retarget_mlmodel(local / MLMODEL, copy.model_id, run_id, copy.artifact_location)
            log.info("uploading it as logged model %s", copy.model_id)
            target.log_model_artifacts(copy.model_id, str(local))
    except BaseException:
        target.finalize_logged_model(copy.model_id, LoggedModelStatus.FAILED)
        raise
    target.finalize_logged_model(copy.model_id, LoggedModelStatus.READY)
    target.log_outputs(run_id, [LoggedModelOutput(copy.model_id, 0)])
    return copy.model_id


def _check_checkpoint(local: Path, digest: str) -> None:
    """The downloaded logged model at `local` must hold the checkpoint `digest`."""
    flavor = Model.load(local / MLMODEL).flavors["python_function"]
    path = local / flavor["artifacts"][CHECKPOINT_ARTIFACT]["path"]
    actual = checkpoint_digest(path)
    if actual != digest:
        raise ValueError(f"the downloaded checkpoint has content hash {actual}, expected {digest}")


def _log_records(
    client: mlflow.MlflowClient,
    run_id: str,
    model_id: str,
    scores: Mapping[str, float],
    params: Mapping[str, str],
    records: Sequence[Path | str],
) -> None:
    timestamp = int(time.time() * 1000)
    client.log_batch(
        run_id,
        metrics=[
            Metric(k, float(v), timestamp, 0, model_id=model_id, run_id=run_id)
            for k, v in scores.items()
        ],
        params=[Param(k, str(v)) for k, v in params.items()],
    )
    for record in records:
        client.log_artifact(run_id, str(record))


def _register(
    client: mlflow.MlflowClient, name: str, model_id: str, run_id: str, tags: Mapping[str, str]
) -> ModelVersion:
    if not client.search_registered_models(filter_string=f"name = '{name}'"):
        client.create_registered_model(name)
    # A database registry (not a server's) looks the logged model up in the
    # process's tracking store, whatever store the client was made for.
    with _tracking_uri(client.tracking_uri):
        return client.create_model_version(
            name,
            source=f"{MODELS_URI}{model_id}",
            run_id=run_id,
            tags=dict(tags),
            model_id=model_id,
        )


@contextmanager
def _tracking_uri(uri: str) -> Iterator[None]:
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    try:
        yield
    finally:
        mlflow.set_tracking_uri(previous)
