"""Fetch a model for serving: download it, check it, record what it is.

The forecast server doesn't talk to the registry. A step before it (a
Kubernetes init container, `scripts/serving/fetch_model.py`) resolves the
model URI -- a registered version `models:/<name>/<version>`, an alias
`models:/<name>@<alias>`, or a logged model `models:/<model id>` -- downloads
the logged model into a directory, and writes what the server reports about
it beside it (`serving.json`, a `FetchedModel`). The server only loads that
directory, so it needs no registry access or credentials, and the version it
serves is fixed when it starts, not looked up per request.

The downloaded checkpoint must have the content hash the registered version
(or else the logged model) is tagged with, `checkpoint_sha256`
(`serving/promotion.py`): a corrupted or partial download is never served.
And with `checkpoint_md5` -- the hash `dvc.lock` records for the canonical
checkpoint, which the deployment pins next to the version -- the version
must be tagged with that one (`checkpoint_dvc_md5`): the deployment never
fetches a registry version of another checkpoint than the one its commit
gated.
A directory that already holds the logged model isn't downloaded again,
only verified (a restarted pod's init container finds it there).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from mlflow.entities import LoggedModel
from mlflow.entities.model_registry import ModelVersion
from mlflow.models import Model
from pydantic import BaseModel

import mlflow
from mhd_surrogate.serving.metrics import Reference
from mhd_surrogate.serving.promotion import (
    CHECKPOINT_ARTIFACT,
    CHECKPOINT_MD5_TAG,
    DIGEST_TAG,
    MLMODEL,
    MODELS_URI,
    REFERENCE_TAG,
    registered_model_id,
)
from mhd_surrogate.training.export import checkpoint_digest

log = logging.getLogger(__name__)

SERVING_JSON = "serving.json"
MODEL_DIR = "model"
# The validated horizon: how many steps the gate's rollout stayed stable for
# (`serving/promotion.py` logs the gate's verdict as `gate.*` metrics).
VALIDATED_STEPS_METRIC = "gate.rollout_stable_steps"
COMMIT_TAG = "mlflow.source.git.commit"


class FetchedModel(BaseModel):
    """What was fetched: the registered `name` and `version` (the logged
    model's name and None for a bare logged model), the logged model, its
    checkpoint's content hash, the commit it was trained at, the steps its
    gate validated a rollout stable for (None if it wasn't gated), the
    frame shape (C, Nx, Ny) of its signature, and the training data's
    per-channel statistics, the reference for input drift (None if the
    registered version has none, `serving/metrics.py`)."""

    model_uri: str
    name: str
    version: str | None
    model_id: str
    model_type: str
    checkpoint_sha256: str
    source_commit: str | None
    validated_steps: int | None
    frame_shape: tuple[int, int, int]
    reference: Reference | None = None


def fetch(
    model_uri: str,
    tracking_uri: str,
    destination: Path | str,
    checkpoint_md5: str | None = None,
) -> FetchedModel:
    """Download the model `model_uri` of the tracking server (or store)
    `tracking_uri` into `destination`, which must be the checkpoint whose
    `dvc.lock` hash is `checkpoint_md5` if given (see the module
    docstring)."""
    destination = Path(destination)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    version = resolve_version(client, model_uri)
    if checkpoint_md5 is not None:
        _check_pinned(model_uri, version, checkpoint_md5)
    logged = client.get_logged_model(registered_model_id(version) if version else _id(model_uri))
    digest = (version.tags if version else logged.tags).get(DIGEST_TAG)
    if digest is None:
        raise ValueError(f"{model_uri} has no {DIGEST_TAG} tag to check the download against")

    existing = _fetched(destination)
    if existing is not None and existing.model_id == logged.model_id:
        log.info("%s is in %s already", logged.model_id, destination)
    else:
        _download(logged, tracking_uri, destination)
    _check_checkpoint(destination / MODEL_DIR, digest)
    fetched = FetchedModel(
        model_uri=model_uri,
        name=version.name if version else logged.name,
        version=str(version.version) if version else None,
        model_id=logged.model_id,
        model_type=logged.name,
        checkpoint_sha256=digest,
        source_commit=logged.tags.get(COMMIT_TAG),
        validated_steps=_validated_steps(logged),
        frame_shape=_frame_shape(destination / MODEL_DIR),
        reference=_reference(version),
    )
    (destination / SERVING_JSON).write_text(fetched.model_dump_json(indent=2) + "\n")
    log.info("fetched %s (%s) into %s", model_uri, logged.model_id, destination)
    return fetched


def resolve_version(client: mlflow.MlflowClient, model_uri: str) -> ModelVersion | None:
    """The registered version `model_uri` names (`models:/<name>/<version>`
    or `models:/<name>@<alias>`); None for a logged model, `models:/<id>`."""
    if not model_uri.startswith(MODELS_URI):
        raise ValueError(f"not a models:/ URI: {model_uri}")
    path = model_uri.removeprefix(MODELS_URI)
    if "@" in path:
        name, alias = path.split("@", 1)
        return client.get_model_version_by_alias(name, alias)
    if "/" in path:
        name, number = path.split("/", 1)
        return client.get_model_version(name, number)
    return None


def load_fetched(directory: Path | str) -> FetchedModel:
    """The `FetchedModel` a `fetch` into `directory` recorded."""
    return FetchedModel.model_validate_json((Path(directory) / SERVING_JSON).read_text())


def checkpoint_path(model_dir: Path | str) -> Path:
    """The checkpoint inside a downloaded logged model (its pyfunc
    `checkpoint` artifact, `training/mlflow_model.py`)."""
    flavor = Model.load(Path(model_dir) / MLMODEL).flavors["python_function"]
    return Path(model_dir) / flavor["artifacts"][CHECKPOINT_ARTIFACT]["path"]


def _check_pinned(model_uri: str, version: ModelVersion | None, checkpoint_md5: str) -> None:
    registered = version.tags.get(CHECKPOINT_MD5_TAG) if version is not None else None
    if registered != checkpoint_md5:
        raise ValueError(
            f"the deployment pins checkpoint {checkpoint_md5}, but {model_uri} is "
            f"{registered or 'not tagged with one'}"
        )


def _id(model_uri: str) -> str:
    return model_uri.removeprefix(MODELS_URI)


def _fetched(directory: Path) -> FetchedModel | None:
    try:
        return load_fetched(directory)
    except OSError, ValueError:
        return None


def _download(logged: LoggedModel, tracking_uri: str, destination: Path) -> None:
    """The logged model's files into `destination/model`, replacing what's
    there; `serving.json` goes first, so a fetch cut short leaves none."""
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SERVING_JSON).unlink(missing_ok=True)
    staging = destination / ".download"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(destination / MODEL_DIR, ignore_errors=True)
    log.info("downloading %s", logged.model_id)
    downloaded = mlflow.artifacts.download_artifacts(
        artifact_uri=logged.artifact_location, dst_path=str(staging), tracking_uri=tracking_uri
    )
    Path(downloaded).rename(destination / MODEL_DIR)
    shutil.rmtree(staging, ignore_errors=True)


def _check_checkpoint(model_dir: Path, digest: str) -> None:
    actual = checkpoint_digest(checkpoint_path(model_dir))
    if actual != digest:
        raise ValueError(f"the fetched checkpoint has content hash {actual}, expected {digest}")


def _validated_steps(logged: LoggedModel) -> int | None:
    values = [m.value for m in logged.metrics or [] if m.key == VALIDATED_STEPS_METRIC]
    return int(values[-1]) if values else None


def _reference(version: ModelVersion | None) -> Reference | None:
    if version is None or REFERENCE_TAG not in version.tags:
        return None
    return Reference.model_validate_json(version.tags[REFERENCE_TAG])


def _frame_shape(model_dir: Path) -> tuple[int, int, int]:
    """(C, Nx, Ny) of the logged model's input signature, a stack of frames."""
    (spec,) = Model.load(model_dir / MLMODEL).signature.inputs.inputs
    return tuple(int(n) for n in spec.shape[1:])
