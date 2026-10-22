from pathlib import Path

import pytest

import mlflow
from mhd_surrogate.models.baselines import Persistence
from mhd_surrogate.models.registry import load_model
from mhd_surrogate.serving import fetch as fetch_module
from mhd_surrogate.serving.fetch import (
    FetchedModel,
    checkpoint_path,
    fetch,
    load_fetched,
    resolve_version,
)
from mhd_surrogate.training.export import checkpoint_digest
from mhd_surrogate.training.mlflow_model import log_surrogate

FRAME = (2, 5, 4)
GATE = {"gate.rollout_stable_steps": 3000.0, "gate.passed": 1.0}


def test_fetch_downloads_the_aliased_version_and_records_what_it_is(stores, promoted, tmp_path):
    model, promotion = promoted(scores=GATE)
    target = mlflow.MlflowClient(tracking_uri=stores[1])
    copy = target.get_logged_model(promotion.model_id)

    fetched = fetch("models:/surrogate@champion", stores[1], tmp_path / "served")

    assert fetched == FetchedModel(
        model_uri="models:/surrogate@champion",
        name="surrogate",
        version="1",
        model_id=promotion.model_id,
        model_type="mean_field",
        checkpoint_sha256=copy.tags["checkpoint_sha256"],
        source_commit=copy.tags.get("mlflow.source.git.commit"),
        validated_steps=3000,
        frame_shape=FRAME,
    )
    assert load_fetched(tmp_path / "served") == fetched
    served = load_model(checkpoint_path(tmp_path / "served" / "model"))
    assert (served.mean == model.mean).all()
    assert sorted(p.name for p in (tmp_path / "served").iterdir()) == ["model", "serving.json"]


def test_fetch_takes_a_version_number_or_a_bare_logged_model(stores, promoted, tmp_path):
    _, promotion = promoted()

    by_version = fetch("models:/surrogate/1", stores[1], tmp_path / "a")
    by_id = fetch(f"models:/{promotion.model_id}", stores[1], tmp_path / "b")

    assert (by_version.name, by_version.version) == ("surrogate", "1")
    assert (by_id.name, by_id.version, by_id.model_id) == ("mean_field", None, promotion.model_id)
    assert by_id.validated_steps is None
    assert by_version.checkpoint_sha256 == by_id.checkpoint_sha256


def test_a_fetched_model_isnt_downloaded_again_until_the_alias_moves(
    stores, promoted, tmp_path, monkeypatch
):
    promoted(0)
    fetch("models:/surrogate@champion", stores[1], tmp_path / "served")
    download = fetch_module._download
    monkeypatch.setattr(fetch_module, "_download", lambda *args: pytest.fail("downloaded again"))

    again = fetch("models:/surrogate/1", stores[1], tmp_path / "served")
    assert (again.model_uri, again.version) == ("models:/surrogate/1", "1")
    assert load_fetched(tmp_path / "served") == again

    _, second = promoted(1)
    monkeypatch.setattr(fetch_module, "_download", download)
    moved = fetch("models:/surrogate@champion", stores[1], tmp_path / "served")
    assert (moved.version, moved.model_id) == ("2", second.model_id)
    served = checkpoint_path(tmp_path / "served" / "model")
    assert checkpoint_digest(served) == moved.checkpoint_sha256


def test_a_checkpoint_that_isnt_the_registered_one_is_refused(stores, promoted, tmp_path):
    _, promotion = promoted()
    copy = mlflow.MlflowClient(tracking_uri=stores[1]).get_logged_model(promotion.model_id)
    stored = Path(copy.artifact_location) / "artifacts"
    (next(stored.iterdir()) / "model.json").write_text("{}")

    with pytest.raises(ValueError, match="fetched checkpoint has content hash"):
        fetch("models:/surrogate@champion", stores[1], tmp_path / "served")
    assert not (tmp_path / "served" / "serving.json").exists()


def test_a_tampered_fetched_directory_is_refused(stores, promoted, tmp_path):
    promoted()
    fetch("models:/surrogate@champion", stores[1], tmp_path / "served")
    (checkpoint_path(tmp_path / "served" / "model") / "model.json").write_text("{}")

    with pytest.raises(ValueError, match="fetched checkpoint has content hash"):
        fetch("models:/surrogate@champion", stores[1], tmp_path / "served")


def test_a_model_without_a_content_hash_is_refused(stores, tmp_path):
    model = Persistence()
    model.save(tmp_path / "checkpoint")
    mlflow.set_tracking_uri(stores[0])
    mlflow.set_experiment("exp")
    with mlflow.start_run():
        info = log_surrogate(model, tmp_path / "checkpoint", FRAME)

    with pytest.raises(ValueError, match="no checkpoint_sha256 tag"):
        fetch(f"models:/{info.model_id}", stores[0], tmp_path / "served")


def test_resolve_version_wants_a_models_uri(stores):
    with pytest.raises(ValueError, match="not a models:/ URI"):
        resolve_version(mlflow.MlflowClient(tracking_uri=stores[1]), "runs:/abc/model")
