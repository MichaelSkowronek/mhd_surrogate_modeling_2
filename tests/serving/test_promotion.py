import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from mlflow.entities.model_registry import ModelVersion

import mlflow
from mhd_surrogate.serving.promotion import (
    Promotion,
    gate_passed,
    promote,
    registered_model_id,
    retarget_mlmodel,
)
from mhd_surrogate.training.export import checkpoint_digest

FRAME = (2, 5, 4)
IDENTITY = {"model_id", "model_uuid", "run_id", "artifact_path"}
EXPERIMENT = "promotions"


def run_promote(checkpoint: Path, stores: tuple[str, str], **kwargs) -> Promotion:
    source, target = stores
    kwargs.setdefault("alias", "champion")
    return promote(checkpoint, source, "exp", target, "surrogate", EXPERIMENT, **kwargs)


def target_client(stores) -> mlflow.MlflowClient:
    return mlflow.MlflowClient(tracking_uri=stores[1], registry_uri=stores[1])


def mlmodel(directory: Path) -> dict:
    return yaml.safe_load((directory / "MLmodel").read_text())


def downloaded(uri: str, model_id: str, directory: Path) -> Path:
    """The files of the logged model `model_id` of the store `uri`."""
    location = mlflow.MlflowClient(tracking_uri=uri).get_logged_model(model_id).artifact_location
    return Path(mlflow.artifacts.download_artifacts(location, dst_path=str(directory)))


def test_promote_registers_a_copy_under_the_alias_that_serves_the_model(
    stores, logged_checkpoint, tmp_path
):
    model, checkpoint, _ = logged_checkpoint()

    promotion = run_promote(checkpoint, stores)

    assert promotion == Promotion(
        "surrogate", "1", promotion.model_id, copied=True, alias="champion"
    )
    version = target_client(stores).get_model_version_by_alias("surrogate", "champion")
    assert (str(version.version), registered_model_id(version)) == ("1", promotion.model_id)
    mlflow.set_tracking_uri(stores[1])
    mlflow.set_registry_uri(stores[1])
    served = mlflow.pyfunc.load_model("models:/surrogate@champion")
    prediction = served.predict(np.empty((0, *FRAME), np.float32), params={"n_steps": 2})
    assert (prediction == model.mean).all()


def test_the_copy_is_the_source_byte_for_byte_but_for_its_identity(
    stores, logged_checkpoint, tmp_path
):
    _, checkpoint, source_id = logged_checkpoint()

    promotion = run_promote(checkpoint, stores)

    source = downloaded(stores[0], source_id, tmp_path / "source")
    copy = downloaded(stores[1], promotion.model_id, tmp_path / "copy")
    files = sorted(p.relative_to(source) for p in source.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(copy) for p in copy.rglob("*") if p.is_file())
    for path in files:
        if path.name != "MLmodel":
            assert (source / path).read_bytes() == (copy / path).read_bytes(), path
    assert checkpoint_digest(copy / "artifacts" / checkpoint.name) == checkpoint_digest(checkpoint)
    before, after = mlmodel(source), mlmodel(copy)
    assert {k: v for k, v in before.items() if k not in IDENTITY} == {
        k: v for k, v in after.items() if k not in IDENTITY
    }
    logged = target_client(stores).get_logged_model(promotion.model_id)
    assert after["model_id"] == after["model_uuid"] == promotion.model_id
    assert after["run_id"] == logged.source_run_id
    assert after["artifact_path"] == logged.artifact_location


def test_the_copy_and_its_version_record_where_it_came_from(stores, logged_checkpoint, tmp_path):
    _, checkpoint, source_id = logged_checkpoint()
    source = mlflow.MlflowClient(tracking_uri=stores[0]).get_logged_model(source_id)

    promotion = run_promote(checkpoint, stores)

    client = target_client(stores)
    copy = client.get_logged_model(promotion.model_id)
    lineage = {
        "promoted_from.tracking_uri": stores[0],
        "promoted_from.model_id": source_id,
        "promoted_from.run_id": source.source_run_id,
    }
    assert copy.tags == {**source.tags, **lineage}
    assert (copy.name, copy.params) == (source.name, source.params)
    version = client.get_model_version("surrogate", "1")
    assert version.tags == {"checkpoint_sha256": checkpoint_digest(checkpoint), **lineage}
    assert version.run_id == copy.source_run_id


def test_the_promotion_run_logs_the_scores_params_and_records(stores, logged_checkpoint, tmp_path):
    _, checkpoint, _ = logged_checkpoint()
    record = tmp_path / "gate.json"
    record.write_text(json.dumps({"val": {"passed": 1.0}}))

    promotion = run_promote(
        checkpoint,
        stores,
        scores={"val.skill_horizon": 7.0, "gate.passed": 1.0},
        params={"zarr_md5": "abc.dir"},
        records=[record],
        run_tags={"mlflow.source.git.commit": "1234"},
    )

    client = target_client(stores)
    copy = client.get_logged_model(promotion.model_id)
    assert {m.key: m.value for m in copy.metrics} == {"val.skill_horizon": 7.0, "gate.passed": 1.0}
    run = client.get_run(copy.source_run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params == {"zarr_md5": "abc.dir"}
    assert run.data.tags["mlflow.source.git.commit"] == "1234"
    assert [a.path for a in client.list_artifacts(run.info.run_id)] == ["gate.json"]
    assert [o.model_id for o in run.outputs.model_outputs] == [promotion.model_id]


def test_promoting_a_registered_checkpoint_again_only_moves_the_alias(
    stores, logged_checkpoint, tmp_path
):
    _, first, _ = logged_checkpoint(0)
    _, second, _ = logged_checkpoint(1)
    client = target_client(stores)

    one = run_promote(first, stores)
    again = run_promote(first, stores)
    assert again == Promotion("surrogate", "1", one.model_id, copied=False, alias="champion")
    two = run_promote(second, stores)
    assert (two.version, two.copied) == ("2", True)
    assert str(client.get_model_version_by_alias("surrogate", "champion").version) == "2"
    back = run_promote(first, stores)

    assert back == Promotion("surrogate", "1", one.model_id, copied=False, alias="champion")
    assert str(client.get_model_version_by_alias("surrogate", "champion").version) == "1"
    assert len(client.search_model_versions("name = 'surrogate'")) == 2
    (experiment,) = [e for e in client.search_experiments() if e.name == EXPERIMENT]
    assert len(client.search_logged_models([experiment.experiment_id])) == 2


def test_a_checkpoint_the_source_experiment_never_logged_is_refused(
    stores, logged_checkpoint, tmp_path
):
    _, checkpoint, _ = logged_checkpoint()
    (checkpoint / "model.json").write_text("{}")  # another checkpoint now

    with pytest.raises(LookupError, match="no logged model of checkpoint"):
        run_promote(checkpoint, stores)
    with pytest.raises(LookupError, match="no logged model of checkpoint"):
        promote(checkpoint, stores[0], "missing", stores[1], "surrogate", EXPERIMENT)

    assert target_client(stores).search_experiments(filter_string=f"name = '{EXPERIMENT}'") == []


def test_a_source_copy_that_isnt_the_checkpoint_is_refused_and_nothing_registered(
    stores, logged_checkpoint, tmp_path
):
    _, checkpoint, source_id = logged_checkpoint()
    stored = mlflow.MlflowClient(tracking_uri=stores[0]).get_logged_model(source_id)
    (Path(stored.artifact_location) / "artifacts" / checkpoint.name / "model.json").write_text("{}")

    with pytest.raises(ValueError, match="downloaded checkpoint has content hash"):
        run_promote(checkpoint, stores)

    client = target_client(stores)
    (experiment,) = [e for e in client.search_experiments() if e.name == EXPERIMENT]
    (copy,) = client.search_logged_models([experiment.experiment_id])
    assert copy.status == "FAILED"
    assert client.get_run(copy.source_run_id).info.status == "FAILED"
    assert client.search_model_versions("name = 'surrogate'") == []


def test_gate_passed_reads_the_verdict(tmp_path):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"val": {"passed": 1.0, "rollout_stable_steps": 3000.0}}))
    assert gate_passed(path)
    path.write_text(json.dumps({"val": {"passed": 0.0, "rollout_stable_steps": 1200.0}}))
    assert not gate_passed(path)


def test_retarget_mlmodel_rewrites_only_the_identity_fields(stores, logged_checkpoint, tmp_path):
    _, _, source_id = logged_checkpoint()
    local = downloaded(stores[0], source_id, tmp_path / "m")
    before = mlmodel(local)

    retarget_mlmodel(local / "MLmodel", "m-new", "r-new", "mlflow-artifacts:/2/models/m-new")

    after = mlmodel(local)
    assert {k: after[k] for k in IDENTITY} == {
        "model_id": "m-new",
        "model_uuid": "m-new",
        "run_id": "r-new",
        "artifact_path": "mlflow-artifacts:/2/models/m-new",
    }
    assert {k: v for k, v in after.items() if k not in IDENTITY} == {
        k: v for k, v in before.items() if k not in IDENTITY
    }


def test_registered_model_id_is_the_model_id_or_else_the_models_uri_source():
    def version(**fields):
        return ModelVersion("m", "1", creation_timestamp=0, **fields)

    assert registered_model_id(version(model_id="m-1", source="models:/m-1")) == "m-1"
    assert registered_model_id(version(source="models:/m-2")) == "m-2"
    with pytest.raises(ValueError, match="has no logged model"):
        registered_model_id(version(source="s3://bucket/model"))


def test_the_reference_statistics_are_a_tag_of_the_version_new_or_registered(
    stores, logged_checkpoint
):
    _, checkpoint, _ = logged_checkpoint()
    first = {"channel_names": ["u_x"], "mean": [1.0], "std": [2.0]}
    second = {"channel_names": ["u_x"], "mean": [1.5], "std": [2.0]}

    run_promote(checkpoint, stores, reference=first)
    tags = target_client(stores).get_model_version("surrogate", "1").tags
    assert json.loads(tags["reference_stats"]) == first
    run_promote(checkpoint, stores, reference=second)
    tags = target_client(stores).get_model_version("surrogate", "1").tags
    assert json.loads(tags["reference_stats"]) == second


def test_without_an_alias_no_alias_moves(stores, logged_checkpoint):
    _, checkpoint, _ = logged_checkpoint()

    promotion = run_promote(checkpoint, stores, alias=None)

    assert (promotion.version, promotion.alias) == ("1", None)
    assert target_client(stores).get_registered_model("surrogate").aliases == {}


def test_the_dvc_hash_is_a_tag_of_the_version_new_or_registered(stores, logged_checkpoint):
    _, checkpoint, _ = logged_checkpoint()

    run_promote(checkpoint, stores, checkpoint_md5="abc.dir")
    tags = target_client(stores).get_model_version("surrogate", "1").tags
    assert tags["checkpoint_dvc_md5"] == "abc.dir"
    run_promote(checkpoint, stores, checkpoint_md5="def.dir")
    tags = target_client(stores).get_model_version("surrogate", "1").tags
    assert tags["checkpoint_dvc_md5"] == "def.dir"
