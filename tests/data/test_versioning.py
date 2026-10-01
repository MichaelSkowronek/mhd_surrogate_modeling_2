import subprocess
from types import SimpleNamespace

import pytest
import yaml

from mhd_surrogate.data.versioning import (
    DataVersionError,
    data_provenance,
    dvc_status,
    ensure_up_to_date,
)

ZARR = "data/processed/re16k_t400.zarr"
STATS = "data/processed/normalization_stats.json"


def _fake_run(stdout="{}", returncode=0, stderr=""):
    def run(cmd, **kwargs):
        assert cmd == ["dvc", "status", "--json"]
        return SimpleNamespace(stdout=stdout, returncode=returncode, stderr=stderr)

    return run


def test_dvc_status_up_to_date_is_empty(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run("{}"))
    assert dvc_status() == {}


def test_dvc_status_returns_changed_stages(monkeypatch):
    out = '{"compute_stats": [{"changed deps": {"scripts/data/compute_stats.py": "modified"}}]}'
    monkeypatch.setattr(subprocess, "run", _fake_run(out))
    assert "compute_stats" in dvc_status()


def test_dvc_status_nonzero_exit_is_an_error(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run("", returncode=255, stderr="boom"))
    with pytest.raises(DataVersionError, match="boom"):
        dvc_status()


def test_dvc_status_unparseable_output_is_an_error(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run("not json"))
    with pytest.raises(DataVersionError, match="unparseable"):
        dvc_status()


def test_dvc_status_missing_binary_is_an_error(monkeypatch):
    def run(cmd, **kwargs):
        raise FileNotFoundError("dvc")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(DataVersionError, match="verify_data_version=false"):
        dvc_status()


def test_ensure_up_to_date_accepts_empty_status():
    ensure_up_to_date({})


def test_ensure_up_to_date_names_the_stale_stages():
    status = {"convert_to_zarr": [{"changed deps": {"data/raw": "modified"}}]}
    with pytest.raises(DataVersionError, match="convert_to_zarr") as exc:
        ensure_up_to_date(status)
    assert "dvc repro" in str(exc.value)


def _write_lock(tmp_path, stats_path=STATS):
    lock = {
        "stages": {
            "convert_to_zarr": {"outs": [{"path": ZARR, "md5": "zarr.dir"}]},
            "compute_stats": {"outs": [{"path": stats_path, "md5": "stats"}]},
        }
    }
    raw = {"outs": [{"path": "raw", "md5": "raw.dir"}]}
    (tmp_path / "dvc.lock").write_text(yaml.safe_dump(lock))
    (tmp_path / "raw.dvc").write_text(yaml.safe_dump(raw))
    return tmp_path / "dvc.lock", tmp_path / "raw.dvc"


def test_data_provenance_reads_the_pinned_hashes(tmp_path):
    lock, raw = _write_lock(tmp_path)
    assert data_provenance(lock, raw) == {
        "raw_md5": "raw.dir",
        "zarr_md5": "zarr.dir",
        "stats_md5": "stats",
    }


def test_data_provenance_missing_output_is_an_error(tmp_path):
    lock, raw = _write_lock(tmp_path, stats_path="somewhere/else.json")
    with pytest.raises(DataVersionError, match="normalization_stats.json"):
        data_provenance(lock, raw)


def test_data_provenance_parses_the_committed_dvc_files():
    # Guards against dvc.yaml/dvc.lock drifting from what this module expects.
    provenance = data_provenance()
    assert set(provenance) == {"raw_md5", "zarr_md5", "stats_md5"}
    assert provenance["raw_md5"].endswith(".dir")
    assert provenance["zarr_md5"].endswith(".dir")
