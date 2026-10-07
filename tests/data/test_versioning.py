from pathlib import Path

import pytest
import yaml

from mhd_surrogate.data.versioning import data_provenance

ZARR = "data/processed/re16k_t400.zarr"
STATS = "data/processed/normalization_stats.json"


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
    with pytest.raises(ValueError, match="normalization_stats.json"):
        data_provenance(lock, raw)


@pytest.mark.skipif(
    not (Path("dvc.lock").exists() and Path("data/raw.dvc").exists()),
    reason="needs the committed dvc.lock and data/raw.dvc (run from the repo root)",
)
def test_data_provenance_parses_the_committed_dvc_files():
    # Guards against dvc.yaml/dvc.lock drifting from what this module expects.
    provenance = data_provenance()
    assert set(provenance) == {"raw_md5", "zarr_md5", "stats_md5"}
    assert provenance["raw_md5"].endswith(".dir")
    assert provenance["zarr_md5"].endswith(".dir")
