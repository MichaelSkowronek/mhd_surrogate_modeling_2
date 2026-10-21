import os

import pytest
import zarr
from omegaconf import OmegaConf

from mhd_surrogate.training import startup


def jax_config(cache_dir, deterministic):
    return OmegaConf.create(
        {
            "jax": {
                "compilation_cache_dir": cache_dir,
                "compilation_cache_max_gb": 1,
                "deterministic_ops": deterministic,
            }
        }
    )


@pytest.mark.parametrize("cache_dir, deterministic", [(None, False), ("cache", True)])
def test_configure_process_applies_the_jax_config(monkeypatch, cache_dir, deterministic):
    calls = []
    monkeypatch.setattr(startup, "enable_compilation_cache", lambda *a: calls.append(("cache", a)))
    monkeypatch.setattr(startup, "enable_deterministic_ops", lambda: calls.append(("ops", ())))
    monkeypatch.delenv("MLFLOW_DISABLE_AGENT_HINT", raising=False)

    startup.configure_process(jax_config(cache_dir, deterministic))

    expected = [("cache", ("cache", 1)), ("ops", ())] if cache_dir else []
    assert calls == expected
    assert os.environ["MLFLOW_DISABLE_AGENT_HINT"] == "1"


def test_open_data_points_to_compute_stats_when_the_stats_are_missing(tmp_path):
    zarr.open_group(store=str(tmp_path / "data.zarr"), mode="w")
    cfg = OmegaConf.create(
        {
            "data": {"zarr_store": str(tmp_path / "data.zarr")},
            "normalization": {"stats_path": str(tmp_path / "missing.json")},
        }
    )

    with pytest.raises(FileNotFoundError, match="compute_stats.py"):
        startup.open_data(cfg)
