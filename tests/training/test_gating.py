"""run_gate end to end on a tiny synthetic zarr store: the DVC `gate`
stage's code path."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr
from hydra import compose, initialize_config_dir

import mhd_surrogate.utils.hydra_resolvers  # noqa: F401
from mhd_surrogate.models.baselines import Persistence
from mhd_surrogate.training import gating
from mhd_surrogate.training.gating import run_gate

ROOT = Path(__file__).resolve().parents[2]
SCORED_STEPS = 16
VAL_SCORES = {
    "stable_steps": float(SCORED_STEPS),
    "energy_rel_error": 0.1,
    "enstrophy_rel_error": 0.1,
    "spectrum_x_lsd": 0.1,
    "spectrum_y_lsd": 0.1,
}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A validation dataset, a persistence checkpoint, its metrics.json and
    the grid config, in a repo-like working directory."""
    monkeypatch.chdir(tmp_path)
    store = zarr.open_group(store=str(tmp_path / "data.zarr"), mode="w")
    frame = np.random.default_rng(0).normal(size=(2, 8, 6)).astype(np.float32)
    store.create_array("v", data=np.stack([frame] * (4 + SCORED_STEPS)))
    (tmp_path / "configs" / "analysis").mkdir(parents=True)
    shutil.copy(ROOT / "configs" / "analysis" / "grid.yaml", tmp_path / "configs" / "analysis")
    Persistence().save(tmp_path / "export" / "model")
    write_val_scores(tmp_path, VAL_SCORES)
    return tmp_path


def write_val_scores(workspace, scores):
    (workspace / "export" / "metrics.json").write_text(json.dumps({"val": scores}))


def config(workspace, *overrides):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(
            "config",
            overrides=[
                f"data.zarr_store={workspace / 'data.zarr'}",
                "data.train_datasets=[a]",
                "data.val_dataset=v",
                "data.test_dataset=never_opened",
                "data.context_steps=4",
                "jax.compilation_cache_dir=null",
                f"checkpoint={workspace / 'export' / 'model'}",
                f"export.dir={workspace / 'export'}",
                "evaluation.stability.block_steps=5",
                "evaluation.gate.rollout_steps=30",
                *overrides,
            ],
        )


def verdict(workspace):
    return json.loads((workspace / "export" / "gate.json").read_text())["val"]


def test_a_stable_model_within_its_limits_passes(workspace):
    result = run_gate(config(workspace))

    assert result.passed
    assert result.rollout.n_steps == 30
    assert verdict(workspace) == {
        "passed": 1.0,
        "rollout_steps": 30.0,
        "rollout_stable_steps": 30.0,
        "rollout_energy_peak_ratio": pytest.approx(1.0),
        "rollout_enstrophy_peak_ratio": pytest.approx(1.0),
    }


def test_a_score_over_its_frozen_limit_fails_and_still_writes_the_verdict(workspace):
    write_val_scores(workspace, {**VAL_SCORES, "enstrophy_rel_error": 3.6})

    result = run_gate(config(workspace))

    assert result.failures == ["enstrophy_rel_error 3.6 over its limit 0.758"]
    assert verdict(workspace)["passed"] == 0.0


def test_a_rollout_that_blows_up_fails(workspace, monkeypatch):
    class Growing(Persistence):
        def predict(self, context, n_steps):
            return np.stack([context[-1] * 1.1 ** (k + 1) for k in range(n_steps)])

    monkeypatch.setattr(gating, "load_model", lambda path: Growing())

    result = run_gate(config(workspace))

    # Energy grows 1.21x per step: its 5-step block from step 5 is over 2x.
    assert result.failures == ["rollout stable for 5 of its 30 steps"]
    assert verdict(workspace)["rollout_stable_steps"] == 5.0


def test_the_test_dataset_is_refused(workspace):
    with pytest.raises(ValueError, match="test dataset"):
        run_gate(config(workspace, "data.val_dataset=never_opened"))


def test_the_gate_needs_a_checkpoint_and_an_export_dir(workspace):
    with pytest.raises(ValueError, match="checkpoint"):
        run_gate(config(workspace, "export.dir=null"))
