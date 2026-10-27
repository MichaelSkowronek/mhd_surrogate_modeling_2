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
    "u_x_mean_offset": 0.1,
    "u_x_mean_std_ratio": 1.0,
    "u_y_mean_offset": 0.1,
    "u_y_mean_std_ratio": 1.0,
    "stable_fraction": 1.0,
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


def write_val_scores(workspace, scores, members=()):
    """metrics.json with the validation scores and, for an ensemble, each
    member's own."""
    groups = {f"val.member_{seed}": member for seed, member in enumerate(members)}
    (workspace / "export" / "metrics.json").write_text(json.dumps({"val": scores, **groups}))


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
    assert [rollout.n_steps for rollout in result.rollouts] == [30]
    assert verdict(workspace) == {
        "passed": 1.0,
        "rollout_steps": 30.0,
        "rollout_stable_steps": 30.0,
        "rollout_stable_fraction": 1.0,
        "rollout_energy_peak_ratio": pytest.approx(1.0),
        "rollout_enstrophy_peak_ratio": pytest.approx(1.0),
    }


def test_a_score_over_its_frozen_limit_fails_and_still_writes_the_verdict(workspace):
    write_val_scores(workspace, {**VAL_SCORES, "enstrophy_rel_error": 3.6})

    result = run_gate(config(workspace))

    assert result.failures == ["enstrophy_rel_error 3.6 over its limit 0.758"]
    assert verdict(workspace)["passed"] == 0.0


def test_a_stable_fraction_under_its_minimum_fails(workspace):
    write_val_scores(workspace, {**VAL_SCORES, "stable_fraction": 0.6})

    result = run_gate(config(workspace))

    assert result.failures == ["stable_fraction 0.6 under its limit 0.8"]
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


class Members(Persistence):
    """A stochastic model whose member for seed k grows `growth[k]`-fold per
    step, streamed so each block knows its member."""

    stochastic = True

    def __init__(self, *growth):
        self.growth = growth

    def predict(self, context, n_steps, seed=0):
        return np.stack([context[-1] * self.growth[seed] ** (k + 1) for k in range(n_steps)])

    def predict_blocks(self, context, n_steps, block_steps, seed=0):
        whole = self.predict(context, n_steps, seed)
        for start in range(0, n_steps, block_steps):
            yield whole[start : start + block_steps]


def test_a_stochastic_models_rollout_needs_80_percent_of_its_members_stable(workspace, monkeypatch):
    monkeypatch.setattr(gating, "load_model", lambda path: Members(1.0, 1.1, 1.0, 1.0, 1.0))
    write_val_scores(workspace, VAL_SCORES, [VAL_SCORES] * 5)

    five = run_gate(config(workspace, "evaluation.ensemble_size=5"))

    # One of five blows up: 4 of 5 is 80%.
    assert [rollout.stable_steps for rollout in five.rollouts] == [30, 5, 30, 30, 30]
    assert five.passed
    assert verdict(workspace)["rollout_stable_fraction"] == 0.8
    assert verdict(workspace)["rollout_stable_steps"] == 30.0  # the median

    write_val_scores(workspace, VAL_SCORES, [VAL_SCORES] * 3)
    three = run_gate(config(workspace, "evaluation.ensemble_size=3"))

    # Two of three is under 80%, though the median member is stable.
    assert three.failures == [
        "2 of 3 members pass every absolute check, under 80% of them: "
        "seed 1: rollout stable for 5 of its 30 steps"
    ]
    assert verdict(workspace)["passed"] == 0.0


def test_the_members_own_validation_scores_are_checked_one_by_one(workspace, monkeypatch):
    monkeypatch.setattr(gating, "load_model", lambda path: Members(*[1.0] * 5))
    drifting = {**VAL_SCORES, "u_y_mean_offset": 1.5}
    write_val_scores(workspace, VAL_SCORES, [VAL_SCORES] * 5)

    assert run_gate(config(workspace, "evaluation.ensemble_size=5")).passed

    write_val_scores(
        workspace, VAL_SCORES, [drifting, VAL_SCORES, VAL_SCORES, drifting, VAL_SCORES]
    )

    assert run_gate(config(workspace, "evaluation.ensemble_size=5")).failures == [
        "3 of 5 members pass every absolute check, under 80% of them: "
        "seed 0: u_y_mean_offset 1.5 over its limit 1; seed 3: u_y_mean_offset 1.5 over its limit 1"
    ]


def test_the_test_dataset_is_refused(workspace):
    with pytest.raises(ValueError, match="test dataset"):
        run_gate(config(workspace, "data.val_dataset=never_opened"))


def test_the_gate_needs_a_checkpoint_and_an_export_dir(workspace):
    with pytest.raises(ValueError, match="checkpoint"):
        run_gate(config(workspace, "export.dir=null"))
