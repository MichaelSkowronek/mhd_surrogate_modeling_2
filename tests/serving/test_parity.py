"""Serve parity (the DVC `serve_parity` stage's code path) on synthetic
series: a model the service serves faithfully passes, one whose served
forecast isn't its own is caught."""

import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from hydra import compose, initialize_config_dir

import mhd_surrogate.utils.hydra_resolvers  # noqa: F401
from mhd_surrogate.models.baselines import Persistence
from mhd_surrogate.serving.parity import (
    ParityResult,
    relative_difference,
    run_parity,
    serve_parity,
    served_forecast,
)

ROOT = Path(__file__).resolve().parents[2]
FRAME = (2, 4, 3)
CONTEXT_STEPS = 4


def series(n_steps: int) -> np.ndarray:
    return (
        np.random.default_rng(0).normal(size=(CONTEXT_STEPS + n_steps, *FRAME)).astype(np.float32)
    )


class StepCounter:
    """Forecasts its step count, ignoring the context: a model whose state
    isn't its last frames and that has no `predict_blocks`, so the service,
    restarting it at every block (`rollout_blocks`), serves another forecast
    than its own past the first block."""

    name = "step_counter"
    window = 1

    def predict(self, context, n_steps):
        steps = np.arange(1, n_steps + 1, dtype=np.float32)[:, None, None, None]
        return np.broadcast_to(steps, (n_steps, *FRAME)).copy()


def test_relative_difference_is_the_largest_gap_in_truth_stds():
    truth = np.array([-2.0, 2.0])  # std 2
    served = np.array([1.0, 1.0, 1.0])
    offline = np.array([1.0, 0.5, 1.0])

    assert relative_difference(served, offline, truth) == pytest.approx(0.25)
    assert relative_difference(served, served, truth) == 0.0


def test_a_faithfully_served_model_passes():
    result = serve_parity(Persistence(), series(150), CONTEXT_STEPS, tolerance=1e-4)

    assert result == ParityResult(0.0, 1e-4, 150)
    assert result.passed


def test_the_served_forecast_is_streamed_in_the_services_blocks():
    frames = series(150)

    served = served_forecast(StepCounter(), FRAME, frames[:CONTEXT_STEPS], 150)

    # The service's default block is 100 steps: the count restarts there.
    assert served[99, 0, 0, 0] == 100 and served[100, 0, 0, 0] == 1


def test_a_model_whose_served_forecast_isnt_its_own_is_caught():
    result = serve_parity(StepCounter(), series(150), CONTEXT_STEPS, tolerance=1e-4)

    assert not result.passed
    assert result.max_difference > 1
    assert result.verdict() == {
        "passed": 0.0,
        "max_difference": result.max_difference,
        "tolerance": 1e-4,
        "steps": 150.0,
    }


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = zarr.open_group(store=str(tmp_path / "data.zarr"), mode="w")
    store.create_array("v", data=series(30))
    Persistence().save(tmp_path / "export" / "model")
    return tmp_path


def config(workspace, *overrides):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(
            "config",
            overrides=[
                f"data.zarr_store={workspace / 'data.zarr'}",
                "data.train_datasets=[a]",
                "data.val_dataset=v",
                "data.test_dataset=never_opened",
                f"data.context_steps={CONTEXT_STEPS}",
                "jax.compilation_cache_dir=null",
                f"checkpoint={workspace / 'export' / 'model'}",
                f"export.dir={workspace / 'export'}",
                *overrides,
            ],
        )


def test_run_parity_writes_the_verdict(workspace):
    result = run_parity(config(workspace))

    assert result.passed and result.steps == 30
    verdict = json.loads((workspace / "export" / "serve_parity.json").read_text())
    assert verdict == {
        "val": {"passed": 1.0, "max_difference": 0.0, "tolerance": 1e-4, "steps": 30.0}
    }


def test_run_parity_needs_a_checkpoint_and_an_export_dir(workspace):
    with pytest.raises(ValueError, match="checkpoint=<checkpoint directory>"):
        run_parity(config(workspace, "checkpoint=null"))


def test_run_parity_never_reads_the_test_dataset(workspace):
    with pytest.raises(ValueError, match="test dataset"):
        run_parity(config(workspace, "data.val_dataset=never_opened"))
