"""End-to-end wiring check for the EDA pipeline:
configs/analysis/split.yaml -> run_all_checks.py's six check scripts ->
comparison.csv.

Unlike the rest of tests/, this runs the real CLI entry point as a
subprocess against the actual local zarr store, so it needs
data/processed/<...>.zarr to exist locally (built by
scripts/data/convert_to_zarr.py) -- it's skipped automatically otherwise,
including in CI, since the ~9GB store isn't checked into git (same pattern
as tests/data/test_data_contract.py).

This deliberately does NOT assert anything about the *values* found (that
stays a human judgement call, per CLAUDE.md's Testing section, and can't
run in CI anyway) -- it asserts the pipeline still *wires together*
correctly: every script runs without error, and the comparison table's
n_steps actually comes from the dataset it claims to, not just that some
number is present. That kind of cross-check matters here: this pipeline
already had one real bug slip past every script's own (isolated) unit
tests -- run_all_checks.py's build_comparison_table silently reading the
wrong manifest key after a schema refactor -- caught only by hand, not by
any test. The old manifest step is gone now (each check script reads
configs/analysis/split.yaml directly and analyzes a dataset's full
recorded length), but the lesson stands: check that data flows through
correctly, not just that each piece works in isolation.

Runs against re16k_t400_0 only, for speed; writes summaries to a tmp_path
(side-effect-free for those), but the check scripts' plots still land in
their default reports/figures/ (gitignored, same as running them by hand --
not worth the added complexity of threading a --out-dir through
run_all_checks.py's job dispatch just to isolate this).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
import zarr

SPLIT_CONFIG = Path("configs/analysis/split.yaml")
DATASET = "re16k_t400_0"
EXPECTED_COLUMNS = {
    "dataset",
    "div_normalized_mean",
    "enstrophy_mean",
    "n_steps",
    "ux_mean_half_change",
    "ux_n_eff",
    "ux_tau_int",
}


def _store_available() -> bool:
    try:
        config = yaml.safe_load(SPLIT_CONFIG.read_text())
    except FileNotFoundError:
        return False
    return Path(config["zarr_store"]).exists()


pytestmark = pytest.mark.skipif(
    not _store_available(),
    reason="zarr store not present locally (build it with scripts/data/convert_to_zarr.py)",
)


def test_eda_pipeline_wires_together_for_one_dataset(tmp_path):
    config = yaml.safe_load(SPLIT_CONFIG.read_text())
    root = zarr.open_group(store=config["zarr_store"], mode="r")
    expected_n_steps = root[DATASET].shape[0]

    summary_dir = tmp_path / "summaries"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/run_all_checks.py",
            "--config",
            str(SPLIT_CONFIG),
            "--dataset",
            DATASET,
            "--summary-dir",
            str(summary_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"run_all_checks.py failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    csv_path = summary_dir / "comparison.csv"
    assert csv_path.exists()
    rows = csv_path.read_text().strip().splitlines()
    header = rows[0].split(",")
    assert set(header) == EXPECTED_COLUMNS, set(header)

    values = dict(zip(header, rows[1].split(","), strict=True))
    assert values["dataset"] == DATASET
    # The cross-check that would have caught the real trainval_steps bug's
    # class of error: n_steps must match the dataset's actual length, not
    # some other field silently substituted for it.
    assert int(values["n_steps"]) == expected_n_steps
