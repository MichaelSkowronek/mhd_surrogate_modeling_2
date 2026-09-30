"""End-to-end wiring check for the EDA pipeline: split_data.py -> manifest ->
run_all_checks.py's six check scripts -> comparison.csv.

Unlike the rest of tests/, this runs the real CLI entry points as
subprocesses against the actual local zarr store, so it needs
data/processed/<...>.zarr to exist locally (built by
scripts/data/convert_to_zarr.py) -- it's skipped automatically otherwise,
including in CI, since the ~9GB store isn't checked into git (same pattern
as tests/data/test_data_contract.py).

This deliberately does NOT assert anything about the *values* found (that
stays a human judgement call, per CLAUDE.md's Testing section, and can't
run in CI anyway) -- it asserts the pipeline still *wires together*
correctly: every script runs without error, and the columns/values in
run_all_checks.py's comparison table actually come from where they claim
to. That second part matters: run_all_checks.py's build_comparison_table
once silently read the wrong manifest key for "trainval_steps" (reporting
the train-only length after a schema refactor renamed what that key
meant), and no existing test caught it -- every script's own unit tests
passed, because each was tested in isolation, not the wiring between them.
This test's cross-check against the manifest's own trainval range is aimed
squarely at catching that class of bug again.

Runs against re16k_t400_0 only, for speed; writes manifest/summaries to a
tmp_path (side-effect-free for those), but the check scripts' plots still
land in their default reports/figures/ (gitignored, same as running them
by hand -- not worth the added complexity of threading a --out-dir through
run_all_checks.py's job dispatch just to isolate this).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SPLIT_CONFIG = Path("configs/analysis/split.yaml")
DATASET = "re16k_t400_0"
EXPECTED_COLUMNS = {
    "dataset",
    "div_normalized_mean",
    "enstrophy_mean",
    "n_steps",
    "test_steps",
    "trainval_steps",
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
    manifest_path = tmp_path / "manifest.json"
    subprocess.run(
        [sys.executable, "scripts/data/split_data.py", "--out", str(manifest_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(manifest_path.read_text())
    assert DATASET in manifest["splits"], f"{DATASET} missing from the generated manifest"
    trainval_start, trainval_end = manifest["splits"][DATASET]["trainval"]
    expected_trainval_steps = trainval_end - trainval_start

    summary_dir = tmp_path / "summaries"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/run_all_checks.py",
            "--manifest",
            str(manifest_path),
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
    # The cross-check that would have caught the real trainval_steps bug:
    # the comparison table's value must match the manifest's own trainval
    # range length, not e.g. the train-only region's length.
    assert int(values["trainval_steps"]) == expected_trainval_steps
