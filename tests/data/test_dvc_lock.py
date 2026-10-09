"""dvc.lock must be up to date with the pipeline's code and params.

dvc.lock records the hash of every stage dependency as of the last
`dvc repro`, and the outputs (the stats, the canonical model, metrics.json)
claim to come from exactly those. A PR that changes a stage's code or
params without re-running it leaves them claiming to come from code that no
longer exists (CLAUDE.md: dvc.lock is committed with the change that
produced it). This test asks DVC itself, dependency by dependency, so it
needs no data: the data deps (`data/...`) and the deps another stage
produces (the `evaluate` stage's checkpoint) are left out -- CI has neither,
and their staleness is `dvc status`'s to report (an upstream stage's code
and params are checked here as its own deps).

Needs the DVC repository (`.dvc/`), so it's skipped in the Docker image,
which copies dvc.yaml and dvc.lock but not `.dvc/`.

The deps one stage takes from another (the checkpoint, metrics.json) are
checked within dvc.lock instead: each stage must record the hash the stage
that produced it recorded. `dvc repro` writes each stage's lock entry as the
stage finishes, so when the `gate` stage fails, `train` and `evaluate` are
already locked with the new checkpoint and scores while `gate` still holds
the old ones; this catches a commit of that state. And the outputs kept in
git (the metrics files) must be the ones the lock records, the gate's saying
it passed: a commit is green only if the gate passed on exactly the
committed model and scores.
"""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def stale_code_and_params_deps() -> dict[str, dict]:
    from dvc.repo import Repo

    stale = {}
    with Repo(str(ROOT)) as repo:
        produced = {str(out.def_path) for stage in repo.index.stages for out in stage.outs}
        for stage in repo.index.stages:
            for dep in stage.deps:
                path = str(dep.def_path)
                if path.startswith("data/") or path in produced:
                    continue
                status = dep.status()
                if status:
                    stale[f"{stage.addressing}: {path}"] = dict(status)
    return stale


@pytest.mark.skipif(not (ROOT / ".dvc").is_dir(), reason="no DVC repository (.dvc/)")
def test_dvc_lock_is_up_to_date_with_every_stages_code_and_params():
    stale = stale_code_and_params_deps()

    assert not stale, (
        "dvc.lock is out of date for these deps; run `uv run dvc repro` and commit "
        f"dvc.lock (and any changed outputs) with the change: {stale}"
    )


def lock() -> dict:
    return yaml.safe_load((ROOT / "dvc.lock").read_text())["stages"]


def produced_dep_mismatches(stages: dict) -> list[str]:
    """Deps whose hash differs from the one their producing stage recorded
    for that output."""
    produced = {
        out["path"]: (name, out["md5"]) for name, stage in stages.items() for out in stage["outs"]
    }
    mismatches = []
    for name, stage in stages.items():
        for dep in stage.get("deps", []):
            if dep["path"] in produced:
                producer, md5 = produced[dep["path"]]
                if dep["md5"] != md5:
                    mismatches.append(f"{name}: {dep['path']} (re-run after {producer})")
    return mismatches


def metrics_files() -> dict[str, str]:
    """Each stage's metrics files (kept in git, `cache: false`), by path."""
    stages = yaml.safe_load((ROOT / "dvc.yaml").read_text())["stages"]
    return {
        path: name
        for name, stage in stages.items()
        for entry in stage.get("metrics", [])
        for path in (entry if isinstance(entry, dict) else [entry])
    }


def test_every_stage_used_what_its_upstream_stages_produced():
    mismatches = produced_dep_mismatches(lock())

    assert not mismatches, (
        "dvc.lock records these stages as run on another version of an upstream output; "
        f"run `uv run dvc repro` (and see why it stopped, e.g. a failed gate): {mismatches}"
    )


def test_produced_dep_mismatches_finds_a_stage_left_behind():
    stages = {
        "train": {"outs": [{"path": "model", "md5": "new"}]},
        "gate": {
            "deps": [{"path": "model", "md5": "old"}, {"path": "code.py", "md5": "x"}],
            "outs": [{"path": "gate.json", "md5": "g"}],
        },
    }

    assert produced_dep_mismatches(stages) == ["gate: model (re-run after train)"]
    stages["gate"]["deps"][0]["md5"] = "new"
    assert produced_dep_mismatches(stages) == []


in_git_checkout = pytest.mark.skipif(
    not (ROOT / ".git").exists(), reason="the metrics files are git's (the image has no models/)"
)


@in_git_checkout
def test_the_committed_metrics_files_are_the_ones_dvc_lock_records():
    recorded = {out["path"]: out["md5"] for stage in lock().values() for out in stage["outs"]}

    for path in metrics_files():
        assert hashlib.md5((ROOT / path).read_bytes()).hexdigest() == recorded[path], (
            f"{path} isn't the file dvc.lock records; commit both from the same `dvc repro`"
        )


@in_git_checkout
def test_the_committed_canonical_model_passed_its_gate():
    (gate,) = [path for path, stage in metrics_files().items() if stage == "gate"]
    verdict = json.loads((ROOT / gate).read_text())

    assert verdict["val"]["passed"] == 1, f"the canonical model failed its gate: {verdict}"
