"""dvc.lock must be up to date with the pipeline's code and params.

dvc.lock records the hash of every stage dependency as of the last
`dvc repro`, and the outputs (the stats, the canonical model, metrics.json)
claim to come from exactly those. A PR that changes a stage's code or
params without re-running it leaves them claiming to come from code that no
longer exists (CLAUDE.md: dvc.lock is committed with the change that
produced it). This test asks DVC itself, dependency by dependency, so it
needs no data: the data deps (`data/...`) are left out -- CI has none, and
their staleness is `dvc status`'s to report.

Needs the DVC repository (`.dvc/`), so it's skipped in the Docker image,
which copies dvc.yaml and dvc.lock but not `.dvc/`.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def stale_code_and_params_deps() -> dict[str, dict]:
    from dvc.repo import Repo

    stale = {}
    with Repo(str(ROOT)) as repo:
        for stage in repo.index.stages:
            for dep in stage.deps:
                if str(dep.def_path).startswith("data/"):
                    continue
                status = dep.status()
                if status:
                    stale[f"{stage.addressing}: {dep.def_path}"] = dict(status)
    return stale


@pytest.mark.skipif(not (ROOT / ".dvc").is_dir(), reason="no DVC repository (.dvc/)")
def test_dvc_lock_is_up_to_date_with_every_stages_code_and_params():
    stale = stale_code_and_params_deps()

    assert not stale, (
        "dvc.lock is out of date for these deps; run `uv run dvc repro` and commit "
        f"dvc.lock (and any changed outputs) with the change: {stale}"
    )
