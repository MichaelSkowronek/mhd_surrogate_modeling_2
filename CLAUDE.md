# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project goal

This is a portfolio project to demonstrate MLOps skills, not a project sized
to only what the data strictly requires. When choosing between a minimal
solution and a more "proper"/industry-standard MLOps tool or practice
(orchestration, experiment tracking, CI/CD, containerization, distributed
compute, etc.), prefer the latter even if it's arguably overkill for the
current data size or team of one — as long as it's implemented correctly and
its purpose is explained, not just bolted on for a résumé keyword. This
doesn't override writing correct, working code, or the testing/logging
conventions below; it shifts which tools are worth reaching for at all.

## Layout

- Organize code by pipeline stage. Under `src/mhd_surrogate/` the
  subpackages are `data/`, `analysis/`, `training/` and `utils/`; under
  `scripts/` the subdirectories are `data/`, `analysis/`, `viz/` and
  `training/`. Put new code in the one matching its stage rather than at the
  top level of `src/mhd_surrogate/` or `scripts/`, and add a new
  subpackage/subdirectory only when a genuinely new stage appears.
- Mirror that structure in `tests/` (`tests/data/`, `tests/analysis/`, ...).
- Hydra configs live in the `configs/` tree (`config.yaml` plus its groups).
  Plain-YAML configs read directly by the data/analysis scripts live in
  `configs/analysis/`; don't put them inside a Hydra group directory, where
  Hydra would treat them as group options.
- Scripts are run from the repo root and use paths relative to it. When
  moving files, update the things coupled to the layout: imports, Hydra's
  `config_path` (relative to the script), `parallel.SCRIPTS_DIR`-relative
  job paths, `tests/conftest.py`, and path references in the docs.
- See the README's "Project layout" section for the full tree.

## Data analysis

- The split is at the *dataset* level now, not a per-dataset time split:
  `configs/data/re16k.yaml` designates 7 datasets for training (used in
  full), 1 for validation (used in full) and 1 (`re16k_t400_5`) as the
  final held-out test set. `re16k_t400_5` must never be read by any script,
  for any purpose, including this project's own analysis suite -- it's
  excluded from `configs/analysis/split.yaml`'s `datasets` list for exactly
  this reason; don't add it back.
- `scripts/analysis/*.py` scripts read `configs/analysis/split.yaml` (via
  `--config`, not a manifest) and analyze each configured dataset's full
  recorded length (`arr.shape[0]`) -- there's no more internal train/val/
  test region to bound reads by within a dataset, since only whole-dataset
  exclusion (`re16k_t400_5`) matters. When adding a new check script,
  follow the same pattern: `zarr_store`/`datasets` from the config,
  `filter_datasets` for `--dataset`, no manifest.
- See the README's "Train / val / test split" section for the full history
  (the earlier per-dataset trailing split it replaced, and why) and
  rationale.

## Docker

- The project is containerized (`Dockerfile`, `docker-compose.yml`). Keep
  it in sync with things that change it:
  - New runtime dependencies or dependency groups/extras in `pyproject.toml`
    need `uv sync` re-run in the image (cache-only effect if `uv.lock` is
    unchanged, otherwise a new locked dependency to install).
  - New top-level directories the app reads/writes at run time (like
    `data/`, `reports/`, `outputs/`) need a bind mount in
    `docker-compose.yml`, and, if git-tracked as empty, a `.gitkeep` so
    Docker doesn't create them as root before the container's non-root user
    can write to them.
  - Layout moves affecting `scripts/`, `configs/`, or `src/` need matching
    `COPY` paths in the `Dockerfile`.
  - New services the app talks to at run time (databases, object stores,
    ...) belong in `docker-compose.yml`, bind-mounted for persistence like
    the existing MLflow stack (`postgres/`, `seaweedfs/` under `mlflow/`),
    not a Docker-managed named volume that a `down -v` or volume prune can
    silently delete.
- See the README's "Docker" section for the full rationale and the
  tracking-stack architecture.

## Code style

- Write Python code following PEP8.

## Testing

- Cover pure computational logic (in `src/mhd_surrogate/` and the
  computational core functions inside `scripts/**/*.py`) with unit tests,
  using synthetic or analytic cases with a known correct answer where
  possible (e.g. a field with known analytic divergence/vorticity, a
  synthetic AR(1) process with known autocorrelation).
- Don't test CLI/argparse/plotting glue, or whether a result is physically
  reasonable for the real data — that stays a human judgement call from the
  printed output and plots.
- pytest runs twice in CI — `tests.yml` against an editable install,
  `docker.yml` against the image's non-editable one — deliberately, not
  redundantly: they exercise different installs, and the two can diverge
  (`parallel.SCRIPTS_DIR`'s old `parents[3]`-from-`__file__` lookup worked
  under an editable install but silently broke non-editably, inside the
  image). Don't remove either on the assumption they're the same check.
- `tests.yml`'s pytest run is coverage-gated at 95% (`pyproject.toml`'s
  `[tool.coverage]`), scoped to `src/mhd_surrogate/` only — not `scripts/`,
  where CLI/argparse/plotting glue (deliberately untested, per above) would
  make a coverage number meaningless. Not in pytest's default `addopts`, so
  a local `pytest tests/foo.py` subset run stays fast and isn't gated on
  partial coverage; only the full-suite CI run is. If new `src/` code drops
  coverage below 95%, add the missing test rather than lowering the gate.
- See the README's "Tests" section for the full rationale and examples.

## Logging

- Use `logging` (via `src/mhd_surrogate/utils/logging_config.py`'s `setup_logging`,
  called once at the start of `main()`, plus `add_log_level_arg` for a
  `--log-level` flag), not `print`, for status/progress/diagnostic messages
  in `scripts/**/*.py`.
- Keep the actual computed results (per-dataset stats, comparison tables) as
  plain `print()` — they're formatted report output meant to be read
  directly or piped, not log records.
- Don't set the root logger's level directly (`logging.basicConfig(level=...)`
  alone) — every third-party dependency inherits from it and leaks internal
  logging as noise. Set the level on this project's own loggers instead
  (`setup_logging` already does this).
- Training metrics (loss, LR, grad norm, throughput) go to MLflow via
  `mlflow.log_metrics(..., step=...)`, never into log files or `print`; the
  log file holds events (checkpoint saved, resumed from, warnings,
  tracebacks). Training entry points use `training/tracking.py`'s
  `tracked_run` (uploads the per-run log file even on a crash) and raise
  `DivergenceError` on NaN/blown-up loss, and they disable Hydra's own job
  logging (`hydra/job_logging: none`) in favor of `setup_logging(...,
  log_file=...)`; otherwise Hydra's root-level INFO handler leaks
  third-party logging.
- See the README's "Logging" section for the full rationale.

## Git workflow

- Trunk-based: `main` only, no long-lived `develop`. Work happens on
  short-lived branches (`feat/...`, `fix/...`, `refactor/...`,
  `experiment/...`) branched from `main`, opened as a pull request back
  into `main`, and squash-merged -- `main`'s history ends up one commit per
  PR. Delete the branch after merge.
- Keep PRs small: one script, one feature, or one coherent change per PR --
  roughly the granularity of a single "commit this" request in a session,
  not a batch of unrelated work.
- CI (`tests.yml`, `docker.yml`, `pre-commit.yml`) runs on PRs into `main`
  and on push to `main`; branch protection on `main` requires a PR and a
  green CI run before merging.
- Tag milestones with an annotated git tag (e.g. `v0.1.0-eda-complete`)
  paired with a GitHub Release summarizing what the milestone covers -- not
  strict semver, since this project has no release artifact in the
  traditional sense yet.

## Maintaining this file

- After finishing a piece of work, consider whether it introduced a
  standing convention, architectural decision, or non-obvious constraint
  (like the ones already in this file) that would trip up future work if
  left undocumented. If so, propose adding it here — don't add it
  silently — and let the user decide.
- Don't propose documenting things the code/git history already makes
  obvious, or one-off details that only matter to the current task.

## Licensing (Apache 2.0)

- This project is licensed under Apache 2.0. If you vendor or adapt Apache-licensed third-party code into this repo, retain its original copyright notice and add a note describing what was changed.
