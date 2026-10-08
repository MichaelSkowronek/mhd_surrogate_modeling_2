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
  subpackages are `data/`, `analysis/`, `models/`, `training/`,
  `evaluation/` and `utils/`; under `scripts/` the subdirectories are
  `data/`, `analysis/`, `viz/`, `training/` and `evaluation/`. Put new code in the one
  matching its stage rather than at the top level of `src/mhd_surrogate/` or
  `scripts/`, and add a new subpackage/subdirectory only when a genuinely new
  stage appears.
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

- The train/val/test split is at the dataset level:
  `configs/data/re16k.yaml` designates 7 datasets for training (used in
  full), 1 for validation (used in full) and 1 (`re16k_t400_5`) as the
  final held-out test set. `re16k_t400_5` must never be read by any script,
  for any purpose, including this project's own analysis suite -- it's
  excluded from `configs/analysis/split.yaml`'s `datasets` list for exactly
  this reason; don't add it back.
- Every decision is made on the validation dataset: hyperparameters, early
  stopping, and comparisons between models (architecture, window length,
  ...), since choosing between models is tuning too. The test dataset is
  read once, at the end: the frozen finalists (baselines included) are
  refit on train + val with the epoch count fixed from their tuned run and
  normalization stats recomputed over train + val (normalization is the first
  step of fitting, not a separate split), and all of them are scored on test
  by one evaluation. Nothing is changed or
  chosen after seeing test scores.
- Candidates are ranked by validation stable steps (how long the forecast
  stays bounded, `evaluation/stability.py`), then skill horizon, ties broken
  by RMSE at lead 10. Mean RMSE over all leads is not a selection metric: it
  rewards forecasts that smooth toward the mean. The physics scores are
  guardrails, not objectives: a candidate replaces the incumbent only if it
  is stable over the whole validation forecast, its energy and enstrophy
  errors are no more than 0.03 worse and its spectrum distances no more than
  0.05 worse (about the spread a different realization scores).
- `scripts/analysis/*.py` scripts read `configs/analysis/split.yaml` (via
  `--config`) and analyze each configured dataset's full recorded length
  (`arr.shape[0]`): datasets are held out whole, so no region within a
  dataset needs bounding. When adding a new check script, follow the same
  pattern: `zarr_store`/`datasets` from the config, `filter_datasets` for
  `--dataset`.
- See the README's "Train / val / test split" section for the full history
  (the earlier per-dataset trailing split it replaced, and why) and
  rationale.

## Data versioning

- The raw `.npy` files are versioned with DVC (`data/raw.dvc`); the
  pipeline is `dvc.yaml` (raw -> zarr -> normalization stats -> the
  canonical model's `train` stage), run with `uv run dvc repro`. Don't
  rebuild the zarr, the stats or the canonical model by running the scripts
  by hand and leave `dvc.lock` out of date: a new pipeline step or
  dependency goes in `dvc.yaml`, and `dvc.lock` is committed with the change
  that produced it (`tests/data/test_dvc_lock.py` fails CI when a stage's
  code or params changed without the re-run). Exploration (sweeps, other models) stays in Hydra/MLflow,
  not `dvc exp`.
- The zarr store stays `cache: false` (a regenerable ~9 GB derivative, not
  stored in DVC's cache or the remote). The raw data, small outputs like the
  stats file, and the canonical model's checkpoint are cached and pushed.
- The test dataset rule applies to the pipeline too: no stage may read
  `re16k_t400_5`, and `compute_stats` reads `train_datasets` only. DVC moving
  the raw directory's bytes (it includes that file) is versioning, not
  reading.
- Training entry points log the data version they were started against
  (`data_provenance`'s `dvc.lock` hashes as MLflow params, plus `dvc.lock`
  and the stats file as artifacts), as `train.py` does. They don't verify it
  against disk: staying in sync is the workflow's job (`dvc repro`), and an
  in-script check was deliberately dropped (see the README).
- Once a sweep's winner is selected, prune the losing runs' checkpoints
  (each run stores ~1-2 GB twice: its MLflow logged model and the Hydra
  output's `model/` directory). Delete the logged model
  (`MlflowClient.delete_logged_model`, then its artifact directory) and the
  `model/` directory, and tag the run `checkpoint_pruned`; keep the run
  record itself (params, metrics, log, small artifacts) -- the README's sweep
  tables and later comparisons rely on it. The same goes for a Ray Tune
  search's trials (`scripts/training/tune.py`), plus each losing trial's
  `training_state/` directory (resumable weights). Trials ASHA stopped have
  no logged model; they're already KILLED and tagged `pruned`, which is
  distinct from `checkpoint_pruned`. The same goes for a trial the search's
  time budget (`search.time_budget_s`) stopped, tagged `time_budget`
  instead: it was cut short, not judged worse.
- Never use `mlflow gc`, not even with `--run-ids`: besides the given runs,
  it permanently purges every deleted logged model in the store (including
  the pruned ones' records). To delete a run outright, delete its logged
  model as above, then `MlflowClient.delete_run` (a soft delete) and remove
  its artifact directory.
- Smoke runs (checking that code works, not producing results) log to a
  dated experiment, `mlflow.experiment_name=smoke-tests-<YYYY-MM-DD>`, not
  the main `mhd-surrogate` one. Delete them as above once the change is
  verified. A deleted experiment's name stays reserved (MLflow only
  soft-deletes it), hence the date.
- Wall-clock timings (`fit_seconds`, `eval_seconds`,
  `forecast_seconds_per_frame`, ...) are MLflow metrics only, never in a DVC
  stage's metrics/outputs (e.g. `Evaluation.scores`, which becomes
  `metrics.json`): they differ run to run, and DVC outputs must reproduce
  byte-identically.
- To re-run a stage from a git worktree (e.g. `.claude/worktrees/<name>`),
  don't regenerate the data there: point it at the main checkout's cache
  (`uv run dvc cache dir --local <main>/.dvc/cache`, which writes the
  untracked `.dvc/config.local`), symlink the zarr store into
  `data/processed/`, `dvc checkout` the stats, and run
  `dvc repro --single-item <stage>` once the main checkout's `dvc status` is
  clean. Sync the worktree's fresh `.venv` with `uv sync --extra gpu` first:
  without it JAX silently falls back to CPU, whose float32 results differ
  from the GPU-produced canonical outputs (a spurious `metrics.json` diff and
  a new ~GB checkpoint in the cache).
- See the README's "Data versioning (DVC)" section for the full rationale.

## Models

- New models implement `src/mhd_surrogate/models/base.py`'s interface,
  register in `models/registry.py` with a `configs/model/` file, and take and
  return raw fields (normalizing internally), so every model is scored in the
  same units.
- Register a model by import path (`"module:Class"` in `MODELS`), not by
  importing it in the registry: the registry imports a model only when it's
  built or loaded, so the DVC `train` stage depends only on the canonical
  model's code and editing another model doesn't mark it stale.
- Neural models build on `models/neural.py`'s `AutoregressiveSurrogate` (a
  new network supplies only `build_network` and `pad_multiple`) and train
  with `training/trainer.py`. An iteratively trained model sets
  `iterative = True` and takes `fit(datasets, hooks)` (`base.FitHooks`:
  validation scores, MLflow training curves, a resumable state directory).
  Early stopping and the hyperparameter search maximize
  `evaluation.metrics.selection_score`, the selection rule as one number,
  never a training loss or another metric.
- Training entry points share `training/run.py`'s `run_training`
  (`train.py` and every Ray Tune trial call it). A change to what a run
  does goes there, not into `train.py`. NN-only code
  (`training/trainer.py`, `training/tuning.py`, the neural models) stays
  out of the DVC `train` stage's deps while a non-neural model is
  canonical: the deps list exactly what `train.py` imports (checked by
  `tests/training/test_dvc_train_stage.py`).
- JAX entry points call `utils/jax_cache.py`'s `enable_compilation_cache`
  before anything is jitted (JAX ignores cache config changes after its
  first compile). The cache is a performance knob, not a DVC param. But a
  cold GPU compile autotunes kernels and doesn't reproduce bits across
  compiles (for the U-Net it changes scores in the 3rd digit), so anything
  that must reproduce -- the DVC stages that produce the canonical model and
  its scores -- runs with `jax.deterministic_ops=true`
  (`utils/jax_determinism.py`, also set before the backend starts);
  exploration leaves it off for speed.

## Parallelism

- Parallel work goes through `src/mhd_surrogate/utils/parallel.py`
  (`run_parallel` for subprocess jobs, `map_tasks` for functions), with
  `--backend sequential|processes|ray` and `--workers` added via
  `add_backend_args`; don't add ad-hoc pools or call `ray.init` directly
  (the layer disables Ray's `uv run` working-directory packaging, which would
  otherwise upload the repo's ~20 GB of data).
- Functions run on a pool or Ray must live in `src/` (importable by workers)
  and be plain Python functions of picklable arguments.
- Anything merged from parallel tasks must merge in a fixed order (see
  `pool_moments`): DVC outputs must be byte-identical on every backend.
- Memory-heavy jobs: order the jobs so heavy ones aren't adjacent, and give
  Ray a per-script memory request (`JOB_MEMORY_GB` in `run_all_checks.py`);
  the suite runs ~2.5 GB jobs and exhausted a 16 GB machine at 12 workers.
- See the README's "Parallel backends" section for measurements and rationale.

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
- Build the image with `GIT_COMMIT=$(git rev-parse HEAD) docker compose
  build`: the image has no git, so runs started in it get their MLflow
  source-commit tag from that build arg (`tracked_run`); without it they're
  untagged. Don't add git or mount `.git` instead: the checkout's current
  commit needn't be the one the image was built from.
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
  `DivergenceError` on NaN/blown-up loss. Their logging is Hydra's
  `job_logging` with the project config
  (`configs/hydra/job_logging/project.yaml`: DEBUG file in the run's output
  directory, INFO console), not `setup_logging`, which is for the argparse
  scripts.
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
- Merge with `gh pr merge --squash --auto`: GitHub merges once the required
  checks pass, so the next piece doesn't wait on CI. Branch it from the
  pending PR's branch, not `main`, so it builds on and is tested against
  that change; after the squash-merge, `git rebase --onto origin/main
  <old-branch>` (a plain rebase would re-apply the commits the squash
  already contains). Branch protection's up-to-date requirement makes the
  squash commit's tree identical to the old branch's, so the rebase changes
  no files and earlier testing stays valid -- unless the pending PR needed
  a fix to pass CI, in which case rebase onto the fix and re-test. Start
  MLflow runs meant to be kept only after the rebase, so their logged
  commit is on `main`. Don't enable `--auto` on the stacked PR until then:
  up to date with `main` and green, it would merge first and carry the
  pending PR's changes into its own squash commit (#40 swallowed #39 this
  way). Push it and open the PR, but enable auto-merge only after the
  rebase.
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
