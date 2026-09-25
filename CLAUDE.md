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
- See the README's "Logging" section for the full rationale.

## Licensing (Apache 2.0)

- This project is licensed under Apache 2.0. If you vendor or adapt Apache-licensed third-party code into this repo, retain its original copyright notice and add a note describing what was changed.
