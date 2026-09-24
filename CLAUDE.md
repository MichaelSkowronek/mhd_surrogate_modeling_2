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
doesn't override writing correct, working code, or the testing convention
above; it shifts which tools are worth reaching for at all.

## Code style

- Write Python code following PEP8.

## Testing

- Cover pure computational logic (in `src/mhd_surrogate/` and the
  computational core functions inside `scripts/*.py`) with unit tests,
  using synthetic or analytic cases with a known correct answer where
  possible (e.g. a field with known analytic divergence/vorticity, a
  synthetic AR(1) process with known autocorrelation).
- Don't test CLI/argparse/plotting glue, or whether a result is physically
  reasonable for the real data — that stays a human judgement call from the
  printed output and plots.
- See the README's "Tests" section for the full rationale and examples.

## Licensing (Apache 2.0)

- This project is licensed under Apache 2.0. If you vendor or adapt Apache-licensed third-party code into this repo, retain its original copyright notice and add a note describing what was changed.
