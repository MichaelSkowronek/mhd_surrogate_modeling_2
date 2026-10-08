# mhd_surrogate_modeling_2

Surrogate modeling of a magnetohydrodynamic (MHD) flow. This second project is
independent work on the same data as the first one (mhd_surrogate_modeling).

The data is a 2D slice of a 3D direct numerical simulation (DNS) of an MHD
flow. Working with a 2D slice is itself a research hypothesis: the imposed
magnetic field drives the flow toward a near-uniform state along one of the
three spatial axes, so a 2D slice is treated as a reasonable stand-in for the
full 3D field. There are 9 datasets (`re16k_t400_0.npy` through
`re16k_t400_10.npy`, excluding two known-bad ones): the same DNS run
(Re=16000 for all nine) restarted, with enough small randomness in each
restart that the nine are distinct, not identical, realizations of the
same regime. Since it's one regime, not different ones, the split is at
the dataset level (`configs/data/re16k.yaml`): 7 datasets for training
(used in full), 1 held out in full for validation, 1 (`re16k_t400_5`) held
out in full as the final test set -- see "Train / val / test split" below.
The analysis suite (`configs/analysis/split.yaml`) runs across the 8
non-test datasets; `re16k_t400_5` is excluded from it entirely and must
never be read by any script, analysis included.

## Status

Exploratory data analysis is done (tagged
[`v0.1.0-eda-complete`](https://github.com/MichaelSkowronek/mhd_surrogate_modeling_2/releases/tag/v0.1.0-eda-complete)),
and the dataset-level train/val/test split is implemented and verified
against real data. The forecast-evaluation protocol and metrics are in place
(see "Forecast evaluation"), and the two baselines every model has to beat --
persistence and the mean field -- are fitted, checkpointed and scored on the
validation dataset through the training entry point (see "Baselines"). The
first model with dynamics, DMD, beats both (see "DMD"), and time-delay
(Hankel) DMD with 4 frames of context beats DMD (see "Hankel DMD") -- it is
the canonical model. The first neural network, a U-Net, is built and
trained, with early stopping on validation, resumable training and a Ray
Tune hyperparameter search (Optuna + ASHA; see "U-Net"). Its first search
beats Hankel DMD on the short horizon (skill horizon 15 vs 7) but its long
rollouts blow up, failing the physics guardrails, so Hankel DMD stays
canonical. Selection now ranks by how long a forecast stays bounded first
(see "How candidates are ranked"), and with noise on its training inputs
the U-Net is stable over 3000-step rollouts and passes every guardrail
(skill horizon 10-12 against Hankel DMD's 7; see "Input noise"), making it
the candidate to replace Hankel DMD as the canonical model.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
uv run pre-commit install
```

**GPU.** A plain `uv sync` installs the CPU build of JAX. For training on an
NVIDIA GPU add the `gpu` extra (CUDA 13 wheels, a few GB; needs a recent
driver, no separate CUDA toolkit):

```bash
uv sync --extra gpu          # add --extra ray if you use the Ray backend:
                             # `uv sync` is exact and removes unlisted extras
uv run python -c "import jax; print(jax.devices())"   # [CudaDevice(id=0)]
```

The default Docker image is CPU-only; see "GPU image" under Docker below.
`train.py` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` (and `app-gpu` sets it in
the container): JAX's default of preallocating 75% of GPU memory fails under
WSL with the desktop on the same GPU, and retries with a burst of
out-of-memory errors.

Pre-commit hooks ([pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks):
whitespace/EOF/YAML-TOML-JSON/large-file/merge-conflict/case-conflict checks,
plus [ruff](https://docs.astral.sh/ruff/) lint + format) run automatically on
`git commit`. Run them on demand with:

```bash
uv run pre-commit run --all-files
```

## Project layout

```
src/mhd_surrogate/   importable package, split by pipeline stage
  data/              dataset, grid, normalization, versioning, conversion
  analysis/          fields, summary, spectral
  training/          mlflow_utils, tracking, export, mlflow_model, run, trainer, tuning
  evaluation/        protocol, metrics, diagnostics, evaluate
  models/            base (interface), baselines, dmd, hankel_dmd, neural, unet, registry
  utils/             logging_config, parallel, jax_cache, hydra_resolvers
scripts/             CLI entry points, same split (plus viz/)
  data/              explore_data, convert_to_zarr, compute_stats
  analysis/          check_*.py, run_all_checks, benchmark_backends
  viz/               make_video, make_all_videos, make_forecast_video
  training/          train, tune
  evaluation/        check_rollout_stability
tests/               mirrors src/ and scripts/ (data/, analysis/, training/, evaluation/, models/, utils/, viz/)
dvc.yaml, dvc.lock  data pipeline (raw -> zarr -> stats) and its pinned hashes
data/raw.dvc         DVC pointer to the raw .npy files (.dvc/ holds the remote config)
configs/             Hydra tree: config.yaml (tune.yaml for the search) + data/, dataset/, normalization/, model/, evaluation/, mlflow/, search/ groups
  analysis/          plain-YAML configs (split.yaml, grid.yaml)
```

New code goes under the subpackage/subdirectory matching its pipeline stage,
with its tests in the mirrored `tests/` subdirectory. `configs/analysis/` is
the home for the plain-YAML (non-Hydra) configs the data/analysis scripts
read directly — including `split.yaml`, which every `scripts/analysis/*.py`
script consumes — while everything Hydra composes for training lives in the
rest of `configs/`. Scripts are run from the repo root (`uv run
scripts/analysis/check_split.py`), since data and config paths are relative
to it.

## Tests

```bash
uv run pytest
```

Unit tests live in `tests/`, covering the pure computational logic: the
`src/mhd_surrogate/` modules (`grid`, `fields`, `dataset`, `summary`,
`spectral`, `dmd`, `hankel_dmd`, `pod`, `spod`, `parallel`, `mlflow_utils`,
`logging_config`) plus the computational core functions inside the
`check_*.py`/`make_video.py` scripts (e.g. `per_timestep_stats`,
`field_acf`, `spectrum_sum`, `divergence_stats`, `vorticity_stats`,
`compute_field`) — the `scripts/**/*.py` files aren't part of the installed
package, so `tests/conftest.py` adds each `scripts/` subdirectory to
`sys.path` to import them directly.
Several of these tests convert ad-hoc checks done during development into
permanent regression tests: a synthetic AR(1) process with a known
autocorrelation `phi^lag` (autocorrelation), a synthetic sine wave with
known variance and peak wavenumber (spectrum), and analytic velocity fields
with known divergence/vorticity (e.g. solid-body rotation `u_x=-y, u_y=x`
has constant vorticity 2 everywhere, since both fields are linear so finite
differences are exact).

What's deliberately **not** tested here: the CLI/argparse/plotting glue in
each script, and whether a result is *physically* reasonable for the real
DNS data (e.g. "is a ~36% divergence residual acceptable for a quasi-2D
slice?") — that's a human judgement call made by reading the printed stats
and looking at the plots, not something to assert on in a test.

`src/mhd_surrogate/`'s coverage is gated at 95% in CI (`pyproject.toml`'s
`[tool.coverage]`; `uv run pytest --cov --cov-report=term-missing` to check
locally) — deliberately scoped away from `scripts/`, where the untested
CLI/plotting glue above would make a coverage number meaningless (33-50%
per file there, by design, not a problem to fix). Not wired into pytest's
default `addopts`, so a local subset run (`pytest tests/foo.py`) stays fast
and isn't gated on partial coverage; only the full-suite CI run is.

### Data contract tests

`tests/data/test_data_contract.py` is different from the rest of `tests/`: it
runs against the real zarr store (the datasets and store path listed in
`configs/analysis/split.yaml`) rather than synthetic data, checking objective,
storage-agnostic structural invariants — every configured dataset exists,
shape is `(T, 2, Nx, Ny)`, dtype is `float32`, all values are finite, values
stay within a broad sanity bound (`MAX_ABS_VALUE = 100`, meant to catch
corrupted data, not enforce a tight physical range), and spatial shape is
consistent across datasets. It does **not** check statistical
representativeness or physical plausibility — that stays with
`check_split.py` and the other `check_*.py` scripts.

It reads the zarr store path from config rather than a hardcoded local one,
and reads in chunks rather than loading full arrays, so it keeps working
unchanged if the store moves from local disk to object storage (S3/GCS)
later — zarr supports both through the same API. Since the ~9GB store isn't
checked into git, these tests skip themselves automatically when it isn't
present locally, including in CI.

`tests/analysis/test_pipeline_integration.py` is the same idea, one level
up: instead of checking the raw data's structural contract, it runs the
real CLI entry point as a subprocess -- `scripts/analysis/run_all_checks.py`'s
six check scripts against one dataset -- and checks the *pipeline* still
wires together, not that any result is correct or physically reasonable
(still a human judgement call, unchanged). Concretely: every script runs
without error, and the comparison table's `n_steps` column is cross-checked
against the dataset's actual length read straight from the zarr store,
since each script already has thorough unit tests in isolation but nothing
previously checked the wiring between them -- which is exactly how a real
bug (`run_all_checks.py` silently reading the wrong manifest key after an
earlier schema refactor, reporting the wrong step count) slipped through
unnoticed until caught by hand. That was against the now-retired
per-dataset split manifest (see "Train / val / test split"'s "Earlier: a
per-dataset time split" section); the test still guards against the same
class of bug in the current, simpler pipeline. Same skip-if-store-absent
behavior as the data contract tests above.

## Logging

Every `scripts/**/*.py` CLI script accepts `--log-level` (`DEBUG`/`INFO`/
`WARNING`/`ERROR`, default `INFO`). `src/mhd_surrogate/utils/logging_config.py`'s
`setup_logging` is called once at the start of each script's `main()`.

Status/progress/diagnostic messages (warnings like "skipping this dataset,
wrong shape", orchestration progress like `run_all_checks.py`'s per-job
`[ok]`/`[FAILED]` lines) go through `logging`. The actual computed results —
the per-dataset stat lines, comparison tables, "plot: ..."/"summary: ..."
path confirmations — stay as plain `print()`, since they're formatted report
output meant to be read directly or piped/redirected, not log records.

Two things worth knowing about `setup_logging`:

- **Root logger stays quiet.** `logging.basicConfig` configures the root
  logger, which every third-party dependency (matplotlib, zarr, mlflow, ...)
  inherits from — setting it straight to `DEBUG`/`INFO` leaked their internal
  logging as noise (e.g. matplotlib logging an INFO line per ffmpeg frame
  written). So the root stays at `WARNING` or the requested level, whichever
  is quieter, and only this project's own loggers (`__main__` for a script
  run directly, `mhd_surrogate` for library code like `parallel.py`) get the
  requested level. Requesting `ERROR` still quiets third-party `WARNING`s
  too, since that's stricter than the `WARNING` floor.
- **stdout is reconfigured to line-buffer.** `print()` (stdout) and
  `logging` (stderr by default) interleave correctly in a terminal, but
  stdout is fully block-buffered rather than line-buffered once
  redirected/piped, so a script's printed report could appear to arrive
  "late" relative to its own logged status lines under redirection without
  this.

## Data

Raw `.npy` files are not stored in git; they're versioned with
[DVC](#data-versioning-dvc) (`data/raw.dvc`). Fetch them with `uv run dvc
pull` (see below) before running any scripts, which puts them in `data/raw/`:

```
data/raw/re16k_t400_0.npy
```

Each file is a single time series of shape `(T, 2, H, W)`, channel-first
float32:

- axis 0: time steps (`re16k_t400_0.npy` has 1248)
- axis 1: velocity components, `u_x` then `u_y`
- axes 2, 3: the 2D spatial grid (`re16k_t400_0.npy` is 1151 x 127)

`re16k_t400_7.npy` and `re16k_t400_8.npy` are excluded from the dataset
(known data quality issues) — only indices 0-6, 9, and 10 exist.

### Origin: from the 3D DNS to this dataset

The data comes from a 3D direct numerical simulation (DNS) of an MHD flow on
a grid of 2301 x 121 x 481 points (x, y, z), uniform in x and non-uniform in
y and z. The 2D dataset used here is derived from it as follows:

| axis | DNS points | DNS spacing | processing | this dataset |
|------|-----------:|-------------|------------|-------------:|
| x    | 2301 | uniform | subsampled (1151 = every second point of 2301, i.e. a stride of 2) | 1151 |
| y    | 121  | non-uniform | linear (k = 1) spline interpolation onto a uniform grid | 127 |
| z    | 481  | non-uniform | central slice taken, axis removed | (single plane) |

The domain size of the DNS is `Lx = 25`, `Ly = 2` and `Lz = 7` (an earlier
value of `Lx = 12*pi` ~37.70, from "ratiox = 12.0 in pi units", was
corrected -- that reading of the DNS setup was inconsistent).
Only `Lx` and `Ly` enter the analyses (see the Grid section), since z is not
part of the dataset.

- **z (slice):** the flow varies little along z, the axis along which the
  magnetic field keeps it close to uniform, except in the thin Hartmann
  layers at the z boundaries. Taking the central slice avoids those layers
  and is the basis of the 2D (quasi-2D) hypothesis of this project. Hartmann
  layers, and any variation along z, are therefore not represented.
- **y (interpolation):** the uniform grid has slightly more points (127) than
  the DNS (121), so the interpolation adds no information. Where the DNS
  spacing is finer than `dy` (presumably near the walls) information is lost,
  and y-derivatives are piecewise constant.
- **x (subsampling):** the stride of 2 halves the x resolution and the
  Nyquist wavenumber. Whether a low-pass filter was applied first is not
  known; without one, energy above the new Nyquist wavenumber is aliased into
  the resolved range.
- **t (warm-up):** the DNS was run for a warm-up (spin-up) phase of 400 time
  steps before the saved time series begins; `t400` in the filenames refers
  to this. The recorded snapshots (1248 for `re16k_t400_0.npy`) therefore
  start after the warm-up, not at initialization. Whether "time steps" here
  is the same unit as the snapshot spacing used elsewhere in this README
  (autocorrelation lags, `buffer_steps`) is not known — it may be the DNS's
  internal integration steps, which are not necessarily saved 1:1 with
  snapshots. This is consistent with the "Stationarity check" section below,
  which directly compares the first and second half of train+val and finds
  no meaningful shift in mean or dominant period across all 9 datasets --
  no early-time transient is visible, though a slow, small drift in
  variance is (see that section for the full picture).

### Grid

The domain lengths are in `configs/analysis/grid.yaml` and are used by every
derivative-based or wavenumber-based analysis: `Lx = 25` along axis 2 and
`Ly = 2` along axis 3, with uniform spacing `dx = 25/1150` (~0.02174) and
`dy = 2/126` (~0.01587).

The DNS grid is uniform in x but **non-uniform in y**. The values in the data
were mapped onto the uniform y grid with a linear (k = 1) spline
interpolation. Consequences to keep in mind: y-derivatives (divergence,
vorticity) are piecewise-constant approximations, and small scales in y,
especially near the walls where the DNS is presumably refined, are smoothed
or lost by the interpolation.

## Exploring the data

```bash
uv run scripts/data/explore_data.py
```

Prints shape, dtype, value range, and NaN/Inf counts for each `.npy` file in
`data/raw/`, and saves a preview plot per file to `reports/figures/`.

## Converting to zarr

```bash
uv run scripts/data/convert_to_zarr.py
```

(`--backend`/`--workers` choose how the files are converted in parallel; see
"Parallel backends".) Writes all simulations in `data/raw/` into a single zarr store at
`data/processed/re16k_t400.zarr` (also gitignored). Since each simulation has
a different number of time steps on the same spatial grid, they are stored as
separate arrays within one group rather than stacked into one array:

```python
import zarr

root = zarr.open_group("data/processed/re16k_t400.zarr", mode="r")
root["re16k_t400_0"]  # shape (1248, 2, 1151, 127)
```

## Data versioning (DVC)

The raw `.npy` files (~10 GB) are versioned with [DVC](https://dvc.org): git
holds a small pointer (`data/raw.dvc`, a content hash of the directory) and
the bytes live in a DVC remote, so any commit identifies exactly which data it
was run against, and a fresh clone can fetch it. Only the raw files are
tracked as data; everything derived from them is a pipeline output.

**Remote.** The SeaweedFS S3 gateway already in the compose stack (see
"Tracking stack") serves a second bucket, `dvc`, next to MLflow's `mlflow`
one; `s3-init` creates both and the gateway is published on
`127.0.0.1:8333`. Because it speaks the S3 API, pointing the remote at real
S3 later is a one-line change. Its files live in the bind-mounted
`mlflow/seaweedfs/`, i.e. on the same disk as the working copy, so for now it
guards against accidental deletion and gives the versioning workflow, not
against disk loss.

```bash
docker compose up -d seaweedfs s3-init        # remote up, buckets created
# credentials stay out of git (.dvc/config.local is gitignored); the defaults
# match docker-compose.yml / .env.example:
uv run dvc remote modify --local seaweedfs access_key_id mlflow
uv run dvc remote modify --local seaweedfs secret_access_key mlflow-secret
uv run dvc pull                               # raw data -> data/raw/
uv run dvc repro                              # raw -> zarr -> normalization stats
uv run dvc push                               # after changing tracked data
```

**Pipeline (`dvc.yaml`).** `convert_to_zarr` turns `data/raw` into the zarr
store, and `compute_stats` turns the zarr store plus the train split into
`data/processed/normalization_stats.json`. DVC re-runs a stage only when its
dependencies changed: the stats stage depends on just the `train_datasets` and
`zarr_store` keys of `configs/data/re16k.yaml` (via `params`), so editing the
val/test names doesn't invalidate it. `dvc.lock` pins the hash of every input
and output, so which data and which stats a commit used is recorded in git.
The stats file embeds no timestamp, so re-running the stage on the same inputs
reproduces it byte for byte (the lock file doesn't change).

The lock is only true if it's refreshed with every change to what a stage
depends on. `tests/data/test_dvc_lock.py` asks DVC for the status of every
stage's code and params deps (not the data, which CI doesn't have; ~0.5 s)
and fails if any changed since `dvc.lock` was written, so a PR that edits,
say, the evaluation code without re-running the `train` stage fails CI
instead of leaving the canonical model's `metrics.json` claiming code that
no longer exists (#52 slipped through exactly that way; #54 refreshed the
lock).

Design choices:

- **Raw only.** The zarr store is `cache: false`: it's a regenerable ~9 GB
  derivative, so it isn't copied into DVC's cache or pushed to the remote
  (its hash is still in `dvc.lock`, and `dvc repro` rebuilds it in about a
  minute). The small stats file *is* cached and pushed, so `dvc pull` gets the
  exact stats without re-downloading the raw data.
- **Directory, not per-file.** `data/raw` is tracked as one directory, so
  `data/raw/.gitkeep` is gone and the old `.gitignore` entries for it were
  dropped; DVC writes `data/.gitignore` instead.
- **Hardlinked cache.** DVC copies by default, which would store the 10 GB
  twice; this machine's `.dvc/config.local` sets `cache.type =
  hardlink,copy` (a per-machine choice, not committed). The cache files are
  read-only, so don't edit raw files in place.
- **The test dataset.** `re16k_t400_5.npy` is in the tracked directory, and
  `dvc pull`/`dvc add` move its bytes around. That's versioning, not reading:
  no stage opens it, and `compute_stats` reads only `train_datasets`.
- **Not in the container.** DVC is a dev dependency, so the runtime image has
  no `dvc`; run `dvc pull`/`repro` on the host, and the container sees the
  results through the `data/` bind mount (and `dvc.lock`, mounted read-only,
  which the provenance logging below reads).

**Staying in sync is the workflow's job.** Nothing in `train.py` checks that
the zarr store and stats on disk match `dvc.lock`; the way to be sure is to
produce them with `dvc repro`, which re-runs exactly the stages whose inputs
changed (and `dvc status` shows what's out of date). That's the usual DVC
division of labor, and for the canonical model the `train` stage below makes
it hold by construction. An enforcing
check inside the training script was tried and dropped: it needed a
DVC-only image and a compose dependency to cover the container, and could
still be bypassed by starting the image directly, so it was a guarantee that
only held on some paths.

**Provenance in MLflow.** Each run logs the hashes `dvc.lock` pins as params
(`data_version.raw_md5`, `.zarr_md5`, `.stats_md5`, searchable in the UI) and
uploads `dvc.lock` and the normalization stats file as artifacts. Together
with MLflow's own git-commit tag, a run can be traced to the code, raw data
and statistics it was started against. These are the hashes the lock *pins*,
not a verification of what was on disk: they're right when the pipeline was
run through `dvc repro`, and wrong if the data was changed behind its back.

CI doesn't pull data (the remote is local), but `tests.yml` runs `dvc dag`
so a malformed `dvc.yaml` fails the PR.

### Training stage

```bash
uv run dvc repro train      # refit only if data, code or the relevant config changed
uv run dvc metrics show     # the canonical model's validation (and train) scores
uv run dvc metrics diff     # ... compared with the last commit
```

DVC is more than data versioning: `dvc.yaml` is a pipeline of stages with
declared dependencies, parameters and outputs, and the `train` stage extends
it from the data to the **canonical model** -- Hankel DMD at the delays and
rank its validation sweep selected (see "Delay sweep"; before it, plain DMD
at rank 750). It depends on the zarr store, the normalization stats,
`train.py`, the whole `src/mhd_surrogate` package and the grid config, plus
the config keys that change the fitted model or its scores (`model.rank`,
`delays`, `spatial_rank`, `stabilize`, the split, `context_steps`,
`std_mode`, the evaluation settings; performance knobs like chunk sizes are
left out so tuning them doesn't retrain). Its outputs are the checkpoint
(`models/hankel_dmd/model`, ~890 MB, cached and pushed like the stats) and
`models/hankel_dmd/metrics.json`, declared as DVC metrics and kept in git, so `dvc metrics diff` shows how a code change
moved the scores between commits. `dvc.lock` then pins exactly which data,
code and config produced the committed model, and `dvc repro` rebuilds it only
when one of them changes. The stage runs `train.py` with `export.dir` set,
which puts the checkpoint and metrics at that fixed path instead of the
run's timestamped Hydra directory; it still logs to MLflow like any run.
`.dvcignore` excludes `__pycache__`, which importing the package would
otherwise change, making the stage look out of date after every run.

**Not bitwise reproducible, and why that's fine.** Unlike the data stages
(re-running `convert_to_zarr` and `compute_stats` reproduces their hashes
exactly), retraining gives a checkpoint with different bytes every time: the
GPU's float32 reductions aren't bitwise deterministic, so two runs on the same
inputs differ around the 8th significant digit of every score. `dvc.lock`
therefore records what *was* produced, and `dvc repro` guarantees the model
matches the current data, code and config -- not that a retrain reproduces it
bit for bit. To keep that noise out of git, `metrics.json` is rounded to 6
significant digits, so retraining an unchanged model leaves it unchanged.
(Use `dvc repro -f -s train` to force a retrain: `-f` alone re-runs the whole
chain leading to the stage, data stages included.)

**Retrains and cache growth.** The stage's code deps are exactly the modules
`train.py` imports, transitively -- not the whole package -- so editing an
unrelated module (`analysis/pod.py`, say) doesn't make it stale. A test
(`tests/training/test_dvc_train_stage.py`) computes that import set and fails
if a module isn't covered, so the list can't drift into calling a stale model
up to date. Only the canonical model's modules are deps, not all of `models/`:
`models/registry.py` maps each model name to an import path
(`"mhd_surrogate.models.dmd:DMD"`) and imports a model's module only when that
model is built or loaded, so adding or editing another model (a baseline, a
neural network) doesn't mark the canonical stage stale. The test can't see a
string import in the import graph, so it adds the trained model's module
itself, looked up from the stage's `model=` override; switching the canonical
model makes the test name the new model's files as missing deps. Retraining is
always explicit (`dvc repro`), and every retrain adds a new ~880 MB cache
entry -- even for an irrelevant change, since the checkpoint's bytes differ
each time -- and old entries are never removed automatically. Clean up the
local cache now and then with

```bash
uv run dvc gc --workspace --all-commits
```

which keeps every version any git commit references and deletes the retrains
that were never committed. Don't add `--cloud` without thinking: that deletes
from the remote too. When only the dep *list* changes, not the code, `uv run
dvc commit -f train` records the new deps against the existing model instead of
retraining.

**When to retrain, and what it doesn't touch.** Retraining on a new commit
never changes an earlier result. Every commit's `dvc.lock` pins the exact
data, code, config and checkpoint, and its `metrics.json` holds the scores, so
a tagged result stays reproducible: `git checkout <tag> && dvc pull` restores
that exact model and its scores, and `dvc repro -f` on that commit retrains it
from that commit's code to check them (equal up to GPU float noise, hence the
rounding). Retraining on `main` serves a different purpose: keeping the
committed model consistent with the committed code. Retrain deliberately, not
on every commit -- in the PR whose change is *meant* to alter the model (a
fix, new data, a newly selected parameter), so its `metrics.json` diff shows
by how much next to the code that caused it; or as a regression check after a
risky change that *shouldn't* alter it, where an unchanged `metrics.json`
confirms it didn't. A stage left stale by an unrelated change can wait for
the next deliberate retrain.

**Division of labor with MLflow.** MLflow is the experiment record: every
run, sweep and model, with its metric histories and (later) the registry --
exploration happens there, through Hydra multiruns. DVC answers a different
question for the few canonical artifacts: exactly what produced this, and is
it still up to date? `dvc exp run` is deliberately not used for sweeps, so
there aren't two experiment systems to look in. The final evaluation (refit
the finalists on train + val, score them on test, see "How validation and
test are used") will become stages too, added only once the finalists are
frozen and marked `frozen: true`, so `dvc repro` can never re-read the test
set by accident.

## Train / val / test split

The split is at the *dataset* level, not a per-dataset time split:
`configs/data/re16k.yaml` designates 7 of the 9 datasets for training (used
in full), 1 (`re16k_t400_6`) held out in full for validation, and 1
(`re16k_t400_5`) held out in full as the final test set. All 9 datasets are
the same DNS run (Re=16000) restarted with small randomness -- different
realizations of one regime, not different physical parameters (see "Origin"
above) -- so training on multiple realizations is the natural default, and
holding out whole datasets is what actually tests generalization to an
unseen realization, rather than just forecasting further into one the model
already saw part of. val/test were chosen via a seeded random draw over all
9 (`np.random.default_rng(0).choice(datasets, size=2, replace=False)`), not
cherry-picked.

**`re16k_t400_5` (the test set) must never be read by any script, for any
purpose, including this project's own analysis suite below.** It's excluded
from `configs/analysis/split.yaml`'s `datasets` list for exactly this
reason -- don't add it back. `re16k_t400_6` (val) stays in that list; val
data is fair game to look at during development, unlike test.

**How validation and test are used.** Validation is where every decision is
made: hyperparameters, early stopping, and also comparisons *between* models
(architecture, window length, ...). Choosing between models is tuning at a
higher level, so choosing by test score would bias the winner's test score
upward, by more the more candidates are compared that way. The test set is
read once, at the end:

1. Freeze the finalists: the baselines plus the learned models chosen on
   validation.
2. Refit each on train + val. A refit can't early-stop (there is no held-out
   data left), so it uses the epoch count from its tuned run, scaled for the
   extra data if warranted. The normalization stats are recomputed over
   train + val as well: normalization is just the first step of fitting a
   model, and fitting it on less data than the rest of the model would need a
   reason of its own. (During development they stay train-only, as the
   `compute_stats` stage computes them now; the train + val stats will be a
   separate output when the final evaluation is built.) The baselines are
   refit too (the mean field over all 8 datasets), so the comparison stays
   fair.
3. Score all of them on test in one evaluation, and report that table.
   Nothing is changed or chosen after seeing it.

Reporting every finalist's test score side by side is the point of step 3;
what would spoil the test set is going back to iterate after looking at it.

**How candidates are ranked.** A surrogate is useful for as long as its
forecast stays close to the truth, and only if it stays bounded after that.
The primary criterion is the validation **stable steps**: how many leading
steps of the forecast stay bounded (see "Forecast evaluation"; the whole
837 for a stable forecast). Then the validation **skill horizon** (lead
steps with RMSE <= 0.5), ties broken by **RMSE at lead 10**. Stability comes
first because the short-horizon scores can't see a blow-up: the first
U-Net search picked, by skill horizon alone, models whose energy is 1000x
the truth's by the end of the validation forecast (see "First search").
Among stable forecasts, which is what a usable candidate has to be, the
rule is skill horizon and RMSE at lead 10 as before; among unstable ones,
lasting longer wins, so early stopping and the search are pushed toward
stability even while nothing is stable yet. The mean RMSE over all leads is deliberately not used:
after a few dozen steps every forecast of this chaotic flow has decorrelated,
and from there pointwise error favors whatever sits closest to the mean -- in
the DMD rank sweep it ranks the smoothest, lowest-rank model first. The
physics scores are **guardrails**, not objectives (optimizing one directly
invites gaming it): a candidate replaces the incumbent only if it is stable
over the whole validation forecast, its energy and
enstrophy errors are no more than 0.03 worse and its x/y spectrum distances
no more than 0.05 worse. Those tolerances are about the spread a different
realization of the flow scores (energy <1%, enstrophy <3%, spectrum
0.01-0.03, see "Forecast evaluation"), so a candidate isn't rejected for a
difference that is noise. The oscillation scores are reported but not used to
decide: on a single validation realization their noise (period error
0.15-0.19 for perfect dynamics) is as large as the differences between
models so far.

One caveat: validation is a single realization of the flow, so decisions made
on it are noisy. Scoring a different realization with perfect dynamics
against it already gives a `u_y` period error of 0.15-0.19 (see "Forecast
evaluation"), so two models closer than that spread are not reliably
different. For decisions that matter, such as the final architecture,
leave-one-dataset-out cross-validation over the 8 non-test datasets gives a
spread instead of a single number; at one training per fold it is reserved
for the shortlist, not every experiment.

Every `scripts/analysis/*.py` script below reads `configs/analysis/split.yaml`
directly and analyzes each configured dataset's *entire* recorded length --
there's no more per-dataset region to precompute or bound reads by, since
only whole-dataset exclusion matters now.

```bash
uv run scripts/analysis/check_split.py  # describes each configured dataset in full
```

`check_split.py` computes, per time step, the spatial mean/std/min/max of
each channel plus (when there are 2 channels, i.e. velocity components) a
kinetic energy proxy per direction (`0.5*u_x^2`, `0.5*u_y^2`) and in total,
and the spatial `u_x`-`u_y` correlation. It produces two plots per dataset:
every series over time (`<name>_split_check.png`) and a value histogram per
channel (`<name>_split_hist.png`) — so a drift or trend would be visible
rather than hidden inside a single aggregate number.

For `re16k_t400_0`: `u_x` mean 1.0, std 0.822, range [-3.23, 4.87]; `u_y`
mean ~0 (-0.0002), std 0.424, range [-2.97, 2.63] -- `u_x` (streamwise)
carries most of the flow's energy and variance, as expected. Kinetic energy
is 0.838 (`u_x`) + 0.090 (`u_y`) = 0.928 total, with small std relative to
the mean (~2%) -- consistent with the "slow energy variation" found in the
autocorrelation section being a real but modest-amplitude effect. `u_x`-`u_y`
correlation is essentially zero (0.006 +- 0.039), i.e. the two velocity
components are spatially uncorrelated on average, as expected for a
shear-dominated channel flow. (These specific numbers predate the switch to
full-length reads and haven't been re-run against the extra ~34-46% of each
dataset now in scope; given the stationarity check below already found no
meaningful shift between a dataset's first and second half, they're not
expected to move much -- worth a refresh before relying on exact values,
not worth blocking on.)

### Earlier: a per-dataset time split (retired)

Before the dataset-level split above, `configs/analysis/split.yaml` also
configured a *within*-dataset trailing split -- `val_steps`/`test_steps`/
`buffer_steps`, a generated `split_manifest.json`, and every
`scripts/analysis/*.py` script bounding its reads to a `[0, trainval_end)`
region -- so that developing the analysis suite couldn't see data that
would later be held out for evaluating a model trained per-dataset. That
plan (train/evaluate one dataset at a time) is superseded now that all 9
datasets are confirmed to be one regime rather than different ones (see
"Train plan" reasoning above): training pools multiple datasets, held out
*whole*, so a within-dataset time split no longer serves the purpose it was
built for. `trailing_split`, `scripts/data/split_data.py`, and the
generated manifest are gone; `scripts/data/dataset.py`'s `load_full_dataset`
(whole-dataset windowed sampling) replaced the manifest-based loader.

That earlier design is kept here as history, not deleted outright, because
the numbers it produced are still referenced elsewhere in this document
(the autocorrelation section's `buffer_steps` discussion, and the ~24-40
step reference period T=40 that motivated `val_steps`/`test_steps`'s
sizing): `val_steps` (160) and `test_steps` (400) were fixed absolute step
counts (`4T`/`10T`), not fractions of a dataset's length, since seven
independent analyses below all converged on that ~24-40 step dominant
oscillation as this flow's best-characterized timescale; `buffer_steps`
(20 = `T/2`) dropped that many snapshots at each internal boundary, derived
from the autocorrelation check's finding that direct field decorrelation
crosses zero within 6-11 steps across all 9 datasets. See "Where this
leaves the split-sizing question" below for the full original reasoning.

**A note on everything below:** most of the findings in the rest of this
document -- "across all 9 datasets", specific numbers for `re16k_t400_0`,
etc. -- were produced while all 9 datasets were still in scope for the
analysis suite and each script read a fixed-size `train+val` region rather
than a dataset's full length. Re-running any script now touches only the 8
configured datasets (never `re16k_t400_5`) and reads each one in full. The
qualitative conclusions aren't expected to change -- the stationarity
check below already found no meaningful shift between a dataset's first
and second half, which is direct evidence the extra data wouldn't shift
things much -- but the exact numbers throughout predate this change and
haven't all been individually re-verified against it.

## Incompressibility check

```bash
uv run scripts/analysis/check_divergence.py [--dx DX --dy DY]
```

Computes `div(u) = du_x/dx + du_y/dy` per time step (second-order central
differences, layout assumed `(T, 2, Nx, Ny)` with `x` = axis 2) over the
train+val region, and plots the RMS divergence, the RMS normalized by the
RMS of the two derivative terms, and divergence maps at the first/middle/last
step of that region. The long axis (axis 2) is the streamwise x direction.
Grid spacing is derived from the domain lengths in `configs/analysis/grid.yaml`
(see the Grid section): `dx = 25/1150`, `dy = 2/126`. `--dx/--dy` override
the config (e.g. `--dx 1 --dy 1` for grid units).

For `re16k_t400_0`'s train+val region, the normalized divergence is ~0.42
(RMS divergence ~1.05, essentially unchanged from the pre-overhaul ~0.41,
as expected since divergence was already stationary in time). In grid units
(`dx = dy = 1`) it was ~0.53, and swapping the two axes was much worse
(~0.93 in grid units), confirming `x` = axis 2 (measured against the
previous split; not re-verified, but this is a static geometric check
unaffected by the split boundary). The divergence maps show large-scale
structure tied to the flow features rather than grid-scale noise.

The divergence is only mildly sensitive to the exact `dx/dy` ratio: a
least-squares fit on 13 snapshots (best-fit ratio ~1.66) gives ~0.37, close
to but not equal to the ~0.41 from the confirmed ratio (~1.37, i.e.
`dx/dy = (25/1150)/(2/126)`). So that fit was never a reliable way to infer
the grid spacing from the data alone -- it was a rough cross-check, not a
source of truth, and the domain length has since been supplied directly
(and corrected once already; see the Origin section).

So the 2D field is not exactly incompressible, which is expected: the
quasi-2D hypothesis is only approximate, so `du_z/dz` in the slice does not
vanish, and this residual (~41% of the derivative magnitude) is a rough
measure of how far the slice is from ideal 2D. Whether it is "good enough"
is a modeling judgement. The linear interpolation in y (piecewise-constant
`du_y/dy`, smoothed wall layers) and the DNS's own discretization would also
contribute and cannot be separated from `du_z/dz` here. Since the flow varies
little along z away from the Hartmann layers (see the Origin section), a
large `du_z/dz` alone would be somewhat surprising, so the interpolation and
discretization may account for more of this residual than first assumed; this
was not tested.

## Vorticity and enstrophy

```bash
uv run scripts/analysis/check_vorticity.py [--dx DX --dy DY]
```

Computes the out-of-plane vorticity `w = du_y/dx - du_x/dy` (same central
differences, layout and `configs/analysis/grid.yaml` spacing as the divergence check)
and the enstrophy proxy `0.5*w^2` (spatial mean per time step, matching the
`0.5*u^2` energy convention), over the train+val region. Only the
z-component exists in a 2D slice, so this is a 2D enstrophy, not the full 3D
one. It prints mean/std for the mean vorticity and the enstrophy, plots both
over time, and maps the vorticity at the first/middle/last step of that
region.

Vorticity scales with 1/length and enstrophy with 1/length^2, so the absolute
values below depend on the domain lengths in `configs/analysis/grid.yaml`. The
`du_x/dy` term inherits the y-interpolation caveat (see the Grid section):
the thin wall layers, which dominate the vorticity extremes, are the part of
the field most affected by the linear interpolation onto the uniform y grid.

For `re16k_t400_0`'s train+val region, the maps show shear layers and jets
near the inlet (x < ~150), large coherent vortices of roughly channel-width
size downstream, and thin high-vorticity layers along both y walls, which
dominate the extremes. Mean enstrophy is ~27.7 (essentially unchanged from
the pre-overhaul ~27.5), with slow variations over time; the spatially
averaged vorticity is a regular oscillation (period ~25-30 steps, amplitude
~0.03; see the autocorrelation section) with a small mean (~0.012), tiny
compared with the local vorticity magnitude of order 10-20.

## Spatial power spectrum

```bash
uv run scripts/analysis/check_spectrum.py [--dx DX --dy DY]
```

Computes 1D power spectra E(k) of `u_x` and `u_y` along x (axis 2) and along
y (axis 3), each averaged over the other spatial axis and over time, over the
train+val region. The domain is not periodic (inlet, walls in y), so each
line has its mean removed and a Hann window applied before the FFT. Spectra
are one-sided and normalized so that E(k) integrated over the angular
wavenumber equals the window-weighted variance; this was verified on
synthetic sine waves (integral 0.500 for a unit-amplitude sine, peak at the
expected wavenumber). Output: a log-log plot (`<name>_spectrum.png`) and a
summary of the peak and total variance per direction and channel.

Wavenumbers use the domain lengths in `configs/analysis/grid.yaml` (`k = 2*pi /
wavelength` in physical units; see the Grid section). Because the flow is not
homogeneous in x (the inlet region differs from the developed region), the x
spectrum averages over a non-stationary signal. In y the data was linearly
interpolated from a non-uniform DNS grid, which acts as a smoothing filter
and gives a piecewise-linear profile, so the high-k part of the y spectrum
cannot be attributed purely to the flow.

For `re16k_t400_0`'s train+val region (re-measured after the overhaul above;
peak location and shape are essentially unchanged from the pre-overhaul
numbers, as expected since the spectrum is dominated by spatial structure
rather than the exact time range averaged over):

- **x direction:** the spectrum peaks at k ~ 2.0 (wavelength ~3.1, about 1.6
  channel widths for `ly=2`), on the scale of the large coherent vortices seen
  in the vorticity maps (a rough visual match, not measured). Above the peak
  it decays as a power law with slope ~ -2.9 to -3.0 over k ~ 10-90 for `u_x`
  (log-log fit; ~-2.9 for `u_y` over the same range, noisier at k ~ 3-10:
  -2.6 to -3.1). It flattens at the highest wavenumbers (the axis ends at
  k ~ 144); the cause was not investigated (grid-scale content, leakage from
  the wall layers, numerical noise, or aliasing from the factor-2 x
  subsampling if no filter was applied are all possible).
- **y direction:** no interior peak; the spectrum decreases monotonically
  from the lowest resolved wavenumber (dominated by the cross-stream
  profile and wall layers), with slope ~ -3.6 for `u_x` and ~ -3.2 for `u_y`
  over k ~ 10-60, flattening above k ~ 130. `u_x` has roughly 10x the power
  of `u_y` at low k, closing to a factor of a few at high k (read from the
  plot). Given the interpolation, these y slopes and the high-k flattening
  are probably not clean flow properties.
- The slope of ~ -3 in x (-2.9 measured) is the classic 2D enstrophy-cascade value, but it
  is also what smooth fields dominated by isolated vortices and shear layers
  give, and the 1D spectra of a bounded, inhomogeneous domain are not a
  clean test, so this is consistent with rather than evidence for such a
  cascade.

The train-vs-test spectral comparison this section used to report is gone:
it directly compared test-region power against train, which is exactly the
kind of held-out-set snooping the train+val/test overhaul above removes.

## Temporal autocorrelation

```bash
uv run scripts/analysis/check_autocorrelation.py [--max-lag N]
```

Two analyses, both over the train+val region only, with lags in snapshot
steps (the physical time between snapshots is not stored in the data, so no
physical time scales here):

1. **Field autocorrelation:** the pointwise fluctuation `u' = u - <u>_t`
   (about the train+val time-mean field) correlated with itself at lag `tau`,
   pooled over all grid points via an FFT along time, per channel.
2. **Scalar autocorrelation** over the whole train+val series: spatial means
   of `u_x` and `u_y` and the per-direction kinetic energy, with the
   approximate `+-1.96/sqrt(N)` white-noise band.

The estimators were validated on synthetic AR(1) data (measured
autocorrelation matches `phi^lag` to within ~0.005 at every lag, and is
insensitive to a mean offset and scale). The reported lags, integral time
`tau_int` and `N_eff = N/(2*tau_int)` are rough: `tau_int` is truncated at
the first zero crossing, which ignores the negative lobe and the recurring
oscillations described below, and the tail of the estimate is noisy.

For `re16k_t400_0`'s train+val region (729 steps):

- **Fast decorrelation, then oscillation.** The field autocorrelation
  (`u_x`: rho(1) = 0.898, `u_y`: 0.824) falls below 1/e after 6 and 5 steps
  respectively and crosses zero after 9 and 7 steps, with a negative lobe
  (down to ~-0.4 near lag ~14). It then keeps oscillating with a period of
  ~30 steps (peaks near lags 32, 62, 88, 116, 146, still ~0.3 at lag ~146),
  so the flow contains a persistent quasi-periodic component.
- **Slow energy variation.** The kinetic energy per direction has much
  longer memory than the field autocorrelation (`tau_int` 30.9 steps for
  `u_x`, 16.7 for `u_y`), with `N_eff` over train+val of only ~12 (`u_x`)
  and ~22 (`u_y`) — even the sized-up val/test windows below hold just a
  handful of effectively independent samples of this slow variation. This estimate is itself
  sensitive to how much of the series is available (an earlier, longer
  pre-overhaul window found a noticeably different tail), since the tail of
  the FFT-based estimate is noisy for a diagnostic this slowly varying
  relative to the window; treat it as rough and expect it to keep moving as
  the analysis window changes.
- **Implication for the split, checked across all 9 datasets:** snapshots
  are highly correlated at short lags, so a random split would leak
  information, as assumed. The direct (monotone) correlation crosses zero
  within 6-11 steps and drops below 0.05 within 6-10 steps in every
  dataset/channel -- consistently tight, and this is the part a buffer
  actually removes. The recurring oscillatory correlation is a different
  matter: every dataset shows one (period ~24-40 steps, dataset-dependent),
  and its amplitude does *not* decay away -- it's still routinely 0.2-0.5 at
  lag ~150 in every dataset checked, sometimes higher for the scalar
  diagnostics. No buffer size within a practical fraction of the train+val
  region removes this component; it's a structural feature of the dynamics,
  not something a gap between regions fixes. Given that, `buffer_steps: 20`
  in `configs/analysis/split.yaml` clears the fast decorrelation with margin
  (roughly 2x the worst-case 11-step zero crossing) without pretending a
  larger buffer would meaningfully reduce the periodic component's
  contribution at the boundary -- that would cost train+val data for
  diminishing returns instead.

Only temporal autocorrelation is covered; spatial autocorrelation (integral
length scales) and the enstrophy autocorrelation are not.

## Stationarity check

```bash
uv run scripts/analysis/check_stationarity.py
uv run scripts/analysis/check_stationarity.py --dataset re16k_t400_0
```

Every method below (Welch spectra, DMD, POD, SPOD) implicitly assumes the
process is statistically stationary over the analysis window -- Welch
averages segments together, DMD fits one linear operator for the whole
window, SPOD averages cross-spectral density across segments -- but none of
them test that assumption directly. The DNS discards a 400-step warm-up
(spin-up) phase before the saved series begins (see "Origin: from the 3D
DNS to this dataset" above), so a strong startup transient wasn't expected
in what's recorded, but that's a design choice made upstream of this
project, not something verified against the actual data here until now.

This is a lightweight, direct check: split train+val into two contiguous
halves and compare, per channel, the spatial mean/std, the kinetic energy
proxy, and the domain-mean series' own dominant period (a plain periodogram
per half -- each half is already short, and Welch would need to shrink
segments further to fit). A meaningful shift between halves would be direct
evidence of non-stationarity within the window.

Across all 9 datasets, the mean is flat: `u_x`'s half-to-half change is
under 0.3% everywhere, and `u_y`'s (reported as an absolute delta, not a
percentage, since its mean is ~0 by the channel's own symmetry -- a
relative change there is a near-zero-denominator artifact, not a
meaningful number) stays at the ~1e-4 level in both halves. `u_y`'s
dominant period -- the meaningful one to check, since `u_y` (unlike `u_x`)
has a sharp, isolated spectral peak to begin with -- stays firmly in the
already-established ~24-40 step range in both halves of every dataset,
several nearly unchanged (`re16k_t400_10`: 29.5 -> 29.5, `re16k_t400_9`:
34.6 -> 34.7): direct evidence the dominant oscillation itself is a stable,
recurring feature, not an artifact of analyzing the whole window at once.
(`u_x`'s own "dominant period" swings much more between halves, e.g.
51.8 -> 207.0 for `re16k_t400_0` -- expected, not new evidence of
non-stationarity, since `u_x`'s spectrum has no clean isolated peak to
begin with, a broad low bump as already found in the domain-mean spectrum
section below; a short periodogram on a peakless spectrum bounces around
regardless of stationarity.)

One honest secondary finding: `std`/energy shows a small, fairly consistent
*increase* from first to second half for `u_x` in 8 of 9 datasets
(+0.8% to +3.7%; `re16k_t400_9` is the one decrease, -2.2%), and a more
variable one for `u_y` (-3.3% to +21.4%, `re16k_t400_10` notably higher
than the rest). Small enough not to undermine the spectral/DMD/POD/SPOD
methods' stationarity assumption, but a real, small, mostly-consistent
drift in variance worth keeping in mind rather than treating the question
as fully closed.

## Frequency analysis

The autocorrelation check above found a persistent quasi-periodic
component but could only infer its period indirectly, from zero crossings.
The seven scripts below read it off directly, from several independent
angles -- four spectral (FFT-based), then three based on data-driven mode
decompositions (DMD, POD, SPOD) -- and are what train/val/test ended up
sized off of (see the Train / val / test split section above). They are
exploratory and, unlike the check scripts above, not wired into
`run_all_checks.py`. The shared FFT machinery (`power_spectrum`,
`welch_spectrum`, `dominant_periods`) lives in
`src/mhd_surrogate/analysis/spectral.py`, used by the first two scripts
below.

### Point spectrum

```bash
uv run scripts/analysis/check_point_spectrum.py
```

FFT of `u_x(t)` and `u_y(t)` at a few fixed grid points, over train+val.
Two estimators, plotted together: a plain periodogram, and Welch's method
(averaging the periodogram over overlapping segments). A single
periodogram bin is a high-variance estimate (effectively 2 degrees of
freedom) -- not strong evidence against being a random fluctuation on its
own -- so a peak that survives Welch's averaging is real, not an artifact
of that variance; this was checked directly against synthetic white noise
(the estimated variance at a bin drops roughly as expected from averaging
independent segments).

Default points: `core_mid`/`core_downstream` (mid-channel, two streamwise
stations) and `wall_bottom`/`wall_top` (near each wall, same downstream
station as `core_downstream`) -- all past the inlet's developing region. A
near-inlet point was tried and dropped: it showed a persistent negative
mean `u_x` (backflow) in every dataset, and watching the video confirms
that region is either backflow or laminar, neither useful here.

Across all 9 datasets, every point locks onto the *same* dominant period
per dataset (~24-40 steps, matching the autocorrelation check's range),
with a physically sensible pattern in how it shows up: at the walls it
dominates `u_x` (up to 43% of variance in one Welch peak), in the core it
dominates `u_y` (up to 51%) -- consistent with a no-slip wall suppressing
the wall-normal (`u_y`) component. Welch confirms, and often sharpens, the
periodogram's peak wherever it was already strong; where the periodogram
peak was weak, Welch correctly declines to confirm it rather than
manufacturing a spurious one.

### Spatially averaged spectrum

```bash
uv run scripts/analysis/check_spatial_mean_spectrum.py
```

Same periodogram + Welch analysis, applied to the domain-averaged series
instead of a single point: the spatial mean of `u_x`, `u_y`, and the
per-channel kinetic energy proxy `0.5*u^2` (matching `check_split.py`'s
per-timestep spatial means/energy).

`u_y` gives the cleanest signal in the whole suite: a single, sharp,
isolated peak at the same ~24-40 step period, with Welch power fractions
of 30-51% and essentially no competing peak, across all 9 datasets.
Averaging over the *whole* domain reinforcing rather than washing out the
signal confirms it is a coherent, domain-wide mode, not a localized
artifact. `u_x` shows no such peak -- a broad, low bump with periodogram
and Welch disagreeing on the exact period -- and the energy series show a
monotonically rising ("red") spectrum with no isolated peak at all, still
climbing at the longest period the window allows. `dominant_periods`'s
"top period" for a still-rising spectrum like that is an edge artifact
(the boundary bin trivially beats its only neighbor), not a real spectral
line -- documented directly in its docstring after this was caught by
looking at the plot, not the printed numbers. The corrected reading:
global kinetic energy has no resolvable characterizing period within a
single train+val window (~500-730 steps); its true slow timescale exceeds
what this much data can measure, consistent with (and more precise than)
the pre-overhaul note about a slow ~500-700 step oscillation.

### 2D wavenumber spectrum

```bash
uv run scripts/analysis/check_wavenumber_spectrum.py
```

The "Spatial power spectrum" section above computes two 1D spectra,
`E(kx)` and `E(ky)`, each averaged over the *other* spatial axis; it can't
say whether energy at a given `kx` and a given `ky` occur together in the
same structure. This keeps both wavenumbers: a full 2D FFT per snapshot,
averaged over time, over the train+val region.

Getting this right needed a real fix along the way. The first version
detrended each snapshot by its own scalar spatial mean, which leaves a
channel's (near time-invariant) mean *profile* shape -- e.g. `u_x`'s
cross-channel shear profile -- inside what gets called the "fluctuation".
Present in every snapshot, it dominated the time-averaged spectrum as a
broad blob centered at `kx=0`, drowning out the actual coherent structure.
Fixing it to subtract the true time-mean *field* (matching
`check_autocorrelation.py`'s `u' = u - <u>_t` convention) moved the peak
from `kx=0` to `kx~-2` -- verified numerically, not just by eye: the
reported peak is ~54x the power at the origin, not merely nominally
different.

Across all 9 datasets, `u_x` and `u_y` consistently agree on
`|kx| ~ 1.76-2.26` (wavelength ~2.78-3.57, matching the ~2.0/~3.1 found in
the 1D spectrum above), and `u_x` additionally shows `ky=3.12` (wavelength
~2.02, essentially the channel width `ly=2`) in every dataset, while
`u_y`'s peak sits at `ky=0`.

### Space-time (dispersion) spectrum

```bash
uv run scripts/analysis/check_spacetime_spectrum.py
```

The wavenumber and the period above were found independently -- they could
be two unrelated features of the flow that merely coexist. This computes
the joint spectrum `E(kx, ky, omega)`, a full 3D FFT over x, y *and* time
together, so a peak at their combination is direct evidence of one
coherent, moving structure (a traveling wave, with phase velocity
`omega/kx`), not a coincidence of two separate analyses.

Unlike every other script here, this can't process time in independent
chunks -- it's itself an FFT axis -- so it loads the whole train+val
region for one channel into memory at once (matching
`check_autocorrelation.py`'s `field_acf`). That peaks at ~6 GB and
~15-20s per dataset/channel, clearly heavier than the rest of the suite,
so it is deliberately not run in parallel across datasets the way
`run_all_checks.py` runs the other scripts.

Across all 9 datasets, `u_x` and `u_y` peak at the same period (matching
to one decimal in 8 of 9 datasets) and the same `|kx|` -- direct
confirmation these are one linked structure, not separate coincidences.
The dispersion plot shows something more nuanced than a single clean
traveling wave, though: a nearly *vertical* ridge -- `kx` staying close to
~1.8-2.3 across almost the entire `omega` range, rather than a diagonal
line through one frequency. The dominant spatial wavenumber (vortex size)
is present across a broad range of frequencies, with only a statistical
preference toward the ~24-40 step period, rather than a pure periodic
oscillator -- consistent with quasi-periodic vortex shedding, not a clean
single-tone wave. (A minor cross-check note: this script's single-bin peak
`kx` can differ slightly from the 2D script's `omega`-pooled peak `kx` for
the same dataset -- expected, since the ridge is not perfectly uniform
across `omega`, not a discrepancy between the two analyses.)

### Dynamic Mode Decomposition (DMD)

```bash
uv run scripts/analysis/check_dmd.py --dataset re16k_t400_0
uv run scripts/viz/make_dmd_video.py --dataset re16k_t400_0
```

The scripts above all assume energy sits on a fixed grid of FFT
frequencies. DMD instead fits the best linear dynamical system `A` (in a
least-squares sense, `x_{t+1} ~= A x_t`) directly to the snapshot sequence
and eigendecomposes it: each eigenvalue gives a mode's frequency *and*
growth/decay rate, at whatever frequency the data actually supports, not a
grid-locked bin. Uses "exact DMD" (Tu et al., 2014); `u_x` and `u_y` are
stacked into one state vector per snapshot (DMD models their joint
dynamics, not each channel independently), with the time-mean field
subtracted first. The SVD is truncated to rank 100 for robustness against
small, noise-dominated singular values -- standard DMD practice; for
`re16k_t400_0` this captures 91.5% of the variance.

Modes are selected by "power" (amplitude at the first snapshot, the
standard DMD convention), but a fast-decaying mode can have high power yet
vanish within a handful of steps, contributing far less to the actual
recorded series than a lower-power but near-neutral mode that persists
throughout. `re16k_t400_0` shows this directly: mode 94 has the highest raw
power (101) of the top 5, but decays at -0.0255/step (half-life ~27
steps), giving it an rms-power-over-the-whole-window of only 15.8 --
*fourth* of the five once decay is accounted for. Mode 70 (period 29.3
steps, matching the range found above) has lower raw power (85.2) but
barely decays (-0.00236/step, half-life ~294 steps), so its rms power
(42.7) is actually the highest of the five -- the opposite ranking. Fixed
by adding an `rms_power` (root-mean-square power over the whole recorded
window) alongside `power`; `check_dmd.py` prints modes sorted by it, and
both the mode-shape plot and `make_dmd_video.py`'s default mode selection
use it too. The top `rms_power` mode's shape matches the same coherent
structure the spectral methods above found -- a genuinely different method
(a fitted linear operator, not any kind of FFT) arriving at the same
answer.

### Proper Orthogonal Decomposition (POD)

```bash
uv run scripts/analysis/check_pod.py --dataset re16k_t400_0
uv run scripts/viz/make_pod_video.py --dataset re16k_t400_0
uv run scripts/viz/make_pod_video.py --dataset re16k_t400_0 --mode-index 0 --mode-index 1 --combine
```

POD (aka PCA) gives up DMD's per-mode frequency (a POD mode's time
coefficient typically mixes many frequencies) in exchange for modes that
are mutually orthogonal and exactly ranked by the variance ("energy") they
capture over the whole window, with no eigenvalue problem or complex
arithmetic -- just the SVD of the same (channel-stacked,
mean-subtracted) state matrix DMD uses. It's also cheaper: no
eigendecomposition or least-squares step.

Across all 9 datasets, modes 0 and 1 have nearly identical energy --
e.g. `re16k_t400_0`: 12.79% vs 12.43%; `re16k_t400_4`: 21.38% vs 21.24%;
`re16k_t400_10`: 17.08% vs 16.77% (`re16k_t400_5` is the one partial
exception, at 17.08% vs 13.56%). This is the textbook signature of POD
needing *two* modes in spatial quadrature to represent one traveling
structure, since a single real orthogonal mode can't do it alone -- and it
was confirmed visually too: both modes show the same vortex-street pattern,
spatially offset. `make_pod_video.py --combine` sums a chosen set of modes'
actual (recorded, non-extrapolated) time coefficients into one video;
combining modes 0+1 this way visibly travels, while either mode alone just
pulses in place -- direct confirmation of the quadrature reading. Together
the two modes capture 24-43% of all velocity-fluctuation variance,
dataset-dependent; 53-84 modes are needed to reach 90% of the variance
overall, consistently ~10-14% of each dataset's total mode count (487-828,
i.e. `min(state_dim, n_snapshots)`).

### Spectral Proper Orthogonal Decomposition (SPOD)

```bash
uv run scripts/analysis/check_spod.py --dataset re16k_t400_0
```

SPOD (Towne, Schmidt & Colonius, 2018) gets DMD's single-frequency modes
*and* POD's spatial orthogonality at once: split the snapshot sequence into
overlapping, Hann-windowed segments (same Welch convention as the point/
domain-mean spectra above), Fourier transform each segment in time, then --
independently at each frequency -- take the SVD of that frequency's
segment coefficients, exactly as POD does in the time domain. The result is
a set of modes *at each frequency*, ranked by the share of that frequency's
energy they capture; each oscillates at exactly one frequency, like a DMD
mode, but with no growth/decay to estimate, since SPOD assumes the process
is statistically stationary. Verified directly: summing the eigenvalues at
a frequency exactly equals summing that frequency's Welch power spectral
density over every channel -- SPOD only reorganizes each frequency's
energy into orthogonal modes, it doesn't add or remove any of it.

For `re16k_t400_0`, energy concentrates sharply in the leading mode around
T=28.6 (22.1% of the total), T=33.3 (18.0%) and T=40.0 (15.1%) --
`re16k_t400_4` similarly at T=28.6 (33.9%), T=33.3 (20.7%) and T=25.0
(9.8%), a visibly *more* concentrated spectrum, consistent with that
dataset's higher POD mode-0/1 energy above. In both cases the leading
mode's shape is the same vortex-street structure DMD and POD found -- yet
another confirmation (see "Where this leaves the split-sizing question"
below), and a third genuinely different method (per-frequency
eigendecomposition, neither a whole-window SVD nor a fitted linear
operator) arriving at it.

### Where this leaves the split-sizing question

Seven independent analyses (pooled autocorrelation, per-point, domain-mean,
and joint space-time spectra above, plus DMD, POD and SPOD) all agree on
the same ~24-40 step (dataset-dependent) oscillation as the
flow's best-characterized frequency -- confirmed independently, not just
detected once. It is a broad, quasi-periodic feature rather than a pure
tone, which is itself useful to know: it argues for buffer/margin choices
with headroom rather than a razor-precise period. Kinetic energy's slow
variation remains a genuine open question -- its own characterizing
timescale exceeds what a single train+val window can resolve, and even the
sizing below only holds a handful of effectively independent samples of it
(see the autocorrelation section's `N_eff` estimate).

**Decision (at the time, since superseded):** `val_steps`/`test_steps` in
`configs/analysis/split.yaml` were sized as a multiple of a conservative
reference period T=40 (the upper end of the confirmed range): val = 4T =
160 steps, test = 10T = 400 steps, with `buffer_steps: 20` (= T/2) applied
at both the train/val and val/test boundaries. Fixed absolute sizes rather
than fractions of each dataset's length, so every dataset would get a
comparably meaningful number of periods regardless of how long it happens
to be.

That was the right sizing *for a per-dataset time split* -- once it became
clear all 9 datasets are one regime rather than different ones, the split
moved to the dataset level instead (see "Train / val / test split" above,
and its "Earlier: a per-dataset time split" subsection for what this
decision looked like in practice before that change). T=40 and
`buffer_steps: 20` are still the right numbers, though: they're properties
of the flow itself, not of how the split is structured, and still explain
why `buffer_steps` shows up in the autocorrelation section below.

## Running the suite across all datasets

`configs/analysis/split.yaml` lists 8 of the 9 available datasets
(`re16k_t400_0` through `re16k_t400_10`, excluding the two known-bad ones
and `re16k_t400_5`, the model's held-out test set).

Each of the six `check_*.py` scripts accepts `--dataset NAME` (repeatable;
default: every configured dataset) and now writes a JSON summary of its
key numbers per dataset to `reports/summaries/<dataset>__<script>.json`
(gitignored, like the figures), in addition to its existing printed output
and plots — this is what makes cross-dataset comparison possible instead of
having to read 48 separate walls of text.

```bash
uv run scripts/analysis/run_all_checks.py
uv run scripts/analysis/run_all_checks.py --dataset re16k_t400_0 --dataset re16k_t400_1
uv run scripts/analysis/run_all_checks.py --script check_split.py --workers 4
```

`scripts/analysis/run_all_checks.py` runs every (script, dataset) pair — 48 by
default (6 scripts x 8 configured datasets) — and builds a comparison table
(printed and written to `reports/summaries/comparison.csv`) from the
resulting JSON summaries, with one row per dataset and a handful of
headline numbers (step count, divergence residual, enstrophy, the `u_x`
field decorrelation time and effective sample size, and the `u_x` mean's
half-to-half change from the stationarity check). The full detail stays in
the individual JSON files; the table is meant for a quick side-by-side
look, not the final word.

**Parallelization:** each (script, dataset) pair is independent, so the suite
dispatches them as separate `python check_*.py --dataset X` subprocesses via
`mhd_surrogate.utils.parallel`, on a backend chosen with `--backend`
(`sequential`, `processes` — the default, a thread pool whose threads block on
`subprocess.run` — or `ray`); see "Parallel backends" below for timings and
for the memory constraint that dominates this workload. `make_all_videos.py`
uses the same layer.

**First cross-dataset result:** the divergence residual clusters tightly
across all 8 configured datasets (0.4146-0.4251), and so does enstrophy
(26.6-27.5) — consistent with the `re16k_t400_0` numbers above being
representative of this simulation family rather than a fluke of one run.
The tooling working end-to-end on the new dataset-level split, with every
configured dataset read in full, is confirmed.

## Parallel backends

The preprocessing stages and the analysis suite share one parallel layer,
`mhd_surrogate.utils.parallel`, with three interchangeable backends selected
by `--backend` (and `--workers`):

| backend | what it does |
|---|---|
| `sequential` | a plain loop; the baseline |
| `processes` (default) | a process pool (`map_tasks`) or, for subprocess jobs, a thread pool whose threads block on `subprocess.run` (`run_parallel`) |
| `ray` | [Ray](https://www.ray.io) tasks on a local runtime |

Two shapes of work go through it: `run_parallel` for independent
`python script.py ...` jobs (the analysis suite, video rendering), and
`map_tasks` for a Python function over a list of arguments (conversion and
statistics: one task per dataset). Ray is the optional `ray` extra (it's in
the Docker image and in CI, but not required for the other backends).
`scripts/data/convert_to_zarr.py`, `compute_stats.py`,
`scripts/analysis/run_all_checks.py` and `make_all_videos.py` all take
`--backend`/`--workers`.

**Measured** with `scripts/analysis/benchmark_backends.py` on this machine
(12 cores; WSL was capped at 16 GB for the first measurements and later raised
to 24 GB, both noted below; wall seconds, median of n runs after an untimed warmup,
backends interleaved so the OS page cache doesn't favor one; it times the real
scripts as subprocesses, so Ray's startup is included):

| workload | sequential | processes | ray |
|---|---|---|---|
| `compute_stats` (7 datasets, n=3) | 19.8 | **7.8** (2.5x) | 11.7 (1.7x) |
| `convert_to_zarr` (9 files, n=2) | 49.8 | **12.1** (4.1x) | 16.9 (2.9x) |

| analysis suite, 48 jobs | WSL RAM | workers | wall (s) | vs sequential |
|---|---|---|---|---|
| sequential (n=1) | - | 1 | 451.0 | 1.0x |
| processes (n=2) | 16 GB | 5 | 163.2 | 2.8x |
| ray, memory-aware (n=2) | 16 GB | 5 | 166.6 | 2.7x |
| ray, memory-aware (n=2) | 16 GB | 12 | 148.1 | 3.0x |
| processes (n=3) | 24 GB | 12 | **120.4** | **3.7x** |
| ray, memory-aware (n=3) | 24 GB | 12 | 125.2 | 3.6x |

What this says, honestly:

- **For the preprocessing, Ray is slower than a process pool**, by its fixed
  ~4 s runtime startup, on a workload that finishes in 8-12 s. The work is
  7-9 uneven tasks on one machine, which is exactly what a process pool is for;
  the speedup is capped well below 12x by the task count and the size spread
  (1248 vs ~900 steps). Default stays `processes`.
- **For the suite, memory can be the constraint, and that's where Ray earns
  its place — but not speed.** The six checks differ hugely in peak memory
  (measured on the largest dataset): `check_autocorrelation` 2.2 GB, the other
  five 0.2-0.4 GB. Run 12 at a time script by script, 8 autocorrelation jobs at
  once need ~16 GB: the first attempt at this benchmark exhausted a 16 GB
  machine and had to be killed. A thread/process pool runs `--workers` jobs
  regardless, so under a tight memory cap it has to be sized by hand (5 workers
  in the 16 GB rows). Ray schedules against a declared per-job memory
  (`JOB_MEMORY_GB` in `run_all_checks.py`: 2.5 GB for autocorrelation, 0.5 GB
  for the rest), so it can be given all 12 workers and still not overcommit
  memory; under the 16 GB cap that made it the fastest configuration (148 s),
  and at equal workers the backends tie (166.6 vs 163.2 s).
- **With enough RAM the advantage disappears.** After raising WSL's limit to
  24 GB, 12 `processes` workers (with the dataset-major ordering below) finish
  in 120.4 s against Ray's 125.2 s: the process pool is ~4% faster, about the
  size of Ray's startup, and the run-to-run ranges don't overlap. Available
  memory bottomed out at ~7.5 GB of ~21.6 GB, so peak use was roughly 14 GB
  (inferred from that minimum): 12 workers would have been tight on 16 GB even
  with the better ordering. Ray's value here is that it adapts to whatever the
  memory limit is without hand-tuning `--workers`, not that it is faster.
- **Job order matters too.** The suite used to queue jobs script by script,
  putting all 8 autocorrelation jobs back to back; it now runs dataset by
  dataset, which spreads the heavy script out and is what makes the 5-worker
  `processes` run safe. Both fixes are needed for a backend without memory
  awareness.
- The n is small (2-3 runs, one machine); read these as orders of magnitude and
  the memory finding, not as precise ratios.

**Reproducibility is unaffected by the backend.** Per-dataset moments are merged
in a fixed order (`pool_moments`) regardless of which task finishes first, and
each conversion task writes only its own array, so every backend produces
byte-identical output: `dvc repro` through the parallel path left the hashes of
the zarr store and the stats file in `dvc.lock` unchanged, and a test asserts
the stats file is byte-identical across backends.

**Ray notes.**

- Ray's `uv run` integration packages the *whole working directory* (here ~20
  GB of data) as the workers' environment when the driver runs under `uv run`.
  `utils.parallel` disables it (`RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`, set before
  Ray is imported) — don't call `ray.init` yourself; use the layer's helper.
- Functions run by Ray or a process pool must be importable by the workers,
  which is why the conversion core lives in `src/mhd_surrogate/data/
  conversion.py` instead of the script, and why Ray refuses builtins like
  `pow` as tasks.
- In a container Ray's object store needs shared memory (`/dev/shm` is 64 MB by
  default); compose sets `shm_size: 1gb`. The `ray` extra adds ~0.3 GB to the
  image (2.0 -> 2.3 GB).
- Ray is not warranted at this data size for its own sake. It earns its keep
  here through memory-aware scheduling, and is the natural base for the
  later steps that do need a cluster scheduler (sweeps, KubeRay).

Reproduce:

```bash
uv run scripts/analysis/benchmark_backends.py --workload stats --repeats 3
uv run scripts/analysis/benchmark_backends.py --workload convert --repeats 2
# the suite is memory-heavy: pick --workers to fit your RAM for non-ray backends
uv run scripts/analysis/benchmark_backends.py --workload suite --backend processes --workers 5
```

## Video

```bash
uv run scripts/viz/make_video.py --dataset re16k_t400_0
uv run scripts/viz/make_video.py --dataset re16k_t400_0 --field speed --stride 2 --fps 30
```

Renders an animation of the flow over time to `reports/videos/` (gitignored,
like the figures). Fields: `vorticity` (default, same convention as
`check_vorticity.py`), `u_x`, `u_y`, or `speed` (`sqrt(u_x^2+u_y^2)`). The
color scale is fixed across the whole video, from the 1st/99th percentile of
a strided subsample of frames, so brightness is comparable frame to frame.
`--start/--end/--stride` select a sub-range or downsample in time; `--dx/--dy`
override the grid spacing (relevant only for `vorticity`).

Encoding uses the static ffmpeg binary bundled by the `imageio-ffmpeg`
package, so no system ffmpeg install is required. The default vorticity video
for `re16k_t400_0` (1248 frames, 24 fps) is ~52s and ~19.5 MB, taking about a
minute to render.

Watching the full vorticity video, the flow looks visually consistent across
the train/test boundary — no obvious change in structure, scale, or activity
level. This is a subjective, qualitative check, not a measurement; it
predates the train+val/test overhaul above (the quantitative check it used
to line up with, `check_split.py`'s train-vs-test comparison, no longer
exists), but a human eyeballing the rendered video isn't the automated
per-run data snooping that overhaul addresses, so watching the whole series
this way remains fine.

To render every dataset at once (same field/stride/fps for all of them):

```bash
uv run scripts/viz/make_all_videos.py
uv run scripts/viz/make_all_videos.py --field speed --stride 2 --fps 30
```

Same parallel dispatch as `run_all_checks.py` (see "Running the suite across
all datasets"): one `make_video.py --dataset X` subprocess per dataset,
concurrently. All 9 default (vorticity) videos took ~157s wall time against
~25 minutes of aggregate CPU time on this 12-core machine, ~133 MB total.

### Forecast videos

```bash
uv run scripts/viz/make_forecast_video.py --model-id <m-...>       # a logged MLflow model
uv run scripts/viz/make_forecast_video.py --checkpoint outputs/<date>/<time>/model
uv run scripts/viz/make_forecast_video.py --model-id <m-...> --max-steps 120 --log-to-mlflow
```

Shows how a model's forecast loses the flow, which the scores only
summarize: truth, prediction and their difference stacked full width (the
domain is 1151 x 127, so side by side would leave thin slivers), above the
scored RMSE-vs-lead-time curve with a cursor at the current frame. The
forecast is made exactly as it is scored (80 context steps, then frame k is
lead time k + 1), on the validation dataset by default (training datasets are
allowed, the test dataset is refused). Color scales are fixed for the whole
video: truth and prediction share the truth's 1st/99th percentile range, and
the difference panel is centered on zero with the same half-range, so an
error as large as the flow itself saturates it. `--field`, `--stride` and
`--fps` work as in `make_video.py`; `--max-steps` cuts a short clip (120
frames take ~25 s). With `--log-to-mlflow` the video is attached to the run
that logged the model, under `videos/`, next to its scores. For the mean field
the video makes its failure obvious: the prediction is a smooth band, so the
difference panel is the truth's whole vortex street.

## Training config (Hydra)

```bash
uv run scripts/training/train.py                     # model=mean_field
uv run scripts/training/train.py model=persistence
uv run --extra gpu scripts/training/train.py        # on the GPU (see Setup)
```

The exploration/analysis scripts above stay on plain argparse + PyYAML
(`configs/analysis/split.yaml`, `configs/analysis/grid.yaml`) — they are finished, standalone
tools and don't need config composition. New training/model code uses
[Hydra](https://hydra.cc) instead, since that will need config groups (model,
optimizer, trainer, ...) that compose, with CLI overrides and later multirun
sweeps. `configs/config.yaml` is the root config (`data`, `dataset`,
`normalization`, `model`, `evaluation`, `mlflow`, `seed` and `jax`); `configs/data/re16k.yaml` holds the zarr store path and the
dataset-level split (`train_datasets`, `val_dataset`, `test_dataset` -- see
"Train / val / test split" above), selected via the `data` default.

Model code is in [JAX](https://jax.readthedocs.io), on the GPU when the
`gpu` extra is installed (see Setup); each run logs which backend it used as
the `jax_backend` param.

JAX compiles each jitted function on its first call in every process, and
on the GPU that includes autotuning the matmuls -- for (Hankel) DMD's Gram
matrix alone ~45 s, paid again by every run. `configs/config.yaml`'s `jax`
key turns on JAX's persistent compilation cache
(`src/mhd_surrogate/utils/jax_cache.py`, also used by the forecast-video
script): compiled executables and XLA's autotuning results go to
`.jax_cache/` (gitignored, capped at 2 GB, least recently used evicted
first) and later runs load them instead of recompiling. With a warm cache
the canonical Hankel DMD fit takes 93 s instead of 141 s, the whole
`train.py` run 182 s instead of 224 s. Entries are keyed by the program,
its shapes, the device, the compile flags and the jax version, so an
upgrade or a different GPU misses rather than reuses a stale entry.

The cache is a performance knob, not a DVC dependency: a hit reuses exactly
the executable a cold compile built, and the canonical `train` stage
reproduces byte-identical outputs (checkpoint and `metrics.json`) with a
cold and a warm cache. So `jax.*` isn't in the stage's `params`, and
deleting `.jax_cache/` is always safe. JAX reads the cache config on its
first compile and ignores later changes, so entry points enable it before
anything is jitted; `jax.compilation_cache_dir=null` turns it off.

`configs/dataset/` holds sample-windowing config (`window`/`horizon`/
`stride`), selected via the `dataset` default.
`src/mhd_surrogate/data/dataset.py`'s `WindowedDataset` turns one region
into fixed-size samples: `window` consecutive time steps as input, the
following `horizon` steps as target, a new sample every `stride` steps. It
reads directly from the zarr array (no data is preloaded into memory);
`load_full_dataset` builds one over a dataset's entire recorded length, for
a dataset used wholesale (every train and val dataset now, since there's no
more internal region to bound reads by -- only whole-dataset exclusion,
`test_dataset`, matters). `windowed.yaml`'s `window=4, horizon=1, stride=1`
are placeholder defaults (short history, one-step-ahead prediction), not
tied to any model yet. Values are normalized on the fly when a `Normalizer`
is passed (see "Normalization statistics" below), otherwise returned as
stored, float32.

`scripts/training/train.py` fits the model selected by the `model` group on
the train datasets, saves its checkpoint, and scores it on the validation
dataset (see "Baselines" below for what it logs). It deliberately never opens
`test_dataset`, not even for a shape check. Each run's resolved config and logs are written to
`outputs/<date>/<time>/` (gitignored, like the other run artifacts).
`hydra.job.chdir` is set to `false` so the working directory stays the repo
root; without it, Hydra's default of chdir-ing into the run directory would
break every relative path used throughout this project (`data/raw/...`,
`configs/...`, etc.).

### Normalization statistics

`src/mhd_surrogate/data/normalization.py` computes the per-channel mean and
standard deviation used to normalize the fields. Decisions, and why:

- **Train datasets only.** The statistics are computed from `train_datasets`
  and nothing else; using val or test would leak held-out information into
  the model's inputs, which is exactly what the dataset-level split exists to
  prevent.
- **Per channel, global over time and space.** The flow is statistically
  homogeneous, so per-pixel statistics would only encode a spatial pattern
  that doesn't generalize, and per-timestep ones would remove the dynamics
  being predicted. The channels differ a lot (`u_x` has mean ~1, std ~0.8:
  the flow travels in x; `u_y` has mean ~0, std ~0.42), so each channel gets
  its own mean. Whether the std is per-channel or one shared scalar is a
  config option in the next step.
- **Streaming, float64.** Moments are accumulated `chunk_t` time steps at a
  time with the parallel (Chan et al.) update, so the 8.8 GB store is never
  loaded into memory, and pooled across datasets so the std includes
  between-dataset mean shifts. It's the population std (ddof=0).
- **Stored raw, normalized on the fly.** The zarr stays unnormalized; the
  transform is applied at load time from the stats artifact.

The statistics are written once and read back by other processes (training,
evaluation, inference), so `NormalizationStats` is a pydantic model rather
than a bare dict: loading a JSON file validates that channel names, mean and
std agree in length, std is finite and positive, and the source datasets are
non-empty and unique. `check_compatible(channel_names, train_datasets,
test_dataset)` then refuses stats whose channel order differs, that were
computed from a different set of datasets than the current training split,
or that include `test_dataset`. Hydra configs stay as they are; pydantic is
used only for this persisted artifact.

#### Computing and applying them

```bash
uv run scripts/data/compute_stats.py
```

Reads only `train_datasets`, prints per-dataset and pooled mean/std, and
writes `data/processed/normalization_stats.json` (gitignored with the rest of
`data/processed/`). The per-dataset rows are the check that the statistics are
a stable property of the flow rather than of one realization. On this data the
`u_x` mean is 1.000 and the `u_y` mean 0.000 in every training dataset, `u_x`
std is 0.81-0.82 and `u_y` std 0.41-0.43, and the pooled values are
(1.0001, 0.8151) and (0.0001, 0.4206).

`configs/normalization/` is a Hydra group with one option per std mode,
selected with e.g. `normalization=shared`:

- `per_channel` (default): each channel divided by its own std, which keeps
  `u_y` (about half the std of `u_x`) from being under-weighted in an MSE loss.
- `shared`: all channels divided by one scalar, the RMS of the per-channel
  stds. The channels' relative amplitudes are preserved, so the loss stays
  proportional to physical kinetic-energy error.

Means are per channel in both modes. `Normalizer.from_stats(stats, std_mode)`
builds the `(x - mean) / std` transform (and its `inverse`, for turning
predictions back into physical units); `WindowedDataset`/`load_full_dataset`
take it as an optional `normalizer`. `train.py` loads the stats file, runs
`check_compatible` against the configured channels, train datasets and test
dataset, and fails with a pointer to `compute_stats.py` if the file is
missing; it uses the effective std to scale the validation RMSE.

### Experiment tracking (MLflow)

```bash
uv run scripts/training/train.py
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db  # view runs at http://127.0.0.1:5000
# or, with the Docker stack running (see "Docker"): train.py mlflow=server
```

Each run also records the data version it was started against (hashes from
`dvc.lock` as params, `dvc.lock` and the stats file as artifacts); see "Data
versioning (DVC)".

`configs/mlflow/local.yaml` (selected via the `mlflow` default) points at a
local, self-hosted MLflow backend — no external account needed. It uses a
SQLite database (`mlruns.db`, gitignored) rather than the classic
`./mlruns` file store: MLflow has deprecated the file store in favor of a
database backend, so SQLite is the current recommended approach.
`src/mhd_surrogate/training/mlflow_utils.py`'s `flatten_for_mlflow` turns the
resolved (nested) Hydra config into the flat key-value pairs
`mlflow.log_params` expects (e.g. `data.dataset`, `dataset.window`). Runs
are named after their model (`mean_field`, `persistence`, ...) instead of
MLflow's random names; the scores are described under "Baselines" below.

**Logged models.** The fitted model is logged as an MLflow *model*, not just
as files: `src/mhd_surrogate/training/mlflow_model.py` wraps the checkpoint in
a `pyfunc` model named after the model, so it shows in the runs table's
Models column and gets its own page in the UI, with the model config as its
params and every `val.*`/`train.*` score linked to it (the Models tab compares
models directly). The logged model is self-contained -- the checkpoint as an
artifact, this package's source as `code_paths`, pinned `numpy`/`jax`/`mlflow`
requirements, and a signature (a stack of `(C, Nx, Ny)` float32 frames in and
out, `n_steps` as a parameter) -- so it loads anywhere with

```python
model = mlflow.pyfunc.load_model(f"models:/{model_id}")
forecast = model.predict(context, params={"n_steps": 837})  # last `window` frames are used
```

and `mlflow models serve` can serve it, which is what the serving step will
build on. Models are not registered in the Model Registry automatically:
registering every experiment would bury the meaningful versions, so only the
shortlist and the test-set finalists (see "How validation and test are used")
get registered, deliberately. No input example is logged: one real frame is
~1 MB as JSON, and the signature already pins the shapes.

`src/mhd_surrogate/training/tracking.py`'s `tracked_run` owns the run
lifecycle so entry points don't repeat it: it selects the experiment, starts
the run, logs the resolved config (flattened params plus a `config.json`
artifact) and, whether the run finishes or crashes, uploads the per-run log
file as an artifact, so a run that died overnight keeps its post-mortem
record next to its metrics. A training loop that sees a NaN/blown-up loss
raises `DivergenceError`: the traceback is logged, the run is tagged
`diverged=true` (filterable in the UI) and MLflow marks it FAILED.

What goes where: *metrics* (loss, LR, grad norm, throughput) go to MLflow
via `mlflow.log_metrics(..., step=...)`, never into log files; the log file
holds *events* ("checkpoint saved", "resumed from ...", warnings,
tracebacks); params, config and checkpoints are MLflow params/artifacts.
Logging for training runs is Hydra's own `job_logging`, selected in
`config.yaml` with a project config (`configs/hydra/job_logging/project.yaml`)
instead of Hydra's default: the root logger stays at `WARNING` so
third-party libraries stay quiet, and this project's loggers get `DEBUG`.
The console shows `INFO` and up (`hydra.job_logging.handlers.console.level=DEBUG`
changes that); `train.log`, written into the run's `outputs/<date>/<time>/`
directory, always records `DEBUG` and up. `tracked_run` uploads it to MLflow.

**Dependency note:** `hydra-core` is pinned to the `1.4.0.dev9` pre-release.
The latest stable release (1.3.7) is broken on Python 3.14 (this project's
Python version) — an upstream bug
([facebookresearch/hydra#3121](https://github.com/facebookresearch/hydra/issues/3121)):
its CLI parser fails under Python 3.14's stricter `argparse` validation
before any user code runs. `1.4.0.dev9` fixes it and was verified to resolve
config and run cleanly via a plain `uv sync` (no `--prerelease` flag needed,
since the version is pinned exactly). Swap for the stable 1.4.0 release once
it ships.

## Forecast evaluation

A surrogate is judged by forecasting a held-out dataset's whole series (917
steps for the validation dataset), so that every model -- one-step
autoregressive, whole-series, or a baseline that ignores its input -- is
scored on the same targets. The code is in `src/mhd_surrogate/evaluation/`.

**Protocol** (`protocol.py`). The first `context_steps = 80` steps of the
dataset (`configs/data/re16k.yaml`) are context only and are never scored; the
model is scored on steps 80 onward (837 for validation). Eighty is about twice
the longest characteristic oscillation (~40 steps) found in the EDA, so a model
can be given a full period or two of input, and "does more input help?" is a
comparison on identical targets. A model declares how many of the most recent
context steps it reads (`window`, 0 for none); `forecast` hands it exactly
those, and rejects a window longer than the context since it would see scored
targets. The same protocol applies to the test dataset, but that is read only
once, for the final evaluation.

**Metrics** come in two groups, because the flow is chaotic: pointwise error
saturates after some lead time for *any* model, so a low final-step error is
neither achievable nor the only thing worth measuring.

- *Short horizon* (`metrics.py`): RMSE per lead time, optionally scaled per
  channel by the training std, and the skill horizon (how many steps stay under
  an error threshold; a NaN error counts as over it, so a forecast that blows
  up loses its skill there instead of scoring the whole length).
- *Long horizon* (`diagnostics.py`): does the forecast stay on the attractor?
  Relative error of the time-averaged kinetic energy and enstrophy, the ratio
  of RMS divergence to the true field's (the DNS field is only approximately
  divergence-free on this grid, so the target's own value is the reference, not
  zero), and the log-spectral distance of the x and y spectra. A model that
  regresses toward the mean keeps a modest pointwise error but loses the small
  scales, which these catch. They are computed in chunks, so a long series is
  never held in float64 at once.
- *Stability* (`stability.py`): how long does the forecast stay bounded?
  The forecast is cut into 100-step blocks, and a block fails when its mean
  kinetic energy or enstrophy exceeds 2x the truth's largest block mean
  (`evaluation.stability` in `configs/evaluation/default.yaml`).
  `stable_steps` is where the first failing block starts (all 837 steps if
  none fails), and `energy_peak_ratio` / `enstrophy_peak_ratio` are the
  forecast's largest block mean over the truth's, i.e. how far it strayed.
  The truth's own block means vary by ~5% (energy) and ~18% (enstrophy) on
  the validation dataset, so a 2x excess is a blow-up, not noise. The check
  is one-sided: a forecast that damps toward a smooth state is bounded, just
  wrong, and the guardrails below judge that. It costs ~3 s per forecast,
  cheap enough for every validation during training.
- *Long horizon, in time* (`diagnostics.temporal_scores`): the spatial scores
  look at each snapshot, so they can't see a forecast that has the right
  structure at every instant but the wrong dynamics. The EDA's most robust
  feature is the coherent ~24-40 step oscillation, a single sharp peak in the
  Welch spectrum of the *domain-averaged* `u_y`, so the temporal scores use the
  domain-averaged velocity: log-spectral distance per channel, the relative
  error of `u_y`'s dominant period (peak refined between the coarse bins by a
  parabola) and the ratio of predicted to true power at that peak (below 1 when
  the oscillation is damped). Welch uses the EDA's 200-step segments, so a
  held-out series gives ~7 of them: noisy, coarse estimates. A forecast with no
  oscillation at all (e.g. persistence, or the mean field) has an undefined
  period (`nan`) and a peak ratio of 0. SPOD would give mode shapes rather than
  a score (it needs mode matching between prediction and truth and an
  eigendecomposition per frequency), so it stays a plotting diagnostic.

**Long rollouts** (`scripts/evaluation/check_rollout_stability.py`). The
stability score sees only the scored forecast, 837 steps of the validation
dataset, but a surrogate is meant to be rolled out for as long as one likes.
The check rolls a model out for `--steps` steps (1500 by default) from the
dataset's context, one block at a time (only one block in memory), and
prints each block's mean energy and enstrophy and their ratio to the
truth's largest block mean -- the selection rule's reference, so its first
blocks reproduce `stable_steps`:

```bash
uv run scripts/evaluation/check_rollout_stability.py --checkpoint models/hankel_dmd/model
uv run --extra gpu scripts/evaluation/check_rollout_stability.py --model-id m-... \
    --steps 5000 --block-steps 500
```

For an autoregressive network, predicting block by block is the same
rollout as one long `predict` up to float rounding at the block boundaries;
(Hankel) DMD restarts from the projection of its own last frames at every
block. The test dataset is refused, like in the forecast videos.

**Ensembles** (`protocol.py`, `ensemble.py`). The deterministic models
so far have one answer per context; a generative surrogate (flow matching,
diffusion) samples one, and its forecast is a distribution. The contract is
ready for it without changing anything for the others: a model with
`stochastic = True` takes `predict(context, n_steps, seed=...)` and returns
its sample for that seed, an ensemble is the samples for seeds 0, 1, ...,
`evaluation.ensemble_size` (8), and a deterministic model is a one-member
ensemble. Every score above is the seed-0 member's (one trajectory, as
`predict` gives it; the ensemble *mean* would be smooth and fail the
physics), and the ensemble adds:

- *CRPS* per lead time (`crps_mean`, `crps_lead_<n>`): E|X - y| - E|X -
  X'|/2 over members X, X' and truth y, per pixel. A proper score -- it is
  minimized by forecasting the true distribution -- so unlike the RMSE it
  doesn't reward collapsing toward the mean. For one member it *is* the
  absolute error, so deterministic and generative models share a score.
- *Spread vs error* (`spread_skill_lead_<n>`, more than one member):
  sqrt((M + 1) / M) times the ensemble's spread over the RMSE of its mean,
  ~1 for a calibrated ensemble of M members (Fortin et al. 2014), below 1
  for an overconfident one.

A model can also sample several members per call (`predict_members`;
`evaluation.member_batch`, 1 by default), for a GPU with room to spare --
a bigger cloud GPU, or a small model that leaves this one idle. Member k of
a batch must be the sample for its own seed whatever else is in the batch
(randomness drawn per seed, not split from a shared key), so batching only
changes throughput -- up to float rounding, which GPU kernels can vary with
the batch shape, so the batch size is part of what reproduces an
ensemble's scores. A model without it gets one call per member.

Members are streamed one at a time (an 837-step member is ~1 GB): the
pairwise terms use consecutive members only, each an independent pair, so
they're unbiased like all M (M - 1) / 2 pairs, with some more variance.
MLflow gets the per-lead `crps` curve (and `spread`, `ensemble_mean_rmse`
for an ensemble), and the logged pyfunc model takes `seed` as a parameter
next to `n_steps`, so a server can sample an ensemble one call per seed.
How the ensemble scores enter the selection rule is decided when the first
generative model exists; until then the rule is unchanged.

**How big a score is "perfect"?** Scoring a different realization of the same
flow (train datasets 0 and 10 against the validation targets) -- right
dynamics, wrong phase -- bounds what the metrics can resolve: spatial spectrum
distances of 0.01-0.03, energy and enstrophy errors under 3%, divergence ratio
~1, but temporal spectrum distances of ~0.2, a `u_y` period error of 0.15-0.19
and a peak power ratio of 0.95-1.14 (the true validation peak is at period
25.6, with 38% of the `u_y` power). A model's temporal scores are meaningfully
off only when they clearly exceed those values; a period error under ~0.2 is
indistinguishable from simply being a different realization.

## Baselines

Every model implements one interface (`src/mhd_surrogate/models/base.py`):
`fit` on the train datasets, `predict(context, n_steps)` under the forecast
protocol, and `save`/`load` of a checkpoint directory (`model.json` naming the
model, plus its arrays), so `models/registry.py` can build a model from its
`model` config group or load any checkpoint by name. Models take and return
raw velocity fields; one that wants normalized inputs normalizes internally,
so every model is scored in the same units.

The two baselines bracket what a real model has to do:

- **Persistence** (`model=persistence`) repeats the last context frame. It is
  the best simple forecast one step ahead and useless once the flow has
  decorrelated.
- **Mean field** (`model=mean_field`, the default) predicts the per-pixel
  temporal mean of all 7,115 training steps at every lead time --
  "climatology". Its parameters are the mean field, so it is fitted, saved
  and logged like any other model, not computed as a data-pipeline output. The
  fit streams the train datasets in chunks and sums them on JAX's default
  device (the GPU with the `gpu` extra): 13 s, almost all of it zarr reads.

`train.py` logs, per run: the checkpoint as a logged MLflow model (see
"Experiment tracking (MLflow)"), `fit_seconds`, every scalar score as a
`val.*` metric (and `train.<dataset>.*`, below), and the RMSE curve as the
`val.rmse` history with step = lead time (one batched request rather than
837). Scores that are undefined for a model (the oscillation period of a
forecast with no oscillation) are left out and named in the log. Each scored
dataset also gets two timings: `<prefix>.eval_seconds`, the whole evaluation
(reading the targets, forecasting, scoring), and
`<prefix>.forecast_seconds_per_frame`, the model's `predict` call alone
divided by the number of forecast frames -- the surrogate's cost per step,
comparable across datasets of different lengths and against the solver's.
The timings are MLflow metrics only, not part of the scores written to the
DVC stage's `metrics.json`: wall-clock times differ from run to run, and that
file must reproduce exactly.

Validation results (`re16k_t400_6`, 837 scored steps; RMSE in units of the
training std):

| | persistence | mean field | different realization* |
|---|---|---|---|
| RMSE at lead 1 / 10 / 40 | 0.51 / 1.40 / 1.23 | 0.91 / 0.90 / 0.89 | -- |
| RMSE, mean over all leads | 1.24 | 0.96 | -- |
| energy / enstrophy error | 2% / 11% | 29% / 76% | <1% / <3% |
| divergence ratio | 1.14 | 0.16 | ~1 |
| spatial spectrum distance (x / y) | 0.08 / 0.05 | 3.9 / 2.6 | 0.02-0.03 / 0.01-0.03 |
| `u_y` period error / peak power ratio | undefined / 0 | undefined / 0 | 0.15-0.19 / 0.95-1.14 |

\* train datasets 0 and 10 scored against the validation targets, from
"Forecast evaluation" above: the right dynamics, the wrong phase.

**Sanity check on training data.** The same evaluation also runs on the
training datasets listed in `evaluation.train_datasets` (`re16k_t400_0` by
default; ~35-50 s each), logged as `train.<dataset>.*`. It answers what
validation alone can't: whether a model can fit its own training data at all
(if not, it's a bug or too little capacity, and tuning is pointless) and how
large the train/validation gap is (memorizing trajectories instead of learning
dynamics). It is a diagnostic, not a basis for decisions (see "How
validation and test are used"). The flow is chaotic, so even on its own
training trajectories a rollout decorrelates after a few dozen steps: the gap
shows at short lead times and in the physics scores, not in late-lead RMSE.
For the baselines it is a plumbing check, and comes out as expected:

| | persistence, train / val | mean field, train / val |
|---|---|---|
| RMSE at lead 1 | 0.47 / 0.51 | 0.96 / 0.91 |
| RMSE, mean over all leads | 1.20 / 1.24 | 0.95 / 0.96 |
| energy error | 1% / 2% | 29% / 29% |
| spatial spectrum distance (x) | 0.09 / 0.08 | 3.9 / 3.9 |

The mean field, which averages over this dataset among the 7, is only
marginally better on it, and persistence, which has nothing to fit, scores
the same on both: neither has anything to memorize, so neither has a gap.

Neither baseline has any skill at the 0.5 threshold (persistence is already
at 0.51 one step ahead). Persistence beats the mean field only at lead 1 and
keeps realistic snapshot statistics but no dynamics; the mean field has the
lower pointwise error from lead ~2 on but no small scales, almost no
divergence and no oscillation. A useful surrogate has to beat persistence
early and the mean field late while scoring near the "different realization"
column on the diagnostics.

## DMD

```bash
uv run --extra gpu scripts/training/train.py model=dmd             # rank 750
uv run --extra gpu scripts/training/train.py model=dmd model.rank=100
```

Dynamic Mode Decomposition (`src/mhd_surrogate/models/dmd.py`) fits the best
linear operator `A` (least squares, `z_{t+1} ~= A z_t`) to the training
snapshots and forecasts by applying it repeatedly -- the simplest model with
dynamics, between the baselines (none) and a nonlinear network. The EDA
already showed DMD capturing the flow's coherent ~24-40 step oscillation; as a
forecaster it can carry that forward, but not the turbulence.

- **State:** each frame as a fluctuation about the pooled training mean,
  scaled per channel by the fluctuation's RMS so `u_x` and `u_y` weigh the
  same in the SVD, channels and space flattened (~292k entries). The mean and
  the scale are fitted by the model itself.
- **Fit:** projected exact DMD over the 7 training datasets, with snapshot
  pairs formed within each dataset only (7,108 pairs, never across a dataset
  boundary). The snapshot matrix is too large to factorize directly, so it
  uses the method of snapshots: the 7,115 x 7,115 Gram matrix of all frames
  gives the POD basis (truncated to `rank`) and the reduced operator. The
  Gram matrix is computed with JAX on the GPU in column blocks of equal width
  (the last zero-padded), so the matmul is compiled and autotuned once
  (~45 s), not once per block shape. The Gram matrix squares the singular
  values, so with float32 frames they're resolved only down to ~3e-4 of the
  largest; values below 1e-3 are treated as zero.
- **Forecast:** the last context frame (window 1) is projected onto the basis,
  evolved with the reduced operator's eigendecomposition, and mapped back.
  Eigenvalues outside the unit circle would blow the forecast up over 837
  steps, so `stabilize` moves them onto it (none needed it at rank 100).
- **Cost:** the fit holds all training frames in RAM (8.3 GB, ~14 GB peak)
  and takes ~105 s, mostly the one-off compile and the eigendecomposition of
  the Gram matrix; it logs `fit.rank`, `fit.explained_variance` (0.90 at rank
  100), `fit.unstable_modes` and `fit.n_pairs`.

Validation results at rank 100, the first fit (RMSE in units of the training
std; the default is now rank 750, selected by the sweep below):

| | persistence | mean field | **DMD** | DMD on train |
|---|---|---|---|---|
| RMSE at lead 1 / 10 / 40 | 0.51 / 1.40 / 1.23 | 0.91 / 0.90 / 0.89 | **0.37 / 0.65 / 0.80** | 0.29 / 0.52 / 0.69 |
| RMSE, mean over all leads | 1.24 | 0.96 | **0.83** | 0.82 |
| skill horizon (<= 0.5) | 0 | 0 | **4** | 9 |
| energy / enstrophy error | 2% / 11% | 29% / 76% | 26% / 73% | 26% / 72% |
| spatial spectrum distance (x) | 0.08 | 3.9 | 2.9 | 3.0 |
| `u_y` period error / peak power ratio | undefined / 0 | undefined / 0 | **0.20 / 0.22** | 0.04 / 0.45 |

DMD is the first model with skill and beats both baselines in pointwise error
at every lead, and the first whose forecast oscillates: its `u_y` period is
off by 0.20, at the edge of what a different realization of the flow scores
(0.15-0.19), but at 22% of the true power the oscillation is strongly damped.
Being linear and rank-truncated, it predicts a smoothed version of the
large-scale vortex street and none of the small scales (spectrum distance 2.9,
energy 26% low) -- the forecast video shows exactly that. The training-set
check shows a real gap at short leads and in the oscillation (period error
0.04 on train vs 0.20 on validation): the fitted modes describe the training
realizations' oscillation better than an unseen one's.

### Rank sweep

```bash
uv run --extra gpu scripts/training/train.py -m model=dmd model.rank=25,50,100,150,200,300,500
uv run --extra gpu scripts/training/train.py -m model=dmd model.rank=750,1000
```

A Hydra multirun (`-m`): one training run per rank, in one process, each
logged to MLflow as its own run named after its overrides (`dmd
model.rank=50`) and tagged with the sweep (`tags.sweep`, the timestamp of its
`outputs/multirun/<date>/<time>/` directory, plus `tags.sweep_job`), so the
UI's filter `tags.sweep = '...'` shows one sweep side by side. The jobs share
the process, so the Gram matmul is compiled once: the first fit takes ~95 s,
the rest ~55 s, and the 7 ranks ran in ~15 minutes; the trend was still
improving at 500, so 750 and 1000 followed in a second sweep. Validation
results:

| rank | explained variance | RMSE lead 1 / 10 / 40 | RMSE mean | skill horizon | spectrum distance (x) | `u_y` period error / peak ratio |
|---|---|---|---|---|---|---|
| 25 | 0.77 | 0.53 / 0.68 / 0.73 | 0.80 | 0 | 3.09 | 0.16 / 0.30 |
| 50 | 0.85 | 0.45 / 0.65 / 0.82 | 0.84 | 3 | 2.95 | 0.19 / 0.35 |
| 100 | 0.90 | 0.37 / 0.65 / 0.80 | 0.83 | 4 | 2.92 | 0.20 / 0.22 |
| 150 | 0.93 | 0.32 / 0.68 / 0.80 | 0.84 | 4 | 2.85 | 0.18 / 0.27 |
| 200 | 0.94 | 0.30 / 0.68 / 0.83 | 0.85 | 4 | 2.84 | 0.19 / 0.24 |
| 300 | 0.96 | 0.28 / 0.64 / 0.78 | 0.83 | 4 | 2.76 | 0.18 / 0.21 |
| 500 | 0.97 | 0.26 / 0.64 / 0.71 | 0.82 | 5 | 2.71 | 0.17 / 0.17 |
| **750** | 0.98 | 0.26 / 0.62 / 0.68 | 0.81 | 5 | 2.66 | 0.14 / 0.16 |
| 1000 | 0.99 | 0.26 / 0.63 / 0.65 | 0.80 | 4 | 2.61 | 0.14 / 0.18 |

What the sweep shows:

- **Short leads improve with rank up to ~500, then saturate** (lead 1: 0.53
  -> 0.26, flat from 500 on), and the small-scale content keeps improving
  slowly (spectrum distance 3.09 -> 2.61) -- more modes start the forecast
  closer to the true state. **Past ~500 overfitting begins:** on the training
  dataset lead-1 RMSE keeps falling (0.22 -> 0.20 -> 0.19 at 500/750/1000)
  while validation stays at 0.26, and rank 1000 loses a skill step. The extra
  modes fit the training realizations, not the dynamics.
- **The mean RMSE over all leads barely moves (0.80-0.85), and the lowest
  rank scores best.** After a few dozen steps every forecast has decorrelated,
  and a smoother, lower-rank forecast sits closer to the mean, which is what
  pointwise error rewards there. Selecting by mean RMSE would pick the model
  with the least structure -- the reason the evaluation has a short-horizon
  and a long-horizon half.
- **The oscillation scores don't separate the ranks.** The period error
  (0.16-0.20) is within what a different realization scores (0.15-0.19), and
  the peak power ratio (0.17-0.35) moves without a trend: on a single
  validation realization those differences are noise.
- Energy error (~26%) doesn't depend on rank: the missing energy is in the
  turbulence no linear model carries forward, not in the truncated modes.

**Selected: rank 750**, by the ranking rule in "How candidates are ranked":
500 and 750 tie on skill horizon (5), 750 wins the tie-break on RMSE at lead
10 (0.62 vs 0.64), and its physics scores stay within the guardrails of the
previous default, rank 100 (energy +0.002, enstrophy -0.001, spectra better).
The margin over 500 is small enough to be realization noise; the rule, not
the margin, decides. The cost is size: the rank-750 checkpoint and logged
model are ~880 MB (rank 500: ~590 MB, rank 100: ~117 MB).

## Hankel DMD

```bash
uv run --extra gpu scripts/training/train.py model=hankel_dmd    # 4 delays, rank 1000
uv run --extra gpu scripts/training/train.py model=hankel_dmd model.delays=32 model.rank=1500
```

Time-delay (Hankel) DMD (`src/mhd_surrogate/models/hankel_dmd.py`) is DMD on
the last `delays` frames instead of one: the model's `window` is `delays`.
Plain DMD has to read where the flow is going from a single frame, and a
snapshot of a standing pattern doesn't say whether its amplitude is rising or
falling; a few frames of history do. It answers the question the protocol's
80-step context was sized for -- does more context help? -- so `delays` can
go up to 80.

- **POD first, then delays.** Delay-embedding the raw frames would make the
  state `delays` x ~292k entries and its POD basis ~880 MB *per delay* at rank
  750 (~70 GB at 80 delays, in RAM and as a checkpoint). Instead each frame is
  reduced to its coefficients in plain DMD's POD basis (`spatial_rank` modes,
  750 by default, plain DMD's selected rank), and the delays are taken of
  those: a state of at most 80 x 750 = 60k entries. Only the one spatial
  basis is stored at full size; the delay basis is `delays` x `spatial_rank`
  x `rank` (12 MB at the defaults, ~360 MB at 80 delays and rank 1500). The
  price is that the delays see each frame's 750-mode projection, not the
  frame -- 98% of the variance; the rest is small-scale content no linear
  model carries forward anyway. This is the usual way to apply Hankel DMD to
  high-dimensional fields.
- **Fit, from the same Gram matrix.** Everything comes from the one Gram
  matrix plain DMD computes, in one pass over the frames: the spatial basis,
  every frame's coefficients (`G[:, before] V S^-1`, no second pass), and the
  delay states' Gram matrix as a sum of shifted blocks of the coefficients'
  inner products -- the (pairs x 60k) delay matrix is never formed. Delay
  states never reach across a dataset boundary, so each dataset loses its
  first `delays - 1` frames as starting points (at 80 delays, 553 of the
  7,108 pairs). Then projected exact DMD on the delay states, truncated to
  `rank`, and `stabilize` as in DMD. It logs `fit.spatial_explained_variance`
  (of the frames, kept by the spatial basis) and
  `fit.delay_explained_variance` (of the delay states, kept by the delay
  basis), besides `fit.rank`, `fit.delays`, `fit.unstable_modes` and
  `fit.n_pairs`.
- **Two exact checks.** With `delays=1` it is plain DMD at
  `min(rank, spatial_rank)` exactly (the coefficients of the "before" frames
  are already orthogonal, so the delay POD changes nothing); a test checks
  that, and on the real data `delays=1, rank=750` reproduces plain DMD rank
  750's validation scores. A second test is a standing wave -- one pattern
  whose amplitude oscillates -- which no one-frame linear model can forecast
  and two delays forecast exactly.
- **Two ranks.** `spatial_rank` bounds what any frame can represent; `rank`
  is the dynamics' dimension, now over `delays` x `spatial_rank` features fit
  from ~7k pairs, so it has to be truncated (plain DMD already overfit past
  rank ~500).

### Delay sweep

```bash
uv run --extra gpu scripts/training/train.py -m model=hankel_dmd \
    model.delays=2,4,8,16,32,80 model.rank=250,500,750,1000,1500
```

The two axes interact -- more delays need more modes to hold the same
information -- so the sweep is a grid, not one axis at a time: 30 runs in
one Hydra multirun (~3.4 min each, ~1 h 40 min; the fit is ~2 min, the rest
scoring), `spatial_rank` fixed at 750. `delays=1` isn't in the grid: it is
plain DMD, checked separately (above). Validation results, with the
incumbent, plain DMD rank 750, for reference:

| delays | rank | RMSE lead 1 / 10 / 40 | RMSE mean | skill horizon | skill on train | energy / enstrophy error | spectrum distance (x) | `u_y` period error / peak ratio |
|---|---|---|---|---|---|---|---|---|
| *1 (plain DMD)* | *750* | *0.26 / 0.619 / 0.68* | *0.81* | *5* | *11* | *0.27 / 0.73* | *2.66* | *0.14 / 0.16* |
| 2 | 250 | 0.31 / 0.679 / 0.80 | 0.85 | 4 | 12 | 0.26 / 0.73 | 2.79 | 0.20 / 0.21 |
| 2 | 500 | 0.29 / 0.604 / 0.66 | 0.81 | 4 | 11 | 0.26 / 0.73 | 2.70 | 0.14 / 0.22 |
| 2 | 750 | 0.27 / 0.579 / 0.62 | 0.79 | 6 | 12 | 0.26 / 0.73 | 2.66 | 0.11 / 0.21 |
| 2 | 1000 | 0.25 / 0.574 / 0.61 | 0.78 | 7 | 13 | 0.26 / 0.72 | 2.63 | 0.12 / 0.22 |
| 2 | 1500 | 0.23 / 0.612 / 0.58 | 0.78 | 7 | 14 | 0.26 / 0.72 | 2.66 | 0.07 / 0.12 |
| 4 | 250 | 0.36 / 0.681 / 0.80 | 0.85 | 3 | 13 | 0.27 / 0.74 | 2.80 | 0.19 / 0.14 |
| 4 | 500 | 0.32 / 0.575 / 0.65 | 0.79 | 5 | 10 | 0.26 / 0.73 | 2.66 | 0.10 / 0.13 |
| 4 | 750 | 0.30 / 0.600 / 0.62 | 0.79 | 6 | 13 | 0.26 / 0.73 | 2.63 | 0.09 / 0.15 |
| **4** | **1000** | 0.28 / 0.571 / 0.58 | 0.78 | **7** | 13 | 0.26 / 0.73 | 2.62 | 0.04 / 0.17 |
| 4 | 1500 | 0.25 / 0.618 / 0.66 | 0.80 | 6 | 15 | 0.26 / 0.73 | 2.63 | 0.01 / 0.09 |
| 8 | 250 | 0.39 / 0.674 / 0.66 | 0.80 | 4 | 10 | 0.26 / 0.73 | 2.73 | 0.14 / 0.12 |
| 8 | 500 | 0.35 / 0.624 / 0.61 | 0.78 | 4 | 10 | 0.26 / 0.72 | 2.67 | 0.10 / 0.17 |
| 8 | 750 | 0.34 / 0.627 / 0.60 | 0.76 | 4 | 12 | 0.23 / 0.69 | 2.59 | 0.09 / 0.21 |
| 8 | 1000 | 0.32 / 0.629 / 0.58 | 0.75 | 5 | 14 | 0.24 / 0.70 | 2.55 | 0.10 / 0.28 |
| 8 | 1500 | 0.29 / 0.627 / 0.59 | 0.75 | 6 | 24 | 0.25 / 0.71 | 2.54 | 0.10 / 0.25 |
| 16 | 250 | 0.43 / 0.672 / 0.71 | 0.77 | 2 | 8 | 0.25 / 0.71 | 2.61 | 0.19 / 0.12 |
| 16 | 500 | 0.39 / 0.656 / 0.67 | 0.76 | 4 | 12 | 0.24 / 0.69 | 2.60 | 0.13 / 0.39 |
| 16 | 750 | 0.36 / 0.628 / 0.55 | 0.73 | 3 | 14 | 0.23 / 0.68 | 2.51 | 0.06 / 0.44 |
| 16 | 1000 | 0.35 / 0.628 / 0.59 | 0.74 | 4 | 14 | 0.24 / 0.69 | 2.47 | 0.08 / 0.25 |
| 16 | 1500 | 0.34 / 0.626 / 0.62 | 0.75 | 4 | 15 | 0.23 / 0.68 | 2.38 | 0.07 / 0.26 |
| 32 | 250 | 0.47 / 0.666 / 0.76 | 0.77 | 2 | 12 | 0.21 / 0.65 | 2.45 | 0.04 / 0.21 |
| 32 | 500 | 0.44 / 0.618 / 0.62 | 0.75 | 3 | 18 | 0.21 / 0.65 | 2.40 | 0.11 / 0.36 |
| 32 | 750 | 0.42 / 0.648 / 0.58 | 0.77 | 3 | 15 | 0.23 / 0.67 | 2.35 | 0.16 / 0.39 |
| 32 | 1000 | 0.40 / 0.637 / 0.72 | 0.79 | 3 | 14 | 0.25 / 0.70 | 2.33 | 0.19 / 0.22 |
| 32 | 1500 | 0.38 / 0.634 / 0.54 | 0.74 | 3 | 32 | 0.22 / 0.65 | 2.21 | 0.13 / 0.57 |
| 80 | 250 | 0.53 / 0.625 / 0.73 | 0.92 | 0 | 14 | 0.23 / 0.66 | 2.31 | 0.21 / 0.34 |
| 80 | 500 | 0.49 / 0.632 / 0.77 | 0.92 | 1 | 15 | 0.21 / 0.64 | 2.14 | 0.22 / 0.28 |
| 80 | 750 | 0.47 / 0.653 / 0.70 | 0.85 | 2 | 25 | 0.21 / 0.62 | 2.03 | 0.20 / 0.44 |
| 80 | 1000 | 0.47 / 0.609 / 0.61 | 0.80 | 2 | 46 | 0.17 / 0.59 | 1.97 | 0.04 / 0.68 |
| 80 | 1500 | 0.45 / 0.627 / 0.70 | 0.82 | 2 | 55 | 0.21 / 0.62 | 1.87 | 0.19 / 0.28 |

What the sweep shows:

- **A little context helps.** Two or four delays lift the skill horizon from 5
  to 7 and RMSE at lead 10 from 0.62 to 0.57 -- the first real step beyond
  plain DMD -- provided the rank grows with them (at rank 250 and 500 they
  don't beat it). The history tells the linear operator which way the
  coherent oscillation is moving, which a single frame doesn't.
- **More context hurts the short horizon.** At a fixed rank, lead-1 RMSE
  rises steadily with the delays (rank 1000: 0.25 at 2 delays, 0.47 at 80)
  and the skill horizon falls to 0-2 at 80: the modes are spent on history
  instead of resolving the current state. Raising the rank doesn't buy it
  back, it **overfits**: at 80 delays the training dataset's skill horizon
  climbs to 46-55 while validation stays at 2, the same pattern as plain DMD
  past rank 500, much stronger.
- **More context helps the long horizon and the physics.** Long-delay models
  stay closer to the attractor: spectrum distance 2.66 -> 1.87, energy error
  0.27 -> 0.17, enstrophy 0.73 -> 0.59, mean RMSE down to 0.73 at 16 delays.
  The pointwise track is lost early, but the forecast keeps more of the
  flow's small scales instead of relaxing to a smooth mean. Under the
  selection rule the physics scores are guardrails, not objectives, so this
  doesn't pick the model -- but it's the trade-off a nonlinear model would
  have to break, and worth remembering when choosing its window.
- **Rank 250 is too few** at 2-8 delays: the spectrum distance is more than
  0.05 worse than the incumbent's, failing the guardrail. Stabilization was
  needed only at 80 delays and the two lowest ranks (1-2 modes); every other
  fit was stable as fitted.

**Selected: 4 delays, rank 1000**, by the ranking rule in "How candidates
are ranked": three runs tie on skill horizon (7) -- 2 delays at rank 1000
and 1500, 4 delays at rank 1000 -- and 4 delays wins the tie-break on RMSE at
lead 10 (0.571 vs 0.574 and 0.612). Its physics scores are all within the
guardrails of the incumbent, plain DMD rank 750 (energy -0.002, enstrophy
-0.004, spectra slightly better). As with the rank sweep, the margin over 2
delays is realization noise; the rule, not the margin, decides. It replaces
plain DMD as the canonical model (the DVC `train` stage); the checkpoint is
~890 MB, the rank-750 spatial basis plus a 12 MB delay basis.

## U-Net

```bash
uv run --extra gpu scripts/training/train.py model=unet
uv run --extra gpu scripts/training/train.py model=unet model.rollout_steps=4 model.training.batch_size=4
uv run --extra gpu scripts/training/train.py model=unet resume=outputs/<date>/<time>  # after a crash
```

The first neural surrogate: a U-Net that maps the last `window` frames to
the next one and forecasts autoregressively, feeding its predictions back
in. The code is split so a second network (an FNO, say) only adds the
network:

- `models/unet.py`: the network (`UNet`) and `UNetSurrogate`, which plugs it
  into the shared machinery.
- `models/neural.py`: `AutoregressiveSurrogate`, everything that isn't the
  network (normalization, constant pixels, residual update, rollout loss,
  forecasting, checkpoint).
- `training/trainer.py`: the training loop (optimizer, batches, early
  stopping, resumable state, divergence checks), generic over the network.

**The network.** An encoder-decoder of (3x3 conv, GroupNorm, GELU) blocks,
`depth` levels with `base_channels * 2**level` channels, max-pooling down,
transposed convolutions up, skip connections between levels of the same
resolution. GroupNorm rather than BatchNorm: no running statistics, and it
works at the batch sizes these frames force (8). At the defaults (base 32,
depth 4) it has 7.8M parameters, and its bottleneck (72 x 8 cells) sees most
of the domain.

**The surrogate.**
- *Input:* the last `window` normalized frames stacked as channels, plus two
  coordinate channels. Convolutions are translation-equivariant, but this
  domain isn't homogeneous: it has an inlet, an outlet and two walls.
  Frames are padded from 1151 x 127 to a multiple of `2**depth` (1152 x 128)
  by repeating the edge (zero velocity at a wall), and cropped back after.
- *Output:* the change from the newest frame (`next = last + network(...)`).
  The output layer is zero-initialized, so the untrained model *is*
  persistence, the sane baseline, rather than noise.
- *Normalization* is fitted in `fit`, like DMD's: each channel's mean and std
  over every training frame and pixel. The loss is then in the same
  per-channel units the forecast RMSE is scored in.
- *Constant pixels:* pixels that never change in the training data (both
  walls; the inlet's u_y) are reset to their value after every step. The
  boundary conditions then hold exactly over an 840-step rollout instead of
  drifting, and the loss isn't spent on them.

**Training** (`model.training.*`, `training/trainer.py`):
- *Loss:* the MSE of a `rollout_steps`-step autoregressive rollout from
  `window` true frames. `rollout_steps=1` is one-step teacher forcing.
  Longer rollouts show the model its own compounding errors, which is what
  matters over a long forecast. Each step is rematerialized
  (`jax.checkpoint`), so memory grows by one step's activations at most,
  but time grows linearly (1.3 s/step for 4 steps at batch 8, vs 0.32 s for one).
- *Input noise* (`input_noise_std`, normalized units, 0.1 by default; see
  "Input noise" below for why that value):
  Gaussian noise on each training window's true input frames, the targets
  left clean, so the network learns to pull a perturbed state back toward
  the flow instead of carrying the perturbation on. A rollout feeds on its
  own imperfect outputs; training on slightly-off inputs is the standard
  cheap counter to the drift that follows (Sanchez-Gonzalez et al. 2020;
  Stachenfeld et al. 2022 for turbulence), much cheaper than a longer
  rollout loss. Constant pixels get none. The noise is drawn from a key the
  trainer derives from the seed and the step count, so a run is still
  reproducible and a resumed one draws the same noise.
- *Optimizer:* AdamW with gradients clipped by global norm; the learning
  rate warms up linearly, then decays by cosine over `max_epochs`.
- *Epochs* are `samples_per_epoch` windows (2048 of ~7k by default) drawn
  without replacement. An epoch is the unit of validation and
  checkpointing, so it's kept short enough that early stopping (and the
  hyperparameter search's pruning) can follow the curve.
- *Early stopping on validation:* every `eval_every` epochs the model
  forecasts the whole validation dataset under the protocol, and is scored
  with `evaluation.selection_scores`. That is the selection rule's three
  numbers, stable steps, skill horizon and RMSE at lead 10, folded into one
  `selection_score`: `(n + 1) * stable_steps + skill - r / (1 + r)` for an
  `n`-step forecast, which orders like the rule exactly (one more stable
  step outweighs any skill, one more skill step any RMSE). Training stops
  after `patience` evaluations without a strictly better score, and the
  best epoch's network is the result. These are the same scores models are
  compared by, so early stopping is another decision made on validation.
  The physics guardrails are left out because they're too slow to run
  every epoch; the final model is scored in full like every model.
- *Divergence:* a non-finite loss, or one above `max_loss`, raises
  `DivergenceError`, so the run is tagged `diverged`.
- *Speed:* `compute_dtype: bfloat16` runs the network in bf16 (parameters,
  optimizer and the residual sum stay float32): 177 vs 321 ms/step at
  batch 8 on the RTX 3060, peak GPU memory 2.2 vs 3.7 GB. Training frames
  stay in host RAM as float16 (`host_dtype`, ~4 GB for the 7 datasets) and a
  background thread moves the next batch to the GPU during the current step.
- *Reproducible:* the same config gives the same losses and scores, run to
  run, on the GPU (checked).

**What's logged.** Training curves go to MLflow with the optimizer step as
the step: `loss`, `grad_norm`, `lr` and `samples_per_second` every
`log_every` steps, and `epoch_loss`, `epoch_seconds` and
`val_monitor.{skill_horizon,rmse_lead_10,stable_steps,energy_peak_ratio,enstrophy_peak_ratio,selection_score}`
every epoch. After
training, the model is checkpointed, logged and scored like every model
(`val.*`, `train.<dataset>.*`, `fit.*`, where `fit.best_epoch`,
`fit.stopped_early` and `fit.parameters` come from training).

**Resuming.** Training state is saved after every epoch into the run's
output directory, `training_state/`. It holds the network, the optimizer
state, the batch sampler's RNG and the early-stopping bookkeeping, with
`state.json` written last and atomically. `resume=<output dir>` (with the
original overrides) puts Hydra back into that directory and continues from
the last completed epoch: the same MLflow run (tagged `resumed`), the log file
appended to, and the same batches an uninterrupted run would have drawn.
A resume with a different config is refused, both by the run record
(`training_state/mlflow_run.json`) and by the trainer's own fingerprint,
since the saved weights would belong to another run. Checked on the real data
by `kill -9`-ing a run after epoch 1 and resuming it.

**Costs.** Loading the training frames takes ~15 s and the first compile
~20 s; the persistent compilation cache removes the compile on later runs.
A cold compile can log CUDA out-of-memory warnings: XLA's convolution
autotuner tries algorithms with large workspaces, and they're harmless.
Validation (an 837-step forecast) takes ~8 s. Host RAM: ~6 GB while
training (the float16 frames), ~8 GB during a validation (the validation
dataset and the forecast, ~1 GB each; the forecast is post-processed on the
GPU and copied once, and its error is computed a chunk of lead times at a
time), ~10 GB peak overall.

**Status.** Trained (precision check below), searched once (see "First
search") and stabilized with input noise (see "Input noise"). Before the
noise: on the short horizon it beat Hankel DMD by a wide margin (skill
horizon 15 vs 7), but its long rollouts were unstable and it failed the
physics guardrails. With noise 0.1 it is stable over 3000-step rollouts
(3 seeds of 3) and passes every guardrail with skill horizon 10-12; Hankel
DMD stays canonical until that switch is decided (see "Input noise"). The
defaults in `configs/model/unet.yaml` are the hand-picked starting point
plus input noise 0.1. The
U-Net isn't a DVC stage: only the canonical model is, and the U-Net becomes
canonical only if it beats Hankel DMD on validation. `training/trainer.py` and
`training/tuning.py` aren't dependencies of the current `train` stage, so
working on them doesn't mark Hankel DMD stale.

### Precision check (bf16 vs float32)

```bash
uv run --extra gpu scripts/training/train.py model=unet model.training.max_epochs=15 seed=0
uv run --extra gpu scripts/training/train.py model=unet model.training.max_epochs=15 seed=1
uv run --extra gpu scripts/training/train.py model=unet model.training.max_epochs=15 seed=0 \
    model.compute_dtype=float32 model.host_dtype=float32
```

Before spending a search on bf16, a check that it doesn't cost quality: the
default config trained for 15 epochs in bf16 with two seeds (the
seed-to-seed spread is the yardstick) and fully in float32 (network compute
and the host frames) with the first seed. Validation:

| run | skill horizon | RMSE lead 1 / 10 / 40 | best epoch | energy / enstrophy error | spectrum distance (x / y) | skill on train | train time |
|---|---|---|---|---|---|---|---|
| bf16, seed 0 | **15** | 0.079 / 0.394 / 1.07 | 10 | 22.3 / 50.7 | 0.69 / 1.17 | 11 | 833 s (46 s/epoch) |
| bf16, seed 1 | 13 | 0.087 / 0.404 / 0.94 | 7 | 14.5 / 57.1 | 0.90 / 2.75 | 12 | 827 s (46 s/epoch) |
| float32, seed 0 | 13 | 0.076 / 0.368 / 0.73 | 15 | 11.0 / 65.3 | 0.83 / 2.48 | 12 | 1456 s (83 s/epoch) |
| *Hankel DMD (canonical)* | *7* | *0.281 / 0.571 / 0.58* | | *0.26 / 0.73* | *2.62 / 2.10* | *13* | |

**Verdict: bf16 stays.** On what the search and early stopping select by,
float32 is inside the bf16 spread: its selection score (12.73) ties seed 1's
(12.71), below seed 0's (14.72); so are both spectrum distances and the
train/validation gap. It falls outside the two seeds' range on RMSE at lead
10 and the energy error (better) and the enstrophy error (worse), but a
two-seed range is a weak yardstick -- a third bf16 seed would land outside
the range of the first two with probability 2/3 on any one metric -- and the
signs are mixed, which a systematic precision loss wouldn't give. The energy
and enstrophy errors also measure a rollout that has already blown up (see
below), where 11 vs 22 is a growth rate, not an accuracy. float32 costs 1.75x
per epoch, which in a fixed search budget is ~4 fewer trials.

**What the check showed about the model.** On the short horizon the U-Net
is far ahead of everything so far: skill horizon 13-15 against Hankel DMD's
7, RMSE at lead 1 0.08 against 0.28. But its **long rollouts are unstable**:
the energy error is 11-22 (Hankel DMD: 0.26) and the mean RMSE 4-6. Rolled
out past the validation dataset from its context (`seed 0` above), the
domain energy is 1.2x the true level in the first 100 steps, 6x by step 300
and still growing at step 1500, with no NaN: the classic drift of a network
trained only on one-step errors, which never sees its own outputs. Two other
things: the validation score is noisy from epoch to epoch (seed 0: 8.7,
7.6, 7.6, 9.7, 8.7, 9.7, 9.7, 7.6, 11.7, 14.7, ...), so early stopping picks
partly on noise; and the training dataset's skill (11-12) is no higher than
validation's, so the model isn't overfitting at this budget.

### Hyperparameter search (Ray Tune: Optuna + ASHA)

```bash
uv run --extra gpu --extra ray scripts/training/tune.py                      # 24 trials
uv run --extra gpu --extra ray scripts/training/tune.py search.num_samples=8 model.training.max_epochs=30 \
    search.time_budget_s=9000                                                 # capped at 2.5 h
```

`scripts/training/tune.py` searches the space in `configs/search/unet.yaml`
(learning rate, weight decay, width, depth, window, rollout steps) with Ray
Tune. Its root config, `configs/tune.yaml`, is `config.yaml` plus a
`search` group. Any training override works as it does for train.py.

- **A trial is a training run.** `train.py`'s body moved to
  `training/run.py`'s `run_training`, and each trial calls it with the
  sampled values applied as dotted overrides
  (`training/tuning.py:run_trial`). A trial is therefore exactly what
  `train.py model=unet <overrides>` would do: the same data provenance,
  early stopping, checkpoint, logged model and full validation scores.
  (Code that runs on Ray workers has to be importable from `src/`, which is
  why the run body moved there.)
- **One metric, the selection rule.** A trial reports its validation scores
  to Tune after every validation, and the search maximizes
  `selection_score`: stable steps, then skill horizon with RMSE at lead 10
  as tie-break, the number early stopping already uses. The search optimizes what models are
  chosen by, on validation only.
- **Optuna proposes, ASHA prunes.** Optuna's TPE sampler (seeded) models
  which regions of the space score well, from the trials so far, and samples
  the next configuration there instead of uniformly at random. ASHA
  (asynchronous successive halving) compares trials at rungs of 4, 12 and 36
  validations (`grace_period` x `reduction_factor`^k) and stops those below
  the top third of what reached the rung. Most of the budget goes to
  promising configurations; a bad learning rate is dropped after 4 epochs,
  not 60. Both are needed: ASHA alone samples blindly, Optuna alone trains
  every trial to the end. (Not HyperOpt: Optuna is its maintained TPE
  successor and integrates with Tune the same way.)
- **The defaults are the first trial.** With `search.evaluate_defaults`
  (on by default), Optuna's first proposal is the config's own values of
  the searched keys (`training/tuning.py:default_point`, passed as
  `points_to_evaluate`): the hand-picked defaults are measured under the
  same budget as everything else, and the search has a baseline to beat.
  It counts as one of the `num_samples`, and a default outside its domain
  is refused (a sampler can't evaluate it).
- **MLflow.** The search is a parent run (`unet search`), holding the
  search config and, at the end, the best trial's values (`best.*`) and its
  run id (`best_run_id` tag). Each trial is a nested child run, tagged with
  the search's `sweep` name like a Hydra multirun. A pruned trial's process
  is ended by Ray mid-training, so its run can't close itself: the driver
  marks it KILLED, tags it `pruned=true` and uploads its log. ASHA would also
  stop a trial that reaches its `max_t` iterations, which at the last
  validation would kill it before its final checkpoint and scoring, so
  `max_t` is set one past the last validation.
- **Time cap.** `search.time_budget_s` (seconds; `null`, the default, is
  no cap) is Ray Tune's own `TuneConfig(time_budget_s=...)`: when the
  search has run that long, Tune stops the running trials and starts none
  of the remaining samples. It is a backstop, not the way to size a search:
  a trial stopped by it ends mid-training like a pruned one, with no final
  checkpoint or full scores, so `num_samples` and `max_epochs` should be
  chosen to finish within it. Tune reports a trial stopped by ASHA and one
  stopped by the budget the same way, so the scheduler records the trials
  ASHA stopped (`training/tuning.py:recording_asha`), and the driver tags
  the others' runs `time_budget=true` instead of `pruned=true` (and the
  parent run `time_budget_reached=true`): a budget-stopped trial was cut
  short, not judged worse than the others. Checked with a toy trainable:
  ASHA's stops are recorded under the same trial ids the results carry, the
  trial running at the deadline isn't among them, and the samples not yet
  started are dropped from the results.
- **Resources.** One trial at a time by default (`max_concurrent_trials:
  1`, 1 GPU, 10 GB memory budget per trial). On one GPU, concurrent trials
  split it rather than add throughput, and each trial peaks at ~10 GB of
  host RAM on a 23 GB machine. Ray still earns its place here through ASHA's
  asynchronous scheduling, a per-trial resource budget, and the fact that
  the same script scales to more GPUs or a cluster (KubeRay) by changing
  `resources_per_trial` and `max_concurrent_trials`. Ray is started through
  `utils/parallel.py`'s `init_ray`, as everywhere else.
- **Output.** `outputs/tune/<date>/<time>/` holds the driver's log,
  `trials/<trial id>/` (each trial's log, checkpoint and training state) and
  `ray/` (Tune's own results). The result table is printed best first.

Smoke-tested on the real data (4 trials, 4 epochs of 64 windows,
`grace_period=1, reduction_factor=2`): two trials ran to the end with full
scores and two were pruned (KILLED, tagged, log uploaded). The same seed
proposed the same configurations on a rerun. Choosing the checkpoint to
keep from a real search, and pruning the losers', follows CLAUDE.md's
sweep rules.

#### First search (2026-10-06)

```bash
uv run --extra gpu --extra ray scripts/training/tune.py search.num_samples=10 \
    model.training.max_epochs=30 search.time_budget_s=8100 \
    'search.space={model.base_channels:{type:choice,values:[16,32]}}'
```

Sized for a ~2.25 h budget: 10 samples, 30 epochs, the time cap as a
backstop, and width 48 dropped from the space for this run (2.2x the cost
of width 32 per epoch, and 4 epochs at width 48 with a 4-step rollout would
have cost ~30 min before ASHA could judge it). bf16, per the precision
check. The cap was reached: 8 of the 10 samples ran. Validation (sweep
`tune-2026-10-06-17-08-15`), in order of Optuna's proposals:

| # | learning rate | weight decay | width / depth | window | rollout steps | outcome | validations | best val score | skill horizon | RMSE lead 1 / 10 / 40 | energy / enstrophy error | spectrum distance (x / y) | skill on train |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (defaults) | 1e-3 | 1e-4 | 32 / 4 | 4 | 1 | early-stopped (best epoch 10) | 18 | 12.72 | 13 | 0.080 / 0.392 / 0.89 | 25.6 / 62.5 | 0.93 / 2.59 | 13 |
| 2 | 6.5e-4 | 1.4e-4 | 16 / 4 | 4 | 1 | early-stopped (best 17) | 25 | 11.71 | 12 | 0.083 / 0.405 / 1.03 | 583 / 1122 | 2.62 / 4.58 | 11 |
| 3 | 2.3e-3 | 1.6e-6 | 16 / 3 | 2 | 1 | pruned at rung 12 | 12 | 8.66 | | | | | |
| **4** | **1.6e-4** | **6.8e-4** | **16 / 4** | **8** | **4** | early-stopped (best 17) | 25 | **14.74** | **15** | 0.143 / **0.359** / 0.74 | 330 / 1782 | 2.07 / 3.49 | 14 |
| 5 | 1.0e-3 | 1.2e-5 | 32 / 4 | 1 | 2 | pruned at rung 12 | 12 | 13.72 | | | | | |
| 6 | 2.9e-3 | 2.0e-6 | 16 / 3 | 1 | 1 | pruned at rung 4 | 4 | 9.68 | | | | | |
| 7 | 3.5e-4 | 2.9e-4 | 32 / 4 | 2 | 2 | GPU out of memory (epoch 3) | 2 | 9.67 | | | | | |
| 8 | 2.7e-4 | 2.3e-6 | 32 / 4 | 1 | 2 | stopped by the time budget | 11 | 11.72 | | | | | |
| *Hankel DMD (canonical)* | | | | *4* | | | | *6.64* | *7* | *0.281 / 0.571 / 0.58* | *0.26 / 0.73* | *2.62 / 2.10* | *13* |

Pruned and budget-stopped trials have no final scores (only the selection
score each validation reports). Trial 5's best (13.72, at epoch 10) was
above the rung-12 cut, but ASHA judges the score *at* the rung (10.7), which
the validation noise makes a coin toss.

**Best trial: #4** (`9fc8ffad`, lr 1.6e-4, weight decay 6.8e-4, width 16,
depth 4, window 8, 4-step rollout loss, 1.9M parameters, ~110 s/epoch).
Against Hankel DMD under the selection rule:

- **Ranking:** skill horizon 15 vs 7, RMSE at lead 10 0.359 vs 0.571. The
  U-Net ranks first by a wide margin.
- **Guardrails:** energy error 330 vs 0.26 and enstrophy error 1782 vs 0.73
  (tolerance +0.03), spectrum distance along y 3.49 vs 2.10 (tolerance
  +0.05) -- **failed**, by orders of magnitude. Spectrum distance along x
  passes (2.07 vs 2.62). **It does not replace Hankel DMD**, and no trial
  of this search would: every finished one fails the energy and enstrophy
  guardrails.

**Long rollouts are unstable**, for every U-Net trained so far. Rolled out
from the validation context past the dataset's length (the model's own
`predict` in 100-step blocks, domain-mean energy per block against the true
validation range, 0.91-0.95; see "Long rollouts" under "Forecast
evaluation"):

| steps | 0-100 | 100-200 | 200-300 | 500-600 | 900-1000 | 1400-1500 |
|---|---|---|---|---|---|---|
| best trial (#4, 4-step loss): energy | 0.97 | 7.1 | 40 | 410 | 1234 | 3140 |
| precision run (defaults, 1-step loss): energy | 1.15 | 2.9 | 6.1 | 28 | 88 | 226 |
| Hankel DMD: energy (500-step blocks) | 0.70 (0-500) | | | 0.65 (500-1000) | | 0.65 (1000-1500) |

(Regenerated with `scripts/evaluation/check_rollout_stability.py`, which
replaced the scratch check; the U-Net rows came out within ~10-30% of the
scratch numbers, which an unstable rollout amplifies from rounding.)
No NaN within 1500 steps, but the energy grows without bound in both U-Nets,
while Hankel DMD's settles at a damped, smooth state (energy 0.65, enstrophy
6 vs the true 27). Training on a 4-step rollout keeps the first ~100 steps
at about the truth's energy (0.97) and makes the forecast more accurate at lead
10-40, but it diverges *faster* after that, not slower: a 4-step horizon
teaches it nothing about step 100.

What the search shows:

- **The selection score alone picks unstable models.** The search, early
  stopping and the ranking all optimize the short horizon (skill horizon,
  RMSE at lead 10), which a model can win while blowing up later; the
  guardrails catch it only after training. The physics are checked once,
  at the end, because they're too slow for every epoch. *Since fixed:* the
  selection rule now ranks by stable steps first, a check cheap enough for
  every validation (see "How candidates are ranked"). Re-scored under it,
  the best trial is stable for 100 of the 837 validation steps (energy peak
  995x the truth's, enstrophy 4408x) and Hankel DMD for all 837 (peaks
  0.81x and 0.38x), so Hankel DMD now ranks first by the rule itself, not
  only by the guardrails.
- **The validation score is noisy** (±2 between consecutive epochs of the
  same run), so early stopping and ASHA's rung comparisons decide partly on
  noise. Trial 5 was pruned at a rung where its score happened to dip.
- **Small and slow-learning won:** the best trial is the narrower network
  (width 16) with the lowest learning rate, so capacity isn't the
  bottleneck at this budget.
- **One trial hit a GPU out-of-memory** inside the validation forecast at
  its third validation, after two that succeeded with the same shapes:
  transient pressure on the shared 12 GB GPU (WSL shares it with the
  desktop), not the configuration's size. Ray recorded it as an errored
  trial (its run is FAILED) and went on.
- The time cap worked as designed: the trial running at the deadline was
  stopped and tagged `time_budget=true`, the three ASHA stopped
  `pruned=true`.

The losing trials' checkpoints were pruned after the search (CLAUDE.md's
sweep rules): trials 1 and 2's logged models and `model/` directories (runs
tagged `checkpoint_pruned`), and every losing trial's `training_state/`.
The best trial's checkpoint is kept.

#### Input noise (2026-10-07)

```bash
uv run --extra gpu scripts/training/train.py -m model=unet \
    'model.input_noise_std=0.1,0.3,0.03,0.0' model.training.max_epochs=30
uv run --extra gpu scripts/evaluation/check_rollout_stability.py \
    --checkpoint outputs/multirun/<date>/<time>/<n>/model --steps 3000
```

With the selection rule now ranking by stable steps first, the U-Net was
retried with the cheapest standard fix for autoregressive drift: Gaussian
noise on the training inputs (`model.input_noise_std`, see "U-Net"). The
defaults otherwise (width 32, depth 4, window 4, one-step loss, lr 1e-3),
seed 42, 30 epochs, patience 8; sweep `12-15-40`. Validation, at each run's
selected epoch, plus a 3000-step rollout of its checkpoint
(`check_rollout_stability.py`, 100-step blocks):

| input noise | selected / last epoch | stable steps (val) | skill horizon | RMSE lead 1 / 10 / 40 | energy / enstrophy error | spectrum distance (x / y) | stable steps (train dataset) | 3000-step rollout |
|---|---|---|---|---|---|---|---|---|
| 0 (control) | 10 / 18 | 200 | 13 | 0.080 / 0.392 / 0.89 | 25.6 / 62.5 | 0.93 / 2.59 | 200 | unstable from 200 (energy 1650x by 3000) |
| 0.03 | 20 / 28 | 837 | 11 | 0.072 / 0.440 / 0.87 | 0.165 / 0.547 | 0.81 / 0.63 | 300 | unstable from 300 |
| **0.1** | **5 / 13** | **837** | **10** | 0.091 / **0.486** / 0.93 | **0.129 / 0.244** | **1.35 / 0.73** | **1168 (all)** | **stable** (energy 0.95-1.25x the truth's) |
| 0.3 | 7 / 15 | 837 | 7 | 0.108 / 0.705 / 1.28 | 0.317 / 0.021 | 0.62 / 0.69 | 300 | stable, damped (energy 0.47-0.88x) |
| *Hankel DMD (canonical)* | | *837* | *7* | *0.281 / 0.571 / 0.58* | *0.264 / 0.728* | *2.62 / 2.10* | *1168* | *stable, damped (0.68-0.81x)* |

**Noise makes the U-Net stable.** Without it, no epoch of any U-Net so far
stayed bounded past 200-300 validation steps (500 with the full schedule,
below). With noise 0.1, the selected model stays on the attractor for the
whole validation forecast, the training dataset's (1168 steps) and a
3000-step rollout, its energy within 25% of the truth's throughout -- the
first U-Net that does. Under the
selection rule it ranks above Hankel DMD (equal stable steps, skill
horizon 10 vs 7, RMSE at lead 10 0.486 vs 0.571) and passes every
guardrail with a margin: energy and enstrophy error 0.13 and 0.24 against
Hankel DMD's 0.26 and 0.73, spectrum distances 1.35 / 0.73 against 2.62 /
2.10. The price is short-horizon accuracy: skill 10 against the noise-free
U-Net's 13-15, because the network now also learns to undo perturbations.
Too much noise (0.3) damps the flow; too little (0.03) leaves it only
marginally stable.

**Stability on one validation forecast can be luck.** Stable steps flip
between consecutive epochs of the same run (noise 0.1: 0, 837, 0, 200,
837, 100, 0, ...), and noise 0.03's selected epoch, stable over the
validation forecast, diverges from step 300 in the block-wise rollout,
whose only difference is float rounding where the blocks join, and in the
training dataset's forecast. A marginally stable model passes or fails the
check depending on which trajectory it happens to take. Two consequences:
early stopping on a flipping score keeps an early lucky epoch and stops
before the learning rate has decayed (noise 0.1 stopped at epoch 13 of
30), and a candidate should also pass the long rollout before it replaces
the incumbent.

**The domain-mean `u_y` drifts.** The truth's domain-mean cross-stream
velocity stays at 0 +- 0.006, oscillating with the EDA's ~25-step period.
The forecasts of these four models drift away from it (time means of
0.06-0.64 over the validation forecast; noise 0.1's is 0.10, a quarter of
`u_y`'s per-pixel std of 0.42 and ~20x the amplitude of the truth's
coherent oscillation). The temporal scores show it as `u_y_period_error`
6.806, which every U-Net ever trained has: the slow drift puts the Welch
peak in the lowest-frequency bin, which `interpolated_peak` returns
unrefined (period 200, the segment length; 200 / 25.6 - 1 = 6.806). It is a
spurious net cross-stream flow, not an oscillation, and neither the
stability check nor the guardrails see it. (The fully trained models below
drift far less; see there.)

**Seeds and the full schedule** (sweep `13-40-27`). Two questions the
first sweep left open: is noise 0.1's stability luck of one seed, and is it
the noise or just the learning rate decaying (the noise-free run had
stopped early, at epoch 18)? Each setting again with seeds 0 and 1, all 30
epochs (`model.training.patience=30`, so early stopping can't end a run
before the cosine schedule has decayed; the best epoch is still the one
kept):

| input noise | seed | selected epoch | stable steps (val) | skill horizon | RMSE lead 1 / 10 / 40 | energy / enstrophy error | spectrum distance (x / y) | stable steps (train dataset) | 3000-step rollout |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 23 | 500 | 14 | 0.071 / 0.369 / 0.65 | 0.073 / 2.13 | 0.66 / 1.03 | 200 | unstable from 400 (energy 556x by 3000) |
| 0 | 1 | 21 | 400 | 11 | 0.071 / 0.452 / 0.81 | 0.970 / 3.87 | 0.59 / 1.57 | 400 | unstable from 400 (54x) |
| 0.03 | 0 | 22 | 837 | 13 | 0.069 / 0.418 / 0.75 | 0.230 / 0.362 | 0.72 / 0.79 | 1168 | stable (energy 0.71-0.88x, enstrophy 0.97-1.24x) |
| 0.03 | 1 | 21 | 600 | 12 | 0.072 / 0.420 / 0.76 | 0.126 / 1.01 | 0.77 / 0.75 | 800 | over 2x from 700 (enstrophy up to 5.3x, then back to ~1.4x) |
| **0.1** | **0** | **25** | **837** | **12** | 0.073 / **0.416** / 1.05 | **0.162 / 0.306** | **0.69 / 0.78** | **1168** | **stable** (energy 0.65-1.14x, enstrophy 0.88-1.94x) |
| 0.1 | 1 | 12 | 837 | 11 | 0.080 / 0.475 / 0.89 | 0.240 / 0.625 | 1.28 / 0.74 | 1168 | stable (energy 1.13-1.33x, enstrophy 1.14-1.84x) |

Over the three seeds (42, 0, 1), the selected checkpoint survives the
3000-step rollout **3 of 3 times with noise 0.1, 1 of 3 with 0.03 and 0 of
3 without noise**. The full schedule alone doesn't do it: the noise-free
runs' late epochs blow up more slowly (energy peaks of 1.3-40x instead of
hundreds), but none stays bounded over the validation forecast, and their
enstrophy -- the small scales -- runs away first (13-15x at the selected
epochs). Noise 0.1 is the setting that works, and it is now the default
(`configs/model/unet.yaml`). What it doesn't fix completely is the small
scales: seed 1's late epochs keep their energy within 1.4x of the truth's
but carry 4.5-7.5x its enstrophy, so its best stable epoch is an early one
(12), and its spectrum distance along x is the worst of the three.

The fully trained noise models also mostly lose the domain-mean `u_y`
drift: their time means are -0.030 (noise 0.03, seed 0), -0.0002 and 0.006
(noise 0.1, seeds 0 and 1) instead of 0.06-0.64. It still wanders slowly,
though, with 2-4x the truth's variability (std 0.011-0.021 against 0.0055),
and the lowest frequencies dominate its spectrum instead of the truth's
25.6-step oscillation -- hence the same 6.806 period error.

**Against Hankel DMD.** By the selection rule the top checkpoint is noise
0.03, seed 0 (skill horizon 13), then noise 0.1, seed 0 (12, RMSE at lead
10 0.416): both stable over the validation forecast, both through every
guardrail by a wide margin (energy and enstrophy error 0.23 / 0.36 and 0.16
/ 0.31 against Hankel DMD's 0.26 / 0.73; spectrum distances 0.7-0.8
against 2.1-2.6), both stable over 3000 steps. Every noise-0.1 checkpoint,
and both noise-0.03 ones stable over the validation forecast (seed 42's
included, though it diverges in the long rollout), would replace Hankel DMD
under the rule as written. **The canonical model is unchanged** -- that switch is a separate
decision, and three things argue for making it deliberately rather than by
the letter of the rule:

- Noise 0.03's checkpoint is the top one by luck of its seed (its recipe
  holds 1 time in 3); noise 0.1, seed 0 is the robust recipe's best.
- A stable validation forecast is necessary but, as noise 0.03 at seed 42
  showed, not sufficient: the long rollout should be part of the
  guardrails before a switch, and its length and threshold need choosing.
  Rolled out for 10000 steps, noise 0.1, seed 0 never runs away (energy
  0.65-1.45x the truth's throughout, 0.70-0.86x over the last 1000 steps),
  but 9 of its 100 blocks have bursts of small-scale activity just over
  the 2x line (enstrophy 2.05-2.73x, from step 3300 on), each of which
  recovers: by the letter of the check it is stable for 3300 steps, by eye
  it stays on a slightly too energetic attractor.
- The U-Nets' domain-mean `u_y` wanders slowly instead of oscillating
  (above), which the temporal scores flag but no score decides on.

Cost: ~57 s per epoch including its validation, ~31 min for 30 epochs and
the final scoring; a 3000-step rollout check takes ~35 s on the GPU (it
needs one: an 837-step U-Net forecast on the CPU hadn't finished after 45
minutes).

The losing checkpoints of both sweeps were pruned (CLAUDE.md's sweep
rules); the kept ones are noise 0.1 at seeds 42, 0 and 1 and noise 0.03 at
seed 0 (the stable candidates).

## Docker

The project ships as a container image so the exact environment (Python
3.14, the locked dependency set, the bundled ffmpeg) is reproducible on any
machine, and later stages (training jobs, serving) have a unit to deploy.

- **`Dockerfile`** — multi-stage. `builder` installs the locked dependencies
  with uv (in their own cached layer, so source edits don't reinstall
  everything), then the project non-editable. `runtime` (default) copies only
  the finished virtualenv plus `scripts/` and `configs/` into a fresh
  `python:3.14-slim`: no uv, no dev tools, runs as a non-root user.
  `test` adds the dev group, `tests/` and the DVC pointer files its tests
  parse (`dvc.yaml`, `dvc.lock`, `data/raw.dvc`), and runs pytest.
- **Data is never in the image.** The zarr store is ~9 GB and changes
  independently of the code, so `data/`, `reports/`, and `outputs/` are bind
  mounts, as is the JAX compilation cache (`.jax_cache/`), so the container
  doesn't recompile on every run. `.dockerignore` keeps them out of the build context too.
- **`docker-compose.yml`** — the app container plus a full MLflow tracking
  stack (below).
- **The source commit is baked in, not read from git.** The image has no git
  and no `.git`, so MLflow can't detect a run's commit itself. Mounting `.git`
  wouldn't fix that properly: it shows the checkout's *current* commit, which
  needn't be the one the image's code was built from, and a wrong tag is
  worse than none. Instead the `GIT_COMMIT` build arg becomes the
  `MHD_GIT_COMMIT` env var (and the `org.opencontainers.image.revision`
  label), and `tracked_run` sets it as `mlflow.source.git.commit` on new
  runs. CI passes `github.sha`; locally, build with the variable set (it's
  empty otherwise, and runs go untagged).

```bash
GIT_COMMIT=$(git rev-parse HEAD) docker compose build
docker compose up -d                          # tracking stack; UI on http://localhost:5000
docker compose run --rm app                   # train.py, logging to the stack
docker compose run --rm --no-deps app python scripts/analysis/run_all_checks.py
docker compose run --rm test                  # pytest inside the image
docker compose down                           # stop; all state stays in ./mlflow
```

(`--no-deps` skips starting the tracking stack for scripts that don't log to it.)

### GPU image

`docker compose run --rm app-gpu` runs the same entrypoint on an NVIDIA GPU.
It is a separate service (profile `gpu`) rather than a change to `app`, so the
default image and CI stay CPU-only:

- The `Dockerfile` takes a `UV_EXTRAS` build argument (default
  `--extra server --extra ray`); `app-gpu` builds `mhd-surrogate:gpu` with
  `--extra gpu` added. The CUDA libraries come from the pip wheels, so the
  base image is still `python:3.14-slim` (the image is ~8.7 GB vs ~2.3 GB for
  the CPU one).
- `docker-compose.yml` reserves the host GPU for the service
  (`deploy.resources.reservations.devices`).
- The host needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  registered with Docker (`sudo nvidia-ctk runtime configure --runtime=docker`).
  Under WSL2 the Windows driver provides the GPU; don't install a Linux driver
  inside WSL. Check with `docker run --rm --gpus all ubuntu nvidia-smi`.
- Verify the image sees the GPU:
  `docker compose run --rm --no-deps app-gpu python -c "import jax; print(jax.devices())"`.
- CI doesn't build this image (no GPU on the runners, and the CUDA layers are
  large); `docker compose config -q` still validates the service.

CI (`.github/workflows/docker.yml`) builds both image targets, lints the
`Dockerfile` with hadolint, runs the `test` image, validates
`docker-compose.yml`, and brings the tracking stack up with `--wait` to
check the services' healthchecks. It doesn't run `app`, since that needs
`data/`, which isn't present in CI.

### Tracking stack

```
app ──HTTP──▶ mlflow server ──SQL──▶ postgres     runs, params, metrics
                    └────S3 API────▶ seaweedfs    artifacts (config.json, ...)
```

This is the layout MLflow recommends for real deployments, run locally:

- **Postgres** is the backend store. SQLite (used for plain local runs, see
  above) can't handle concurrent writers, which parallel training runs would be.
- **SeaweedFS** provides an S3-compatible object store for artifacts. Model
  checkpoints and plots don't belong in a relational database. It stands in
  for S3/GCS: because MLflow talks to it over the standard S3 API, pointing
  the same stack at real S3 later is a config change, not a code change.
  (MinIO would be the usual choice, but it no longer publishes container
  images and its repository is archived, so it isn't a sound dependency.)
- The MLflow server runs with `--serve-artifacts`: clients upload artifacts
  to the server, which writes them to the object store, so training code
  needs no S3 credentials or endpoint. Host validation is on
  (`--allowed-hosts`), with the in-network name `mlflow:5000` allow-listed.
- A one-shot `s3-init` container creates the buckets: `mlflow` for artifacts
  and `dvc` for the data remote (see "Data versioning (DVC)").
- The server's extra dependencies (`psycopg2`, `boto3`) live in the
  `server` extra in `pyproject.toml`; training clients don't need them. The
  image also installs the `ray` extra (for `--backend ray`; see "Parallel
  backends"), and compose gives the containers `shm_size: 1gb` for Ray's
  object store.

**Persistence.** All state lives in bind mounts inside the project —
`mlflow/postgres/` and `mlflow/seaweedfs/` (gitignored, with tracked
`.gitkeep` placeholders) — rather than Docker-managed volumes, so
`docker compose down -v` or a volume prune can't delete your run history
(verified: runs and artifacts survive `down -v`). Back it up by copying
`mlflow/`, with the stack stopped. The services run as your host user
(`HOST_UID`/`HOST_GID`, default 1000), so the files are yours and need no
`sudo` to manage. Delete `mlflow/postgres/*` and `mlflow/seaweedfs/*` to
reset tracking.

**Configuration.** Credentials default to local-development values in
`docker-compose.yml`; copy `.env.example` to `.env` to override them. They
are not meant for anything reachable beyond your machine.

**Using the stack from the host.** With the stack up, `uv run
scripts/training/train.py mlflow=server` logs to it via `localhost:5000`
(the `mlflow/server.yaml` config); the default `mlflow=local` still uses the
SQLite file and needs no containers.

The default image is CPU-only; `app-gpu` adds the CUDA build of JAX (see
"GPU image").

`scripts/` is located relative to the working directory (`/app` in the
container), not relative to the installed package, since in the image the
package lives in `site-packages`. Keep bind-mount source directories tracked
(`.gitkeep`): Docker creates missing ones as root, which the non-root
container user can't write to.
