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
Serving is being built: the canonical model is promoted from the experiment
store to a model registry on the tracking server and served by a FastAPI
service that streams forecasts, on a local Kubernetes cluster that Argo CD
deploys from `main` (see "Serving").

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

**Kubernetes.** `deploy/install-tools.sh` installs the pinned kind, kubectl,
helm, kubeconform, argocd and promtool into `~/.local/bin`, each checked
against its release's checksum (see "Kubernetes" under Serving).
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
  training/          mlflow_utils, tracking, startup, export, mlflow_model, run, scoring, gating, trainer, tuning,
                     mlflow_codec, codec_run
  evaluation/        protocol, metrics, quantities, stability, selection, rollout, gate, diagnostics, ensemble, evaluate,
                     reconstruction
  models/            base (interface), baselines, snapshots, dmd, hankel_dmd, neural, unet, registry,
                     codec (codec interface + registry), pod_codec, latent, latent_linear
  serving/           promotion, fetch, stream, schemas, app, client, metrics
  utils/             logging_config, parallel, jax_cache, jax_determinism, hydra_resolvers
scripts/             CLI entry points, same split (plus viz/)
  data/              explore_data, convert_to_zarr, compute_stats
  analysis/          check_*.py, run_all_checks, benchmark_backends
  viz/               make_video, make_all_videos, make_forecast_video
  training/          train, tune, train_codec
  evaluation/        check_rollout_stability, evaluate_checkpoint, gate_checkpoint
  serving/           promote, fetch_model, check_parity, image_tag, smoke_model, smoke_check
tests/               mirrors src/ and scripts/ (data/, analysis/, training/, evaluation/, models/, serving/, utils/, viz/)
deploy/              Kubernetes: kind/ (the local cluster), k8s/ (Kustomize base, overlays,
                     smoke-check, monitoring),
                     argocd/ (Argo CD install + Applications), install-tools.sh, validate.sh
dvc.yaml, dvc.lock  data pipeline (raw -> zarr -> stats) and its pinned hashes
data/raw.dvc         DVC pointer to the raw .npy files (.dvc/ holds the remote config)
configs/             Hydra tree: config.yaml (tune.yaml for the search, train_codec.yaml for codecs) + data/, dataset/, normalization/, model/, codec/, evaluation/, mlflow/, search/ groups
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
say, the evaluation code without re-running the `evaluate` stage fails CI
instead of leaving the canonical model's `metrics.json` claiming code that
no longer exists (#52 slipped through exactly that way; #54 refreshed the
lock). The deps that come from other stages (the checkpoint, `metrics.json`)
it checks within the lock instead, plus the metrics files against git (see
"A retrain has to pass the gate" below).

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
division of labor, and for the canonical model the `train` and `evaluate`
stages below make
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

### Training and evaluation stages

```bash
uv run dvc repro            # re-runs only the stages whose inputs changed
uv run dvc metrics show     # the canonical model's validation (and train) scores
uv run dvc metrics diff     # ... compared with the last commit
```

DVC is more than data versioning: `dvc.yaml` is a pipeline of stages with
declared dependencies, parameters and outputs, and four stages extend it
from the data to the **canonical model** and how it is served -- currently Hankel DMD at the delays and
rank its validation sweep selected (see "Delay sweep"):

- **`train`** fits the model (`train.py ... evaluation.final=false`) and
  stores its checkpoint at a fixed path (`export.dir`): `models/<model>/model`,
  cached and pushed like the stats. Its deps are the data, the stats, the
  code `train.py` imports and the config keys that change the fitted model
  (the model's own, the split, and for a model trained iteratively what early
  stopping scores on).
- **`evaluate`** scores that stored checkpoint
  (`scripts/evaluation/evaluate_checkpoint.py`) exactly as a training run
  scores itself at the end, and writes `models/<model>/metrics.json`,
  declared as DVC metrics and kept in git, so `dvc metrics diff` shows how a
  change moved the scores between commits. Its deps are the checkpoint, the
  data, the evaluation code and the evaluation config.
- **`gate`** checks that the stored checkpoint still meets the bar that made
  it canonical (`scripts/evaluation/gate_checkpoint.py`), and fails
  `dvc repro` if it doesn't. It writes its verdict to `models/<model>/gate.json`
  (also DVC metrics, kept in git): the 3000-step rollout's stable steps and
  peak ratios, and `passed`.
- **`serve_parity`** checks the serving code against that checkpoint
  (`scripts/serving/serve_parity.py`, see "Serve parity" under Serving): the
  forecast the service serves must be the model's own. It writes
  `models/<model>/serve_parity.json`, kept in git, which must say `passed`.

The split is about what a change costs. A new score or a fix in the
diagnostics re-runs `evaluate` on the stored checkpoint (about a minute)
instead of refitting the model (minutes for Hankel DMD, half an hour or more
for a neural network), and the `metrics.json` diff then shows the change's
effect on the *same* model. What early stopping needs from the evaluation
code (`evaluation/selection.py`: the selection score, the stability check)
is a `train` dep; the rest (diagnostics, ensemble scores, the full
`evaluate`) is only `evaluate`'s, because `run.py` imports the final scoring
(`training/scoring.py`) by name and only when a run scores itself, which the
`train` stage doesn't. Both stages still log to MLflow like any run: the
`train` run tags itself and its logged model with the checkpoint's content
hash (`checkpoint_sha256`), and the `evaluate` run finds that logged model by
it and links its scores to it. A re-run that reproduces the checkpoint bit
for bit doesn't log it again: it reuses the logged model of identical
content (see "Logged models" below), so re-running the pipeline doesn't
store another ~900 MB copy each time.

**A retrain has to pass the gate.** A retrain is a new draw of its recipe:
a deterministic retrain of the noise-0.1 U-Net (2026-10-08) came out
stable for only 400 of the 837 validation steps, where three earlier draws of
the same recipe had been stable throughout (see "Input noise"), and nothing
in `train` or `evaluate` would have noticed -- whatever the retrain produced
would have become the canonical model, `metrics.json` and all. The `gate`
stage applies the selection rule's guardrails to the canonical model itself
(`evaluation/gate.py`): it must be stable over the whole validation forecast
and over a 3000-step rollout from the validation context (the same 100-step
blocks and 2x threshold), and its energy and enstrophy errors and spectrum
distances must stay within limits and its domain-mean velocity within the
truth's band (`configs/evaluation/default.yaml`, `gate`). A generative
model must pass the stability and drift checks with at least 80% of its
members, each passing all of them (`gate.min_member_fraction`; its
members' own scores are in `metrics.json`), and the others with its median
member (see "How candidates are ranked"). When it fails, the command exits non-zero: `dvc repro` stops and
the stage stays stale in `dvc status`, while `gate.json` and the printed
failures say what went wrong. That alone doesn't keep the failed model out
of a commit: DVC locks each stage as it finishes, so `train` and `evaluate`
are already locked with the new checkpoint and `metrics.json`, both on disk.
CI closes that gap from `dvc.lock` and git alone, no data needed
(`tests/data/test_dvc_lock.py`): every stage must have run on the version of
an upstream output its producer locked (the failed `gate` still records the
old checkpoint and scores), the committed metrics files must be the ones the
lock records, and `gate.json` must say `passed`. So a commit is green only if
the gate passed on exactly the committed model and scores. The limits are **frozen when a model becomes canonical**:
its validation scores then plus the margins a replacement may be worse by
(0.03 on the errors, 0.05 on the spectra), currently Hankel DMD's; the
drift limits are absolute and the same for every model. They are
not the last `metrics.json`: limits that followed each retrain would let
every retrain get a margin worse than the one before and the bar drift down.
Replacing the canonical model stays a decision made on validation runs;
the gate only checks that a retrain of it still meets the bar that decision
set. It checks, it doesn't fix: for a recipe that passes on some draws only,
the remedy is several seeds selected on validation, not retrying until the
gate is green. (Hankel DMD's fit is deterministic, so for it the gate guards
against code changes; its rollout takes 20 s.)

**Deps are exactly what's imported.** Each stage's code deps are the modules
its entry point imports, transitively -- not the whole package -- so editing
an unrelated module (`analysis/pod.py`, say) doesn't make either stale. A test
(`tests/training/test_dvc_train_stage.py`) computes those import sets and
fails if a module isn't covered, and that the `train` stage really leaves
scoring to `evaluate`. Only the canonical model's modules are deps, not all
of `models/`: `models/registry.py` maps each model name to an import path and
imports a model's module only when it's built or loaded, so editing another
model doesn't mark the stages stale (the test adds the trained model's module
itself, from the `train` stage's `model=` override). A second test
(`tests/data/test_dvc_lock.py`) asks DVC whether any stage's code or params
changed since `dvc.lock` was written, so a PR can't merge with a stale lock.
`.dvcignore` excludes `__pycache__`, which importing the package would
otherwise change. When only the dep *list* changes, not the code, `uv run dvc
commit -f <stage>` records the new deps without re-running.

**Re-runs reproduce their outputs.** The model stages run with deterministic GPU
kernels (`jax.deterministic_ops=true`, see "Training config (Hydra)"), so on
the same GPU and software stack a re-run produces the same checkpoint and
the same `metrics.json`, bit for bit, and a re-run after a change that
shouldn't alter the model (a refactor) is a regression test: anything but an
unchanged `metrics.json` means it did. Since identical checkpoints share a
cache entry, an unchanged re-run doesn't grow the cache either; a real change
adds one entry per checkpoint version, which `uv run dvc gc --workspace
--all-commits` prunes to what some commit references (not `--cloud` without
thinking: that deletes from the remote too). `metrics.json` is still rounded
to 6 significant digits, as a margin against float noise from other sources.
(Use `dvc repro -f -s train` to force a refit: `-f` alone re-runs the whole
chain leading to the stage, data stages included.)

**When to re-run, and what it doesn't touch.** Re-running on a new commit
never changes an earlier result: every commit's `dvc.lock` pins the exact
data, code, config and checkpoint, and its `metrics.json` holds the scores.
Re-run in the PR whose change touches a stage's deps (the lock test insists),
so its `metrics.json` diff sits next to the code that caused it.

**Division of labor with MLflow.** MLflow is the experiment record: every
run, sweep and model, with its metric histories -- exploration happens there,
through Hydra multiruns and Ray Tune. DVC answers a different question for
the few canonical artifacts: exactly what produced this, and is it still up
to date? `dvc exp run` is deliberately not used for sweeps, so there aren't
two experiment systems to look in. The final evaluation (refit the finalists
on train + val, score them on test, see "How validation and test are used")
will become stages too, added only once the finalists are frozen and marked
`frozen: true`, so `dvc repro` can never re-read the test set by accident.

### Reproducibility

"Reproducing our results" means three different things, and the pipeline is
built to deliver each to the extent it can be delivered:

1. **Our numbers, from our model.** `git checkout <commit> && uv run dvc
   pull` restores that commit's exact data, stats and checkpoint; `uv.lock`
   (or the Docker image) the environment; `uv run dvc repro evaluate`
   re-scores the checkpoint. On our GPU that gives `metrics.json` bit for
   bit. On another GPU the kernels differ, so the bits do too, but scoring
   a fixed checkpoint is numerically tame: across four differently compiled
   runs of the noise-0.1 U-Net, RMSE at lead 10 varied in the 4th digit
   (0.4154-0.4170), and the skill horizon and stable steps not at all.
2. **Our model, from our recipe.** Re-running `train` elsewhere gives a new
   draw, not our checkpoint: a different GPU (or cuDNN, or jax version)
   rounds differently, and an autoregressive network trained on a chaotic
   flow amplifies that into a different model, the way another seed would.
   No deep-learning framework promises more; this one doesn't either.
3. **Our conclusions, from our recipe.** What has to hold for any such draw
   is the claim the decisions rest on -- e.g. "input noise 0.1 makes the
   U-Net stable" -- and that is a statistical statement: it is backed by
   several seeds, not by one checkpoint's score, and a retrain elsewhere
   should land within that seed-to-seed spread. Which is also why the
   number of seeds matters: 3 of 3 stable turned into 4 of 5 with two more
   seeds and one deterministic retrain (see "Robustness across seeds").

Bit-for-bit reproduction on our own machine (point 1 and the deterministic
stages) is what makes `dvc repro` a regression test and `metrics.json` diffs
meaningful; points 2 and 3 are what someone else can check, and the
committed checkpoint is the artifact that makes point 1 possible for them at
all. "Our own machine" means its whole software stack, GPU driver included:
after a driver update (to 595.99.02, 2026-10-19) the deterministic `train`
stage gave a new Hankel DMD checkpoint, its scores moved in the 6th digit
and it still passed the gate, while cold compiles under the new driver again
agreed with each other bit for bit.

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
   data left), so it uses the epoch count from its tuned run, unscaled: the
   stable fraction that vouches for that epoch count was measured per epoch,
   and the validation dataset lengthens an epoch by only 13% (917 frames on
   7115). An iteratively trained finalist is refit five times, with seeds
   0-4, fixed before any test score exists. One refit is one draw, and a
   stable validation epoch is often a lucky one (see the stable fraction
   below): the U-Net baseline's is 0.2, so a single refit of it would come
   out stable about one time in five, and on that draw the table would show
   it beating Hankel DMD against all the validation evidence; on the others,
   a rerun with another seed would be choosing after seeing test. A
   closed-form fit (DMD, the mean field) is the same model on every refit,
   so it is fit once. The normalization stats are recomputed over
   train + val as well: normalization is just the first step of fitting a
   model, and fitting it on less data than the rest of the model would need a
   reason of its own. (During development they stay train-only, as the
   `compute_stats` stage computes them now; the train + val stats will be a
   separate output when the final evaluation is built.) The baselines are
   refit too (the mean field over all 8 datasets), so the comparison stays
   fair.
3. Score all of them on test in one evaluation, and report that table. For
   an iterative finalist it gives the share of its seeds stable on test
   (beside its validation stable fraction, the prediction) and each score's
   median over the seeds, every seed's scores listed with it; a mean would
   be meaningless once one seed's energy error is in the hundreds. No seed
   is picked. A generative finalist's seeds are training seeds; each
   refit's ensemble is scored as on validation. Nothing is changed or chosen
   after seeing it.

Reporting every finalist's test score side by side is the point of step 3;
what would spoil the test set is going back to iterate after looking at it.

**How candidates are ranked.** A surrogate is useful for as long as its
forecast stays close to the truth, and only if it stays bounded after that.
The primary criterion is the validation **stable steps**: how many leading
steps of the forecast stay bounded (see "Forecast evaluation"; the whole
837 for a stable forecast). Then, for a stable candidate of an iterative
(neural) training run, the **stable fraction**: the share of the 5
validations nearest its epoch (2 either side, shifted inward at the ends
of the run; `evaluation.stability.neighbors`) that are stable too. Then the
validation **skill horizon** (lead steps with CRPS <= 0.32), ties broken by
**CRPS at lead 10**. Stability comes
first because the short-horizon scores can't see a blow-up: the first
U-Net search picked, by skill horizon alone, models whose energy is 1000x
the truth's by the end of the validation forecast (see "First search").
Among stable forecasts, which is what a usable candidate has to be, the
rule is stable fraction, skill horizon and CRPS at lead 10; among unstable ones,
lasting longer wins, then straying less: the **peak ratio** (the
forecast's largest 100-step block mean of energy or enstrophy over the
truth's), lower first, in place of skill. Most unstable forecasts last 0
or 100 steps (362 of the U-Net runs' 524 unstable validations) while
their peaks range from 3x to 60,000x; skill can't tell a forecast that
strays 3x from one that blows up 1000x, and optimizing it among unstable
models is what picked the first search's blow-ups. So early stopping and
the search are pushed toward stability even while nothing is stable yet.
A stable forecast's peak ratio isn't used: below the truth's range it
means damping, which the stability check is deliberately one-sided about. The stable fraction is there
because a stable epoch is usually a lucky one: the median U-Net run was
stable on the validation forecast at 2 of its 30 epochs, mostly isolated
ones, and only 1 of 21 runs ended on a stable epoch. Keeping the best epoch
picks such a draw, and the final refit (above) trains for that many epochs
with no validation left to pick with, so each of its seeds lands on
whatever that epoch gives. The fraction around the kept epoch is how often its neighborhood
gives a stable network. It does not predict long rollouts: of the U-Net
checkpoints kept by stable steps and skill alone, seven isolated ones
(fraction 0.2) passed the 3000-step rollout and three of the EMA runs'
long stable stretches (0.6) didn't, so the rollout guardrail below stays.
A closed-form fit (DMD) has no epochs: it is the same model on every refit.

The skill is measured on the **CRPS** (2026-10-27; it was the RMSE
before, `<= 0.5`, and the skill horizons in the result tables further down
that predate it are RMSE ones). The two never share a name in MLflow: runs
before the change logged the RMSE one as `skill_horizon`, runs since log
`crps_skill_horizon`, so a comparison across the change can't happen
silently (the threshold is a run param too, `evaluation.skill_threshold`).
Older runs with a logged `crps` curve (the U-Net runs since the ensemble
scores) can be compared from it; others need re-scoring. For a deterministic model the CRPS is the
absolute error; for a generative one it scores the ensemble. Pointwise
error penalizes a realistic forecast twice once it has lost the truth's
phase -- for missing the truth's features and for placing its own where
the truth has none -- while a forecast smoothed toward the mean pays once:
the U-Net's RMSE saturates at ~1.45, Hankel DMD's damped forecast's at
~0.82. A member's RMSE would rank a generative model's realistic samples
below the smooth forecasts they're meant to beat; the CRPS is proper, its
spread term credits an ensemble whose members disagree as much as they
miss, and two members estimate it without bias. 0.32 is half the mean
field's CRPS (0.64 over the first 40 leads), as RMSE 0.5 was about half its
RMSE (0.9). The absolute error of the logged U-Net and Hankel DMD runs is
0.51-0.65x their RMSE at their RMSE horizon, so at 0.32 Hankel DMD keeps
its horizon of 7 and the U-Nets gain 0 to 3 steps (their sharper forecasts
have heavier-tailed errors, which the RMSE squares). Persistence now has
one step of skill (CRPS 0.28 at lead 1, RMSE 0.51), the mean field none.

A **generative** model is judged as an ensemble, by the same rule. Its
skill horizon and tie-break are its ensemble's CRPS. The absolute
guardrails -- stable over the validation forecast, stable over the
3000-step rollout, the domain-mean velocity in the truth's band -- must
all hold for at least 80% of its members, 7 of 8 when it is scored
(`evaluation.gate.min_member_fraction`, the stable fraction's floor): a
member that fails any of them is a broken forecast, served to whoever asks
for its seed, so a member counts only if it passes all three (counted
check by check, one member blowing up and another drifting would leave 6
usable forecasts of 8). Were each member broken with probability q, a q = 0.05
model would pass 94% of the time (requiring all 8 would fail it a third of
the time), q = 0.2 50%, q = 0.375 14%; the median member, 5 of 8, would
let that last one pass 65% of the time. Every other per-member score --
stable steps and peak ratio for ranking, the relative guardrails (energy,
enstrophy, spectra, a typical member against the incumbent) -- is the
median over its members, so no single member, lucky or blown up, decides.
Its stable fraction pools the members of the validations around its epoch
(a stable sample is as much a lucky draw as a stable epoch). During
training each validation forecasts 2 members
(`evaluation.monitor_members`, seeds 0 and 1 at every validation, so
epochs differ only by the network), the fewest the CRPS is unbiased from,
at twice a single forecast's cost; the stable fraction then counts 10
forecasts, and 0.8 means 8 of them. A closed-form generative fit's stable
fraction is the share of its members stable. The evaluation writes each
member's own validation scores as `val.member_<seed>` (in MLflow and
`metrics.json`), which the gate checks member by member.

The mean RMSE over all leads is deliberately not used:
after a few dozen steps every forecast of this chaotic flow has decorrelated,
and from there pointwise error favors whatever sits closest to the mean -- in
the DMD rank sweep it ranks the smoothest, lowest-rank model first. The
physics scores are **guardrails**, not objectives (optimizing one directly
invites gaming it): a candidate replaces the incumbent only if it is stable
over the whole validation forecast and over a 3000-step rollout from the
validation context (`check_rollout_stability.py`: no 100-step block over 2x
the truth's largest energy or enstrophy block mean; a stable validation
forecast alone can be luck, see "Input noise"), its stable fraction is at
least 0.8 (at least 4 of the 5 validations nearest its kept epoch stable,
so a refit for that many epochs likely is too; were each epoch stable with
probability p, a p = 0.9 recipe passes 92% of the time where a limit of 1
would reject it 41% of the time by chance, and a p = 0.5 one 19%; a
closed-form fit scores 1 if stable, being the same model on every refit),
its energy and
enstrophy errors are no more than 0.03 worse and its x/y spectrum distances
no more than 0.05 worse. Those tolerances are about the spread a different
realization of the flow scores (energy <1%, enstrophy <3%, spectrum
0.01-0.03, see "Forecast evaluation"), so a candidate isn't rejected for a
difference that is noise. Finally, its domain-mean velocity stays in the
truth's band, in each channel: `u_x_mean_offset` and `u_y_mean_offset` at
most 1, `u_x_mean_std_ratio` and `u_y_mean_std_ratio` at most 2. Those
limits are absolute, not relative to the incumbent: a limit relative to an
incumbent that damps the domain mean (Hankel DMD: 0.19 and 0.17 on `u_x`,
0.07 and 0.41 on `u_y`) would fail a model with the truth's own
variability. They are in units of the truth's own variability over time,
which is small: its domain-mean `u_x` stays at 1 +- 0.0042 and `u_y` at
0 +- 0.0055 (against per-pixel stds of 0.82 and 0.44), an empirical
property of this flow -- the domain has an inlet and walls and is a slice
of a 3D run, so no conservation law pins them. `u_y`'s fluctuation is the
~25-step oscillation the EDA found; `u_x`'s has no such peak, only a broad
low-frequency bump, so an 837-step time mean of it is noisier. A different
realization scores at most 0.15 and 0.98-1.02 on `u_y`, at most 0.33 and
0.76-0.96 on `u_x` (all 7 train datasets against the validation truth),
so the limits sit 3-7x and 2x above that noise floor. On the scale of the
field they are strict (1 truth std is ~0.5% (`u_x`) and ~1.2% (`u_y`) of
a point's typical velocity, where the energy error tolerates tens of
percent); the truth's band is the scale chosen because a drift beyond it
swamps the oscillation in `u_y`, the one coherent temporal signal
established in this flow, and is a wrong bulk flow in `u_x`. The energy
error sees neither: `u_y`'s mean enters it squared, and the U-Net baseline
loses 7% of its bulk flow (domain-mean `u_x` 0.930, 17 truth stds low)
with an energy error of 1%, the lost mean-flow energy made up by extra
fluctuation energy. Outside the band a forecast carries a spurious net
flow (see "Input noise"); the limits are one-sided like the stability
check, since a still mean is harmless. The
other oscillation scores are reported but not used to decide: on a single validation realization their noise (period error
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

A cache hit reuses exactly the executable some earlier cold compile built
-- which is not the same as saying every cold compile builds the same one.
For Hankel DMD it does (its `train` stage reproduces byte-identical outputs
with a cold and a warm cache), but on the GPU XLA *autotunes* convolutions
and matmuls at compile time, timing candidate kernels that don't all round
the same way, and which one wins varies from compile to compile. For the
U-Net that's visible from the first forecast step: four cold compiles of
one checkpoint gave RMSE at lead 10 of 0.4154, 0.4168, 0.4168 and 0.4170,
while every process reusing one warm cache entry reproduced 0.4164 bit for
bit, and two cold-compiled 2-epoch trainings of the same config already
disagreed on the validation skill horizon (8 vs 9). A warm cache hides
that on one machine but not across machines, cache evictions or jax
upgrades.

**Deterministic GPU kernels** (`jax.deterministic_ops`,
`utils/jax_determinism.py`): `--xla_gpu_deterministic_ops=true` restricts
XLA to deterministic kernels, so every compile on a given GPU and software
stack produces the same bits -- measured: identical checkpoints and
forecasts across cold compiles, for ~50% more time per U-Net training
epoch (47 -> 71 s) and ~8% per forecast (5.1 -> 5.5 s). Turning autotuning
off instead (`--xla_gpu_autotune_level=0`) falls back to kernels so slow an
837-step forecast hadn't finished after 17 minutes. Off by default, which
keeps exploration (sweeps, searches) fast; on for whatever must reproduce
(the DVC stages that produce the canonical model and its scores). It is
part of the cache key, so the two kinds of compile never share an entry.
Bit-identical results across *different* GPU models are out of reach
either way.

XLA's own per-fusion autotuning cache, which JAX keeps in the same
directory, is *not* keyed on the flag, so deterministic runs turn it off
(the compilation cache stays on). Found the hard way: after a smoke run of
the POD codec (an autotuned compile of the same Gram matrix matmul) the
`train` stage's Hankel DMD came out ~1e-6 off the canonical checkpoint --
its deterministic compile had reused the kernel the autotuned one stored
there. With the autotuning cache off it reproduces the checkpoint bit for
bit, whatever an exploration run left in `.jax_cache/`.

So the cache stays a performance knob: deleting `.jax_cache/` is always
safe, and with deterministic ops on, a cold and a warm cache give the same
outputs. JAX reads the cache config on its first compile and XLA its flags
when the backend starts, so entry points set both before anything is
jitted; `jax.compilation_cache_dir=null` turns the cache off.

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
Models column and gets its own page in the UI, with the model config and its
parameter count (`n_parameters`, see "Baselines") as its params and every
`val.*`/`train.*` score linked to it (the Models tab compares
models directly). The logged model is self-contained -- the checkpoint as an
artifact, this package's source as `code_paths`, pinned `numpy`/`jax`/`mlflow`
requirements, and a signature (a stack of `(C, Nx, Ny)` float32 frames in and
out, `n_steps` as a parameter) -- so it loads anywhere with

```python
model = mlflow.pyfunc.load_model(f"models:/{model_id}")
forecast = model.predict(context, params={"n_steps": 837})  # last `window` frames are used
```

and `mlflow models serve` can serve it, which is what the serving step will
build on. A checkpoint is logged once per experiment: a run whose checkpoint
is identical to an already logged model's (same `checkpoint_sha256`, which
every run tags itself with) reuses that model instead of storing another
copy (`training/mlflow_utils.py`'s `log_checkpoint_once`; codecs too) -- it's
tagged `reused_logged_model`, lists the model among its outputs and links
its scores to it. That's every re-run of the deterministic DVC `train`
stage, which had stored another ~900 MB copy of Hankel DMD's checkpoint each
time (seven identical copies of the current one by 2026-10-21). The model belongs to the run that logged it, so pruning
that run's checkpoint also takes it from any run reusing it. Models are not registered in the Model Registry automatically:
registering every experiment would bury the meaningful versions, so only the
shortlist and the test-set finalists (see "How validation and test are used")
get registered, deliberately. The model that is served is registered in a
registry of its own, on the tracking server (see "Promotion"). No input example is logged: one real frame is
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

- *Short horizon* (`metrics.py`): RMSE and CRPS per lead time, optionally
  scaled per channel by the training std, and the skill horizon: how many
  steps the CRPS stays under a threshold (0.32), the CRPS of a deterministic
  model being its absolute error (see "Ensembles" below and "How candidates
  are ranked"). A NaN error counts as over it, so a forecast that blows up
  loses its skill there instead of scoring the whole length.
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
  period (`nan`) and a peak ratio of 0; so does one whose spectrum peaks in
  the lowest bin where the truth's doesn't -- a slow drift, not a period
  (every U-Net's used to read 200 / 25.6 - 1 = 6.806, see "Input noise").
  That drift is measured directly: `u_y_mean_offset` is how far the time
  mean of the forecast's domain-mean `u_y` lies from the truth's, and
  `u_y_mean_std_ratio` how much it varies, both in units of the truth's own
  variability over time (~0 and ~1 for a forecast on the truth's statistics);
  `u_x_mean_offset` and `u_x_mean_std_ratio` do the same for the bulk
  streamwise flow. Unlike the other oscillation scores they are guardrails
  (at most 1 and 2, see "How candidates are ranked"). SPOD would give mode shapes rather than
  a score (it needs mode matching between prediction and truth and an
  eigendecomposition per frequency), so it stays a plotting diagnostic.

**Long rollouts** (`scripts/evaluation/check_rollout_stability.py`). The
stability score sees only the scored forecast, 837 steps of the validation
dataset, but a surrogate is meant to be rolled out for as long as one likes.
The check rolls a model out for `--steps` steps (1500 by default) from the
dataset's context, one block at a time (only one block in memory), and
prints each block's mean energy and enstrophy and their ratio to the
truth's largest block mean -- the selection rule's reference, so its first
blocks reproduce `stable_steps`. A stochastic model is rolled out once per
member (`--members`, the evaluation's 8 by default), a summary line per
member; at least 80% of them (7 of 8) must stay stable, and pass the other
absolute guardrails as well:

```bash
uv run scripts/evaluation/check_rollout_stability.py --checkpoint models/hankel_dmd/model
uv run --extra gpu scripts/evaluation/check_rollout_stability.py --model-id m-... \
    --steps 5000 --block-steps 500
```

The blocks come from `rollout.py`'s `rollout_blocks`, which streams any
forecast too long to hold at once (a served one too). A model with
`predict_blocks` streams the forecast its one `predict` call would make,
so its long rollouts are exactly the forecast it's scored on. The neural
networks carry their window from block to block as they see it, normalized
and padded; (Hankel) DMD carries its mode amplitudes. Any other model is
restarted at every block from the last `window` frames of the forecast so
far, which is the same forecast only if those frames are all of its state; a
restarted stochastic model samples each block with its own seed (the
member's for the first, then one derived from it and the block). Hankel DMD
shows how far off a restart can be: its forecast frames are the newest
blocks of its delay states, whose older blocks needn't be the frames before
(the operator isn't constrained to shift them), so restarting from its own
last frames starts from another state, ~1e-4 of the field's RMS off on the
validation dataset (a rounding error would be ~1e-7). A model whose state is
more than its last frames -- a recurrent or reservoir network's hidden
state, a generative model's random key -- implements `predict_blocks`, or
its long rollouts would quietly be a different forecast from the one it's
scored on. The test dataset is refused, like in the forecast videos.

**Ensembles** (`protocol.py`, `ensemble.py`). The deterministic models
so far have one answer per context; a generative surrogate (flow matching,
diffusion) samples one, and its forecast is a distribution. The contract is
ready for it without changing anything for the others: a model with
`stochastic = True` takes `predict(context, n_steps, seed=...)` and returns
its sample for that seed, an ensemble is the samples for seeds 0, 1, ...,
`evaluation.ensemble_size` (8), and a deterministic model is a one-member
ensemble. Every score of a trajectory above -- RMSE, the long-horizon and
temporal scores, stability -- is the median over the members (each one
trajectory, as `predict` gives it; the ensemble *mean* would be smooth and
fail the physics), a member that produced NaN counting as the worst, and
each member's own scores are kept too: the absolute guardrails count the
members that pass (see "How candidates are ranked"). The ensemble adds:

- *CRPS* per lead time (`crps_mean`, `crps_lead_<n>`): E|X - y| - E|X -
  X'|/2 over members X, X' and truth y, per pixel. A proper score -- it is
  minimized by forecasting the true distribution -- so unlike a member's
  own error it doesn't reward collapsing toward the mean. For one member it
  *is* the absolute error, so deterministic and generative models share a
  score, and the skill horizon is measured on it.
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
pairwise terms use the cycle of members, consecutive pairs plus the one
from the last back to the first, each an independent pair, so they're
unbiased like all M (M - 1) / 2 pairs, with some more variance. Like all
pairs, and unlike the consecutive ones alone, the cycle keeps the CRPS from
going negative: each member is in two pairs, and |X - X'| <= |X - y| +
|X' - y|. With consecutive pairs only, one blown-up member among 8 would
drive the CRPS far below 0 at every lead after it, and a broken ensemble
would look skillful. MLflow gets the per-lead `crps` curve (and `spread`,
`ensemble_mean_rmse` for an ensemble), and the logged pyfunc model takes
`seed` as a parameter next to `n_steps`, so a server can sample an ensemble
one call per seed. How an ensemble is ranked is in "How candidates are
ranked".

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

`parameter_count()` answers "how many parameters does it have": the numbers
the model learned from the training data and forecasts with -- a network's
weights, DMD's modes and dynamics (complex entries count twice), the mean
field's field. Normalization statistics don't count, as a network's input
normalization usually doesn't. Every run logs it as `n_parameters`, both as a
metric (a sortable column in the runs table) and as a param of the logged
model, next to the model config it follows from.

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
"Experiment tracking (MLflow)"), `fit_seconds`, `n_parameters`, every scalar
score as a `val.*` metric (and `train.<dataset>.*`, below), and the RMSE
curve as the `val.rmse` history with step = lead time (one batched request
rather than 837). Scores that are undefined for a model (the oscillation period of a
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
| CRPS (absolute error) at lead 1 / 10 / 40 | 0.28 / 0.98 / 0.84 | 0.64 / 0.64 / 0.65 | -- |
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

Neither baseline had any skill at the RMSE threshold of 0.5 (persistence
was already at 0.51 one step ahead); on the CRPS, at 0.32, persistence has
one step (0.28) and the mean field none. Persistence beats the mean field only at lead 1 and
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
~890 MB, the rank-750 spatial basis plus a 12 MB delay basis. That makes it
the largest model here by parameter count: 224M, 98% of them the spatial
basis (292k entries x 750 modes), against the default U-Net's 7.8M. A
linear model of a 292k-entry state pays for its modes in full resolution;
the network shares its 3x3 kernels across the grid.

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
- *Weight averaging* (`training.ema_decay`, off by default): an exponential
  moving average of the weights, updated after every optimizer step, is
  what's validated, early-stopped on, checkpointed and returned; the
  optimizer keeps training the raw weights. The raw weights move with every
  noisy batch, and for an autoregressive model that's enough to flip a long
  rollout between stable and unstable from one epoch to the next (see
  "Input noise"); the average moves smoothly. It's part of the resumable
  training state.
- *Epochs* are `samples_per_epoch` windows (2048 of ~7k by default) drawn
  without replacement. An epoch is the unit of validation and
  checkpointing, so it's kept short enough that early stopping (and the
  hyperparameter search's pruning) can follow the curve.
- *Early stopping on validation:* every `eval_every` epochs the model
  forecasts the whole validation dataset under the protocol, and is scored
  with `evaluation.selection_scores`: stable steps, skill horizon and CRPS
  at lead 10, folded into one `selection_score`,
  `(n + 1) * stable_steps + skill - c / (1 + c)` for an `n`-step forecast,
  which orders like the rule exactly (one more stable step outweighs any
  skill, one more skill step any CRPS); an unstable forecast scores
  `(n + 1) * stable_steps + n / (1 + log10 peak_ratio)` instead, so one
  more stable step still outweighs any peak. The validations are then ranked by
  `evaluation.selection.RunSelection`, which judges each with its nearest
  validations: a stable one's score gains its stable fraction times
  `(n + 1) * m^2` for windows of `m` forecasts, enough for a larger
  fraction to outweigh any skill and CRPS. A validation's score is
  provisional until its 2 later neighbors are in, so its network is kept
  until then (with the best one already final); a stable epoch can still
  lose its lead to an unstable one after it. Training stops once the best
  validation so far is `patience` validations old, and the best one's
  network is the result. These are the same scores models are compared by,
  so early stopping is another decision made on validation. The physics
  guardrails are left out because they're too slow to run every epoch; the
  final model is scored in full like every model.
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
`val_monitor.{crps_skill_horizon,crps_lead_10,stable_steps,energy_peak_ratio,enstrophy_peak_ratio,selection_score,forecast_steps,members,stable_members}`
every epoch, with the run's best validation so far (`best_epoch`,
`best_selection_score`, `best_stable_fraction`). After
training, the model is checkpointed, logged and scored like every model
(`val.*`, `train.<dataset>.*`, `fit.*`, where `fit.best_epoch`,
`fit.stable_fraction` and `fit.stopped_early` come from training), and
`n_parameters` is the network's weights (runs before #68 logged it as
`fit.parameters`).

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
plus input noise 0.1 and the full 30-epoch schedule for every run (patience
30, so early stopping never ends a run: stable steps flip between epochs,
and an early stop keeps an early lucky epoch and misses the annealed ones;
the best epoch is still the one kept). The
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
  to Tune after every validation, with its best validation so far as early
  stopping ranks it (stable steps, then stable fraction, then skill horizon
  with CRPS at lead 10 as tie-break), and the search maximizes that,
  `best_selection_score`. The search optimizes what models are
  chosen by, on validation only. A trial's best, not its latest score: a
  trial keeps its best epoch, and the score flips between epochs (stable
  steps come in 100-step blocks). On the latest score, ASHA judged a trial
  by whatever its validation at the rung happened to score (trial 5 of the
  first search, below), and Optuna's TPE learned each trial's final epoch,
  since Ray hands the searcher a trial's last result, not the epoch the
  trial keeps. With the stable fraction in it, the best can't be one lucky
  epoch either; and since a later validation can still lower a best whose
  neighbors aren't all in, a trial's result is its last report
  (`get_best_result(scope="last")`), not its highest. The result table
  shows each trial's best score and stable fraction.
- **Optuna proposes, ASHA prunes.** Optuna's TPE sampler (seeded) models
  which regions of the space score well, from the trials so far, and samples
  the next configuration there instead of uniformly at random. ASHA
  (asynchronous successive halving) compares trials at rungs of 4, 12 and 36
  validations (`grace_period` x `reduction_factor`^k; with the default 30
  epochs, 4 and 12) and stops those below the top third of what reached the
  rung. Most of the budget goes to promising configurations; a bad learning
  rate is dropped after 4 epochs, not 30. Both are needed: ASHA alone samples blindly, Optuna alone trains
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
  *Since fixed:* Tune compares trials by their best score so far, so a dip
  at the rung no longer prunes a trial whose earlier epoch was good (see
  "One metric, the selection rule" above).
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
drift far less; see there. Since then the period error is `nan` for a drift
and `u_y_mean_offset` / `u_y_mean_std_ratio` measure it, see "Forecast
evaluation".)

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
rules); the kept ones were noise 0.1 at seeds 42, 0 and 1 and noise 0.03 at
seed 0 (the stable candidates). Since the robustness sweep below, only noise
0.1, seed 0 is kept.

#### Robustness across seeds (2026-10-08)

"3 of 3" was a small sample, and the next draw broke it: a retrain of noise
0.1, seed 0 with deterministic GPU kernels (the way the DVC stage would
train it, a different trajectory from the same seed) came out stable for
only 400 of the 837 validation steps, at every epoch. So the switch to the
U-Net was dropped, and two candidate fixes for the recipe's robustness were
tried against more seeds of the default, all with the full 30-epoch
schedule and every selected checkpoint rolled out for 3000 steps (sweeps
`14-45-17`, `16-50-42`, `18-55-12`):

- **Weight averaging** (EMA of the weights, decay 0.999, validated and kept
  instead of the raw weights; `model.training.ema_decay`, added for this):
  averaging over the late epochs' flips in and out of stability might
  land in between.
- **More noise** (0.2): a wider margin from the edge of stability.

| setting | seed | selected epoch | stable steps (val) | skill horizon | RMSE lead 10 | energy / enstrophy error | spectrum distance (x / y) | `u_y` mean offset / std ratio | stable steps (train dataset) | 3000-step rollout |
|---|---|---|---|---|---|---|---|---|---|---|
| noise 0.1 + EMA | 0 | 24 | 837 | 12 | 0.449 | 0.127 / 0.627 | 0.78 / 0.78 | 12.0 / 5.9 | 300 | unstable from 400 (energy 250x) |
| noise 0.1 + EMA | 1 | 11 | 837 | 12 | 0.431 | 0.189 / 0.336 | 0.96 / 0.95 | 0.30 / 2.2 | 1168 | over 2x from 1700 (enstrophy up to 9.3x) |
| noise 0.1 + EMA | 2 | 18 | 837 | 12 | 0.449 | 0.173 / 0.547 | 0.78 / 0.78 | 9.0 / 8.1 | 800 | over 2x from 400 (enstrophy up to 3.3x) |
| noise 0.1 + EMA | 3 | 11 | 837 | 11 | 0.451 | 0.228 / 0.114 | 0.99 / 0.96 | 2.5 / 6.9 | 1168 | unstable from 900 (energy 78x) |
| noise 0.2 | 0 | 23 | 837 | 8 | 0.566 | 0.390 / 0.267 | 0.80 / 0.65 | 11.7 / 6.3 | 1168 | stable (energy <= 0.82x, enstrophy <= 1.81x) |
| noise 0.2 | 1 | 30 | 837 | 9 | 0.533 | 0.276 / 0.040 | 0.69 / 0.63 | 4.1 / 5.8 | 1168 | stable (<= 0.99x, <= 1.14x) |
| noise 0.2 | 2 | 15 | 300 | 7 | 0.661 | 2.59 / 7.21 | 0.57 / 0.57 | 85 / 52 | 300 | unstable from 300 (energy 548x) |
| noise 0.2 | 3 | 22 | 500 | 10 | 0.470 | 0.149 / 0.560 | 0.81 / 0.58 | 12.4 / 5.6 | 200 | unstable from 1200 (energy 15x) |
| noise 0.1 | 2 | 25 | 837 | 12 | 0.435 | 0.041 / 0.207 | 0.63 / 0.61 | 5.6 / 7.4 | 1168 | stable (<= 1.08x, <= 1.92x) |
| noise 0.1 | 3 | 9 | 837 | 9 | 0.554 | 0.364 / 0.380 | 0.86 / 0.68 | 1.7 / 2.7 | 1168 | stable (<= 0.76x, <= 0.71x) |
| noise 0.1, deterministic kernels | 0 | 30 | 400 | 12 | 0.439 | 0.829 / 3.62 | 0.56 / 0.66 | -- | 300 | -- (fails validation) |

Counting a draw as robust when its selected checkpoint is stable over the
validation forecast *and* the 3000-step rollout, with the full-schedule
runs of the previous sweep included:

- **Noise 0.1 (the default): 4 of 5** (seeds 0-3; the deterministic-kernel
  retrain fails). Through the other guardrails as well (against Hankel
  DMD's scores): **3 of 5** -- seed 3's best stable epoch is an early one
  (9), which damps the flow and misses the energy guardrail (0.364 against
  0.294).
- **Noise 0.2: 2 of 4**, at a lower skill horizon (8-9 instead of 11-12);
  1 of 4 through every guardrail.
- **EMA: 0 of 4.** Every averaged checkpoint is stable over the validation
  forecast and every one blows up in the long rollout. EMA doesn't change
  the training trajectory -- the raw weights train identically, the
  training losses of seeds 2 and 3 are bit for bit the plain noise-0.1
  runs' over all 30 epochs -- so
  this is a clean comparison: the raw weights of seeds 2 and 3 survive 3000
  steps, their averages don't. An average of weights that move in and out
  of stability passes the 837-step check and fails the long one.

Neither fix makes the recipe more robust; both make it worse, and noise
0.1 stays the default. And the domain-mean `u_y` drift, measured since
#61, is far outside the truth's band for every U-Net (offset 0.3-85 truth
stds, std ratio 2.2-8; a different realization scores <= 0.15 and
0.98-1.02, Hankel DMD 0.07 and 0.41): a spurious net cross-stream flow,
now a guardrail (see "How candidates are ranked") that no U-Net passes.
(The domain-mean `u_x`, measured later on the surviving baseline, drifts as
far: its bulk flow ends up 17 truth stds low, so the guardrail covers both
channels since.)

**Decision: Hankel DMD stays canonical**, and the U-Net stays the neural
baseline (noise 0.1, seed 0, run `2cbc89f9`; every other checkpoint of
these sweeps was pruned). A recipe that passes on 3 of 5 draws isn't
reproducible as a single training run; if a U-Net were to become
canonical, robustness would have to come from the procedure -- train
several seeds and select on validation, guardrails included (at 3 in 5 per
draw, three seeds give at least one passing draw ~94% of the time) --
which would make the DVC `train` stage a multi-seed one. With the drift
failing for every U-Net, that's not worth building for this architecture;
the next candidates are generative (flow matching), where seeds and
ensembles are part of the model anyway. The DVC pipeline's `gate` stage
(see "Training and evaluation stages") is what this episode added to the
pipeline: a retrain that comes out like the deterministic one now fails
`dvc repro` instead of becoming the canonical model.

## Codecs (latent spaces)

```bash
uv run --extra gpu scripts/training/train_codec.py                     # codec=pod, rank 750
uv run --extra gpu scripts/training/train_codec.py -m codec.rank=250,1500
```

Latent generative models -- latent flow matching or diffusion, say -- forecast
in a compressed latent space instead of on the 292k-entry grid. The map into
that space and back is a **codec** (`src/mhd_surrogate/models/codec.py`): an
`encode`/`decode` pair on raw frames, normalizing internally like a model.
"Codec", not "autoencoder", because the slot isn't one architecture: POD is a
(linear) codec, and so are an autoencoder, a VAE or a VQ-VAE.

- **Fitted on its own.** A codec is fitted, logged and referenced as a
  component of its own, not as a first stage inside a latent model's `fit`.
  Otherwise every trial of a latent model's sweep, and every seed, would refit
  it -- costly, and its own variance would confound comparisons between the
  latent models. Each run stays single-stage, so `FitHooks`, the trainer and
  resuming don't change. This is also how latent diffusion is usually built:
  the autoencoder is trained first and frozen.
- **Its own selection rule, deliberately narrow.** A codec forecasts nothing,
  so it isn't ranked by `selection_score`. A codec trained iteratively
  early-stops on validation reconstruction RMSE: with the frame given, the
  RMSE-optimal reconstruction is the frame itself, so the mean-reversion that
  makes forecast RMSE a bad selection metric doesn't arise. Choosing *between*
  codecs (latent size, architecture, a VAE's KL weight) is still choosing a
  forecaster, by the `selection_score` of a latent model trained on each:
  reconstruction RMSE always favors the larger latent space and says nothing
  about how forecastable that space is. The reconstruction's physics scores
  are reported, not gated: what the reconstruction loses, the latent model's
  forecasts lose too, and a latent model is held to the usual guardrails
  anyway, so a codec that loses too much fails through its latent models.
- **Its own entry point and experiment.** `scripts/training/train_codec.py`
  (root config `configs/train_codec.yaml`: the training config with a
  `codec/` group in place of the model) fits the codec on the train datasets,
  logs it as an MLflow model (`training/mlflow_codec.py`; its pyfunc
  `predict` is the reconstruction) and scores its reconstruction
  (`evaluation/reconstruction.py`) on validation and on
  `evaluation.train_datasets`: `reconstruction_rmse` (in training stds, as a
  forecast's RMSE) and the physics scores a forecast gets. A latent model
  references the codec by its logged model's id, `models:/<model id>`. It
  starts like the other entry points (`training/startup.py`: JAX config,
  data version, checked stats), and logs to its own experiment,
  `mhd-codecs`, so the forecasting models' runs table holds one kind of run.
- **Own registry and checkpoint format.** Codecs are registered by import
  path in `codec.py`'s `CODECS` (imported only when built or loaded, as models
  are), and a codec checkpoint has `codec.json` instead of `model.json`, so it
  can't be loaded as a model by mistake.
- **The final refit includes the codec.** The codec is part of fitting, so
  the final train + val refit refits it too (its epoch count, for an
  iterative one, fixed from its tuned run; with each of the latent model's
  seeds) before the latent model is refit on it.

**POD codec** (`models/pod_codec.py`, name `pod`): a frame's latent is its
coefficients in the leading `rank` POD modes of the training frames -- of all
rank-`rank` linear codecs, the one that reconstructs the (normalized)
training frames best, in closed form. It is the basis DMD's dynamics live in,
computed by the same code (`models/snapshots.py`: loading, normalization,
the Gram matrix and its POD on the GPU), but from all training frames
rather than DMD's "before" frames. Its training latents are uncorrelated,
each with its mode's variance; `decode(encode(x))` is the orthogonal
projection of `x` onto the basis, so what it loses is the trailing modes'
variance, the small scales, which no latent model on it can forecast. It is
the first codec because it needs no GPU training and exercises the latent
plumbing, and it makes a "POD + neural dynamics" baseline possible.

#### POD rank sweep (2026-10-21)

```bash
uv run --extra gpu scripts/training/train_codec.py -m codec.rank=100,250,750,1500
```

What a POD codec of each rank costs before any latent model is trained on
it (validation dataset; RMSE in training stds; runs in the `mhd-codecs`
experiment, sweep `08-53-54`):

| rank | variance kept (train) | reconstruction RMSE (val / train) | energy / enstrophy error | spectrum distance x / y | `u_y` peak power ratio / period error | parameters | fit |
|---|---|---|---|---|---|---|---|
| 100 | 90.2% | 0.286 / 0.281 | 0.032 / 0.336 | 1.73 / 1.52 | 0.90 / 0.003 | 29M | 47 s |
| 250 | 95.1% | 0.209 / 0.199 | 0.017 / 0.268 | 1.33 / 1.21 | 0.94 / 0.001 | 73M | 46 s |
| 750 | 98.2% | 0.138 / 0.119 | 0.007 / 0.178 | 0.84 / 0.81 | 0.95 / 0.002 | 219M | 47 s |
| 1500 | 99.2% | 0.108 / 0.081 | 0.004 / 0.129 | 0.56 / 0.55 | 0.96 / 0.001 | 439M | 67 s |

- **The codec isn't the bottleneck.** Even rank 100 loses less in its
  reconstruction than the canonical Hankel DMD's forecast does: its
  spectrum distance (1.73) and enstrophy error (0.34) are well under Hankel
  DMD's validation scores (2.62, 0.73), and the coherent `u_y` oscillation survives every
  rank (period error <= 0.003, 90-96% of its peak power). What a linear
  forecast loses, it loses in the dynamics, not in the basis -- so a
  nonlinear latent model on a few hundred POD modes has room to do better
  than DMD before the codec limits it.
- **Small scales are what the trailing modes hold.** Enstrophy (a
  gradient quantity) is lost much faster than energy (0.34 vs 0.03 at rank
  100), and the reconstruction's divergence is lower than the truth's at
  every rank (ratio 0.44-0.67): the DNS field's own small-scale divergence
  on this grid lives in the trailing modes.
- **The trailing modes start to fit the training realizations.** The
  train/val gap in reconstruction RMSE grows with rank (0.005 at rank 100,
  0.027 at rank 1500): beyond a few hundred modes the basis describes the
  training datasets' particular small scales better than an unseen one's.
- **Which rank to use is not decided here.** Per the codec selection rule,
  it is the downstream latent model's `selection_score` that picks a codec,
  not its reconstruction (which always prefers the larger rank). The four
  codecs stay logged for that comparison; at ~1.2 MB per mode each run
  stores 0.1-1.8 GB twice (logged model and Hydra output), ~6 GB in all.

### Latent models

```bash
uv run --extra gpu scripts/training/train.py model=latent_linear model.codec=models:/<codec model id>
uv run --extra gpu scripts/training/train.py model=latent_linear model.codec=models:/<id> model.window=4
```

A latent model (`src/mhd_surrogate/models/latent.py`, `LatentSurrogate`)
encodes its context with a codec, forecasts the next latents and decodes
them: raw frames in and out, so it is trained by `train.py`, scored, gated,
streamed and served like any model, and a new one supplies only its latent
dynamics (`fit_latents`, `forecast_latents`).

- **The codec by reference.** The config names the codec,
  `codec: models:/<model id>` of a `train_codec.py` run (or a codec
  checkpoint directory); being part of the model config, the reference is
  logged as the run's `model.codec` param. `fit` loads the codec frozen and
  encodes the training datasets with it; the latent model never changes it.
- **The checkpoint carries the codec.** `save` writes a copy of the codec
  into the checkpoint (`codec/`), so loading the model, its pyfunc model,
  serving and the DVC stages need nothing but the checkpoint -- the
  reference may since have been pruned. Its `parameter_count` includes the
  codec's: the model forecasts with both.
- **Streaming keeps the latent window.** `predict_blocks` carries the latent
  window from block to block. Restarting from its own decoded frames, the
  model would encode them again, which is the latent it forecast only for a
  codec with `encode(decode(z)) == z` (POD, up to rounding; not an
  autoencoder).

**`latent_linear`** (`models/latent_linear.py`) is the reference latent
model: a vector autoregression, the next latent a linear function of the
last `window` ones (centered on the training latents' mean), fitted in
closed form by least squares over pairs within each dataset, with an
optional ridge penalty. On a POD codec with `window=1` it is DMD without the
eigenvalue stabilization (a test checks it forecasts a known linear system
like DMD), and with more it's the latent analog of Hankel DMD without the
delay truncation; `fit.spectral_radius` and `fit.unstable_modes` show
whether a long forecast can grow. It is there to exercise the latent path
end to end, not as a contender; on the real data (a smoke run on the POD
codec at rank 750) it trains in ~30 s, and its checkpoint, codec included,
streams a stable 3000-step rollout.

## Serving

The canonical model is served from the MLflow Model Registry on the tracking
server (see "Tracking stack"), not from the experiment store it was trained
in, by a FastAPI service that streams forecasts, deployed to a local
Kubernetes cluster that Argo CD keeps in the state `main` describes, and
monitored by Prometheus and Grafana.

### Promotion

```bash
docker compose up -d                  # the tracking server holding the registry
uv run dvc repro                      # in the pull request: train, evaluate, gate
git commit dvc.lock models/ ...
uv run scripts/serving/promote.py     # register the gated checkpoint, pin it
git commit deploy/k8s/overlays/kind/kustomization.yaml
```

Making a model canonical and deploying it is one pull request: the one
that switches the canonical model or retrains it also promotes and pins it.
`scripts/serving/promote.py` puts the DVC pipeline's gated checkpoint
(`models/hankel_dmd/model`) into the registered model `mhd-surrogate` on the
tracking server as a new version (`src/mhd_surrogate/serving/promotion.py`)
and pins that version, together with the checkpoint's hash as `dvc.lock`
records it, in the overlay Argo CD deploys. It refuses anything but the
gated checkpoint:

- the pipeline up to the `gate` stage is up to date (`dvc status --with-deps
  gate` is empty), so the checkpoint, its scores and the gate's verdict on
  disk are the ones `dvc.lock` records, and the gate ran on that checkpoint;
- `dvc.lock`, `metrics.json` and `gate.json` are committed (the promotion
  run is tagged with the commit and its tree: on a pull request the head
  commit, whose tree is the one the squash commit on `main` gets);
- the gate passed.

**Why in the same pull request.** The server loads the checkpoint with the
image's code, and every merge deploys a new image. The gate checked one
pair, a checkpoint and the code at that commit (`test_dvc_lock.py` makes a
change to the canonical model's code retrain and re-gate it in the same
pull request). Promoting later, from `main`, left a window in which the
cluster ran the new code with the previous checkpoint, a pair nothing had
checked. Now CI checks that the pinned checkpoint is `dvc.lock`'s
(`test_the_deployed_overlay_pins_the_checkpoint_this_commit_gated`), so a
pull request that changes the canonical checkpoint can't merge without
pinning a registered version of exactly it, and the fetch refuses a version
not tagged with the pinned hash (`checkpoint_dvc_md5`). A retrain that
reproduces the checkpoint bit for bit (the stages run with deterministic
ops) changes neither the hash nor the pin.

**`champion` names what serves.** Promotion doesn't move the alias: the
deployment's post-sync check does, once it has verified the version serving
(see "Continuous deployment"). A version registered by a pull request that
never merged is never pinned and never gets the alias.

It also attaches the training data's normalization statistics to the
version (`reference_stats`), the reference the server measures input drift
against (see "Monitoring").

**Why a second store.** The experiment store (the local sqlite one) holds
every sweep's runs and models; the registry holds only what was promoted,
on the stack built to be shared (Postgres, S3 artifacts, an HTTP API a
cluster can reach) -- separate experiment and production registries, the
"deploy models" pattern. Open-source MLflow can't copy a model version
between stores (`copy_model_version` stays within one registry), so
promotion copies the logged model itself: it finds the experiment store's
logged model holding the checkpoint by content hash (`checkpoint_sha256`,
which every training run tags, see "Experiment tracking (MLflow)"),
downloads it, hashes the checkpoint again and uploads it as a new logged
model on the server. Only the identity fields of its MLmodel (model id, run
id, artifact location) are rewritten to the copy's; the checkpoint, the
code, the pinned requirements and the signature are the source's, byte for
byte. The registry serves the artifact that was trained and gated, not a
rebuild of it.

**Lineage.** The copy is logged by a run in the server's
`mhd-surrogate-promotions` experiment. The copy and its registered version
are tagged with where it came from (`promoted_from.tracking_uri`,
`promoted_from.model_id`, `promoted_from.run_id` -- the training run in the
experiment store); the run's params are the data version
(`data_version.*`, from `dvc.lock`), its artifacts the files that justified
the promotion (`dvc.lock`, `metrics.json`, `gate.json`), and the validation
and gate scores are metrics linked to the copy, so its model page shows what
it scored.

Promoting a checkpoint that is registered already (same `checkpoint_sha256`)
copies nothing: it keeps its version, and only the tags are refreshed.
Promoting the 865 MB Hankel DMD checkpoint takes ~9 s. A smoke run passes
`--no-pin` with a dated registered model and experiment (`--name
smoke-mhd-surrogate-<date> --experiment smoke-tests-<date>`), deleted
afterwards like any smoke run.

### Forecast API

```bash
docker compose --profile serve up -d       # fetch @champion, serve it on localhost:8000
curl -s localhost:8000/v1/model            # what is served
JAX_PLATFORMS=cpu uv run scripts/serving/check_parity.py   # served == offline forecast
```

A FastAPI service (`src/mhd_surrogate/serving/app.py`):

| Endpoint | |
|---|---|
| `POST /v1/forecast?n_steps=837&seeds=0&block_steps=100` | body: the context as `.npy` (`application/x-npy`), float32 frames in raw units; response: the forecast streamed as `.npy` records (`application/x-npy-stream`) |
| `GET /v1/model` | what is served: registered name and version, logged model, checkpoint hash, `window`, frame shape, whether it's stochastic, training and image commits, the request limits |
| `GET /healthz` | liveness |
| `GET /metrics` | Prometheus metrics (see "Monitoring") |

```python
from mhd_surrogate.serving.client import ForecastClient

client = ForecastClient("http://localhost:8000")
(forecast,) = client.forecast(context, n_steps=837)  # one member: (837, C, Nx, Ny)
```

- **Arrays as `.npy`, streamed.** A frame is 2x1151x127 float32 (1.17 MB),
  an 837-step forecast ~1 GB. JSON (or base64) would inflate that and hold
  the whole forecast in one document -- which is also why this isn't
  `mlflow models serve`, whose `/invocations` takes and returns JSON
  without streaming. The request body is one `.npy` record, the response a
  stream of them (`serving/stream.py`): the forecast's blocks as
  `rollout_blocks` produces them, member after member, so the server holds
  one block (100 frames, ~117 MB, by default), and a model's streamed
  forecast is the one it is scored on. The headers give the members, steps
  and block size (`X-Members`, `X-Steps`, `X-Block-Steps`) and the model
  version. NumPy reads the stream record by record; `serving/client.py`'s
  `ForecastClient` does, and raises if a stream ends before every frame
  arrived (the status is already 200 when a forecast fails mid-stream).
- **Validated input.** pydantic validates the parameters
  (`serving/schemas.py`), and the context is checked against the served
  model: float32, frames of its signature's shape, at least `window` of
  them (extra leading frames are ignored, as under the forecast protocol),
  at most 80 (the protocol's context length), finite. A refusal says why
  (400, 413, 415, 422). `.npy` records holding pickles are refused both ways.
- **The gate sets the horizon.** `max_steps` is the number of steps the
  model's gate validated a rollout stable for (`gate.rollout_stable_steps`,
  which promotion logs on the model; 3000 for Hankel DMD): the service
  doesn't forecast further than the model was shown to stay bounded.
  `MHD_SERVE_MAX_STEPS` overrides it, and is required for a model that was
  never gated.
- **Ensembles.** `seeds` are a stochastic model's members
  (`evaluation/protocol.py`'s `ensemble_size`); a deterministic model
  forecasts one member whatever the seeds, and says so (`X-Members: 1`).
- **One forecast at a time per pod.** A forecast keeps every core busy, so
  a server runs `MHD_SERVE_MAX_INFLIGHT` (1) at a time and answers more with
  503 and `Retry-After`; it scales by replicas, each one uvicorn worker (a
  worker holds the model). Forecasts run in a worker thread, so `/healthz`
  answers within milliseconds during one.
- **Fetched, not looked up.** The server never talks to the registry.
  `scripts/serving/fetch_model.py` (`serving/fetch.py`) resolves the model
  URI (a version, an alias or a logged model), downloads the logged model,
  checks its checkpoint against the version's `checkpoint_sha256` and
  records what it is (`serving.json`); the server only loads that
  directory, so it needs no registry credentials and serves the version it
  started with. In Kubernetes that's an init container, in compose the
  one-shot `serve-fetch` service before `serve`. The model is loaded before
  the server listens, so a server that answers `/healthz` is ready.
- **The image's code loads the model.** The checkpoint is loaded with the
  image's `models/registry.py`, not `mlflow.pyfunc.load_model`: a pyfunc's
  `code_paths` copy of the package only goes on `sys.path`, behind the
  `mhd_surrogate` the process has imported already, so it would never take
  effect (checked in MLflow 3.16). The image must therefore read the served
  checkpoint's format; `/v1/model` shows both commits.

Measured on the CPU (Hankel DMD, compose on this machine): fetching (865 MB)
and loading take ~20 s; a 3000-step forecast (3.5 GB) streams in ~11 s,
with the server at ~1.1 GB. The served 837-step validation forecast equals
the offline one to 5e-6 of the truth's standard deviation
(`scripts/serving/check_parity.py`, against a running service), the same
difference as between streaming that forecast in 100-step blocks on the
host and computing it in one call: the float rounding of computing it in
blocks, not of the server or the image.

### Serve parity

The server loads the checkpoint with the image's code, so the gate, which
checked the checkpoint with the training code, says nothing about the
serving code: a change to the app, the stream or the client could serve
another forecast than the model's while every check so far passed (CI's
smoke deploy serves a toy model; the post-sync check only that the forecast
is finite and shaped right). The DVC stage `serve_parity` checks it against
the real model instead (`src/mhd_surrogate/serving/parity.py`): it runs the
canonical checkpoint's validation forecast through the service in process
-- the app with the deployment's defaults, the `.npy` stream, the client
reading it back -- and compares it with the forecast of the evaluation
protocol, the one the scores and the gate are of.

- **The serving code is a dependency.** The stage's deps are what it runs
  (all of `serving/`, plus what it imports; checked like the other stages'
  by `tests/training/test_dvc_train_stage.py`), so a pull request that
  changes them leaves the stage stale until `dvc repro serve_parity` (about
  12 s) re-runs it, which CI's `dvc.lock` check enforces, and the committed
  verdict must say `passed` (`tests/data/test_dvc_lock.py`). The serving code
  is checked against the canonical model before it merges.
- **On the CPU**, as the deployed pod forecasts (`JAX_PLATFORMS=cpu` in the
  stage's command), where it is deterministic too: a re-run writes the
  same bytes.
- **The tolerance** is 1e-4 of the truth's standard deviation, anywhere in
  the forecast (`configs/serving/default.yaml`, `parity_tolerance`): about
  20 times the 5.2e-6 by which Hankel DMD's forecast in the service's
  100-step blocks rounds differently from one call (each block's matrix
  product has another row count; every frame is computed from the starting
  state, so the rounding doesn't compound), so a change to the block
  arithmetic passes and a change in what is served doesn't. Measured: a
  serving change that scales every frame by 1.001 fails it at 3.9e-3.
- **A neural model matches exactly.** Its rollout is chaotic, so a rounding
  difference at a block boundary would grow over the forecast's hundreds of
  steps, but there is none: its `predict_blocks` hands the next block the
  window as the network sees it (normalized and padded), and on the CPU a
  step's result doesn't depend on the block length. Measured on the U-Net
  baseline: a largest difference of exactly 0 over the 837 steps. A neural
  model that fails the stage serves another forecast for real (one
  restarted from its own output frames at every boundary would, late in a
  long forecast), to fix in its `predict_blocks`, not by loosening the
  tolerance. What a switch to a neural model does need to revisit is the
  time: the U-Net's check took 1 h 20 min on the CPU (two 837-step
  forecasts, about 2.9 s a step), and a CPU pod would take as long per
  request. Where such a model is served decides where the stage runs; on a
  GPU, which compiles each block length separately and doesn't reproduce
  bits across compiles, exact parity would have to be measured again.

It replaced a different design: storing a reference forecast with each
registered version and checking it when a pod starts. That would have
caught a bad pair only after the merge, on a 20-step sample, at the cost of
extra artifacts per version and a check at every start; the stage catches
it before the merge, on the whole forecast, with the machinery that already
guards the model.

### Image publishing

The serving image is the Dockerfile's `runtime` target without the extras
(no Ray, no tracking-server drivers: 1.9 GB instead of 2.3 GB), published
to GitHub's container registry as
`ghcr.io/michaelskowronek/mhd-surrogate-serve:<tag>` by the `docker`
workflow. The repository is public and so is the image, so a cluster pulls
it without credentials.

**Tagged by content, pushed before the merge.** The tag is a hash of the
image's inputs (`scripts/serving/image_tag.py`): the files the Dockerfile
copies into it (`src/`, `scripts/`, `configs/`, `pyproject.toml`,
`uv.lock`, `README.md`; a test checks the list against the Dockerfile), the
Dockerfile and `.dockerignore`, and the build arguments that change what's
in it, all as git's index has them. Not a commit hash, which covers the
whole tree and exists only once the commit does: the content hash is known
while the change is made, so the deployment can pin it in the same pull
request, and it is the same for the pull request's head and the squash
commit that lands on `main` (branch protection makes them the same tree).
The workflow pushes the image under it on the pull request, before the
merge, so by the time `main` changes, the image the deployment names exists
and is the one CI built from that code -- the tested artifact, not a
rebuild after the merge. A tag is never overwritten: if it exists, the
inputs are the same and the build is skipped. That also keeps a tag's image
fixed although the base image (`python:3.14-slim`) floats. The image is
labeled with the pull request's head commit (`MHD_GIT_COMMIT`, which
`/v1/model` reports as `image_commit`), a commit of the same tree as the one
on `main`. `README.md` is an input (`pyproject.toml` names it as the
package's readme), so most pull requests make a new image. Pull requests
from forks don't push: they get no token that can.

### Kubernetes

```bash
deploy/install-tools.sh          # kind, kubectl, helm, kubeconform, argocd, promtool, amtool: pinned, checksummed
docker compose up -d             # the tracking server the cluster fetches models from
deploy/kind/create-cluster.sh    # the kind cluster `mhd`
deploy/argocd/bootstrap.sh       # Argo CD; from then on the cluster follows main
curl -s localhost:8080/v1/model  # the forecast service (Argo CD's UI: localhost:8081)
```

A local cluster (kind, Kubernetes 1.37, one node) runs the forecast service
as a production cluster would, and Argo CD keeps it in the state `main`
describes (see "Continuous deployment").

**Manifests** (`deploy/k8s/`, Kustomize). `base/` is the service, free of
anything environment-specific: namespace `mhd-serving`, Deployment,
Service; `smoke-check/` the post-sync check (see "Continuous deployment").
The overlays pin what varies: `overlays/kind/` is the deployed one,
`overlays/ci/` CI's smoke deploy, built on it. Kustomize rather than a Helm
chart: one service in two environments is a few patches on plain YAML, and a
chart would add a templating language with nothing to template; Helm earns
its keep for third-party charts (monitoring, next). `deploy/validate.sh`
checks every rendered manifest against the Kubernetes 1.37 API (kubeconform,
strict: an unknown field is an error), in CI too.

**The pod.**
- An init container (`scripts/serving/fetch_model.py`) fetches the pinned
  registry version into an emptyDir and checks its checkpoint hash; the
  server container loads it from there, so only the init container talks to
  the registry. A restarted container finds the model in place; a new pod
  fetches it again.
- A startup probe covers the model load (up to 5 minutes), readiness and
  liveness probe `/healthz`, which answers during a forecast.
- One forecast at a time per pod (see "Forecast API"), so it scales by
  replicas. Requests 2 CPUs and 2 GiB, a 3 GiB memory limit, no CPU limit: a
  forecast uses every core it gets, and a limit would only throttle it.
- A rollout starts the new pod before the old one goes
  (`maxUnavailable: 0`), so the service answers throughout, and a new pod
  that can't fetch its model (the tracking stack is down) leaves the old one
  serving.
- Non-root, a read-only root filesystem (`/tmp` and the JAX cache are
  emptyDirs), no capabilities, the runtime's default seccomp profile.

**Three pins, all in git.** The kind overlay pins
- the image tag: the content hash of the image's inputs (see "Image
  publishing"), which the pre-commit hook `serving-image-pin` writes and a
  test checks (`test_the_deployed_overlay_pins_the_image_built_from_this_commit`),
  so a commit's overlay names the image CI built from that commit's code;
- the model version, `models:/mhd-surrogate/1`, not the alias: pods that
  resolve `@champion` as they start would serve two versions once the alias
  moves, a deployment change nobody reviewed. Through `configMapGenerator`,
  a new version is a ConfigMap with a new name, which changes the Deployment
  and so rolls it;
- next to it, the checkpoint's hash as `dvc.lock` records it
  (`MHD_MODEL_CHECKPOINT_MD5`): CI checks it is this commit's canonical
  checkpoint and the init container that the registry's version is tagged
  with it (see "Promotion").

**Reaching the tracking server.** The registry runs on the host (the compose
stack, port 5000). kind's node is a container on a Docker network whose
gateway is the host, so `create-cluster.sh` creates that network itself,
with a fixed subnet (172.30.0.0/16; kind uses an existing `kind` network as
it is), and the overlay maps `mlflow` to the gateway with `hostAliases` --
`mlflow:5000` being a host the server's `--allowed-hosts` already accepts.
A selector-less Service with an EndpointSlice would be the textbook way, but
Argo CD doesn't sync EndpointSlices (they're on its default exclusion
list). With the tracking stack in the cluster, its Service would replace
this.

**No Ingress.** kind maps the NodePorts to localhost (8080: the service,
8081: Argo CD, 8082: Grafana). ingress-nginx, the usual local ingress controller, was
retired in March 2026, and a Gateway API controller would route a single
service on a single machine.

Measured: the kind overlay rolls out in ~11 s (fetching the 865 MB model
from the compose stack and loading it); the smoke check and the parity check
(5e-6, as under compose) pass against `localhost:8080`.

### Continuous deployment

Argo CD (`deploy/argocd/`) runs in the cluster and pulls: it polls the
repository (every 3 minutes) and makes the cluster match `main`. Pushing
from GitHub Actions can't work here -- GitHub's runners can't reach a
cluster on this machine -- and pulling is the better design anyway: no
cluster credentials in CI, and a change made in the cluster by hand is
reverted (`selfHeal`).

- **App of apps.** `bootstrap.sh` installs Argo CD
  (`deploy/argocd/install`: the upstream v3.5.6 manifest, plain HTTP on
  localhost) and applies a single Application, `root`, whose manifests are
  the other Applications (`deploy/argocd/apps/`): `serving`, the kind
  overlay, and `argocd`, Argo CD's own installation, so upgrading it is a
  pull request too (never pruned: a broken render mustn't delete Argo CD).
  Two commands rebuild the cluster from git.
- **A rollout is a merged pull request.** Code: changing an image input
  re-pins the image tag (the pre-commit hook); on the pull request, CI
  builds and pushes that image and deploys it with the CI overlay to a
  throwaway kind cluster (`deploy-smoke` in the `docker` workflow: a
  throwaway tracking server with a tiny model,
  `scripts/serving/smoke_model.py`, then `scripts/serving/smoke_check.py`
  through the NodePort), and only then can it merge; Argo CD then rolls out
  that very image. Model: the pull request that makes a model canonical or
  retrains it runs `promote.py`, which registers the gated checkpoint and
  pins its version and hash in the kind overlay (see "Promotion"); merging
  it deploys the new code and the checkpoint it gated together. Rollback is
  `git revert`.
- **Checked after every sync.** A PostSync hook (`deploy/k8s/smoke-check`, a
  Job running `scripts/serving/smoke_check.py` in the cluster) checks the
  service as deployed: it answers, serves the version the overlay pins
  (`--expect-model-uri=$(MHD_MODEL_URI)`, from the same ConfigMap the pod
  fetched with) and forecasts. Then it points the registry's `champion`
  alias at that version, so `@champion` always names the version verified in
  service (for anyone reading the registry: notebooks, the compose `serve`
  service). A failed check fails the sync in Argo CD and leaves the alias
  where it was; it doesn't roll back (Argo CD doesn't), a `git revert` does. It lives apart
  from `base/` because the base's labels go into selectors, which a Job's
  must not have, and its pod mustn't carry the server's labels, or the
  Service would route requests to it. CI applies it too, without Argo CD,
  where it runs once.
- **What runs is what git says.** Every deployed image and model version is
  a line in `main`'s history, and Argo CD shows whether the cluster matches
  it.
- **Limits.** The cluster is this machine's, so deploys converge only while
  it runs; pointed at a real cluster, the same Applications would change only
  in how pods reach the tracking server.

### Monitoring

```bash
open http://localhost:8082        # Grafana: "MHD surrogate: forecast service", user admin, password:
kubectl -n monitoring get secret monitoring-grafana -o jsonpath='{.data.admin-password}' | base64 -d
kubectl -n monitoring port-forward svc/monitoring-kube-prometheus-prometheus 9090   # Prometheus
curl -s localhost:8080/metrics    # what the service exposes
install -D -m 600 /dev/null ~/.config/mhd-surrogate/smtp-password   # paste a Gmail app password into it
deploy/argocd/smtp-secret.sh      # Alertmanager's SMTP password, from that file (bootstrap.sh runs it too)
kubectl -n monitoring exec alertmanager-monitoring-kube-prometheus-alertmanager-0 -c alertmanager -- \
  amtool alert add TestEmail team=mhd-surrogate --end="$(date -u -d '+5 min' +%FT%TZ)" \
  --alertmanager.url=http://localhost:9093   # a test email
```

The service exposes Prometheus metrics at `/metrics`
(`src/mhd_surrogate/serving/metrics.py`); the cluster runs
kube-prometheus-stack (Prometheus, Grafana, Alertmanager and the Prometheus
operator), an Argo CD Application from the chart's Helm repository, pinned
(`deploy/argocd/apps/monitoring.yaml`). The service brings what monitoring
needs from it itself (`deploy/k8s/monitoring`): a ServiceMonitor, so
Prometheus scrapes it every 15 s, and its Grafana dashboard, JSON in git
(`dashboards/forecast-service.json`) that Grafana's sidecar loads from a
labeled ConfigMap.

- **Service.** Requests by route, method and status
  (`mhd_serve_requests_total`; the matched route's template, so the label
  set stays bounded; not called `endpoint`, the label the Prometheus
  operator gives every scraped series), time to a response's start, forecasts in flight, their
  duration and frames, forecasts that failed after their response started,
  and which model is served (`mhd_model_info`). Counted by a plain ASGI
  middleware: Starlette's `BaseHTTPMiddleware` would pass every streamed
  forecast chunk through a queue of its own.
- **Model, by proxy.** A served forecast has no truth to be scored against,
  so two proxies stand in, with what they measure on known data:
  - *Input drift*: each context's per-channel mean and standard deviation,
    in units of the training data's (`mhd_context_mean_shift`,
    `mhd_context_std_ratio`), the training set's normalization statistics,
    which promotion attaches to the registered version (`reference_stats`)
    and fetch hands the server. Training and validation contexts (4-frame
    windows) stay within +-0.025 training stds of the mean and at 0.83-1.22
    times the std.
  - *Forecast energy*: each streamed block's mean kinetic energy over the
    context's (`mhd_forecast_energy_ratio`), the quantity the selection
    rule's stability check bounds. The validation truth's 100-step blocks
    stay at 1.00-1.05 times its context's; Hankel DMD's 3000-step forecast
    decays to 0.72 (it damps) and never exceeds 0.85. A blow-up grows it.
  - *Probes stay out.* Synthetic requests -- the post-sync smoke check, any
    `ForecastClient(...).forecast(..., probe=True)` -- carry an `X-Probe`
    header: they count as service traffic but not in the two proxies,
    since their made-up contexts would read as drift (the smoke check's
    random frames did, at 2.4 times the training std of `u_y`, until it
    was marked).
- **Alerts** (`deploy/k8s/monitoring/alerts.yaml`, a PrometheusRule). The
  model alerts' thresholds are a decision of the same kind as the selection
  rule's guardrails, set on the measured basis above:

  | Alert | Fires when | Basis |
  |---|---|---|
  | `ForecastServiceDown` (critical) | no server answered a scrape for 5 min | |
  | `ForecastsFailing` | a forecast failed mid-stream within 15 min | |
  | `ForecastServiceSaturated` | over 20% of forecast requests got 503 in 15 min | one forecast per pod: add replicas |
  | `ForecastBlowUp` (critical) | a block had over 2x its context's energy in 15 min | the stability check's 2x; truth <= 1.05x, Hankel DMD <= 0.85x |
  | `ForecastCollapse` | the median block energy fell below 0.5x the context's in 1 h | Hankel DMD damps to 0.72x, no lower |
  | `ForecastInputDrift` | a channel's 5th/95th percentile mean shift beyond +-0.1 training stds in 1 h | train and val contexts within +-0.025 |
  | `ForecastInputScaleDrift` | a channel's 5th/95th percentile std ratio outside 0.7-1.5 in 1 h | train and val contexts at 0.83-1.22 |

  `tests/serving/test_alert_rules.py` runs promtool's rule unit tests on
  synthetic series of the server's own metrics -- its buckets, its label
  formatting -- so each alert is checked to fire when it should and to stay
  quiet otherwise (a threshold moved from 2x to 3x fails it); CI installs
  promtool, of the Prometheus version the stack runs. Live, on the cluster,
  Prometheus loads all seven, healthy and quiet under real validation
  traffic.
- **Notifications.** Alertmanager mails the service's alerts through
  Gmail's SMTP server: every rule carries `team: mhd-surrogate`, the one
  route to the `email` receiver (`deploy/argocd/apps/monitoring.yaml`),
  grouped by alert, sent again every 12 h while it fires and once more when
  it resolves. The chart's own alerts reach no receiver: kind trips several
  of them (`KubePodNotReady` for its static control-plane pods after a
  restart), and an outage of the service is `ForecastServiceDown`'s. The
  SMTP password, a Gmail app password, is the one Secret not in git
  (`alertmanager-smtp`, created by `deploy/argocd/smtp-secret.sh` from
  `~/.config/mhd-surrogate/smtp-password`; `bootstrap.sh` runs it, so a
  rebuilt cluster has it again). Sealed Secrets or External Secrets would
  keep it in git encrypted, but their key would be one more secret outside
  git for every rebuilt cluster to restore: worth it for several secrets,
  not for one. `tests/serving/test_alert_routing.py` checks the config with
  amtool: every rule reaches email, the chart's alerts don't, and the
  password is read from the mounted Secret, never written inline.
- **kind specifics.** The chart's scrapes of etcd, the controller manager,
  the scheduler and kube-proxy are off: kind runs those bound to localhost,
  where Prometheus can't reach them, and they would only fail. Prometheus
  keeps 7 days in an emptyDir (gone with the pod). Grafana's NodePort is
  mapped to localhost:8082 (`deploy/kind/cluster.yaml`; a cluster created
  before that mapping needs recreating: `kind delete cluster --name mhd`,
  then the two commands under "Kubernetes").

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
  stack (below), and the forecast service (`serve-fetch`, `serve`; see
  "Forecast API").
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
docker compose --profile serve up -d          # the forecast service (see "Forecast API")
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

**Restarts.** Postgres, SeaweedFS and the MLflow server have `restart:
unless-stopped`: they come back with the Docker daemon, after a reboot too,
unless they were stopped on purpose (`docker compose stop`/`down`, then
`docker compose up -d` again). The kind cluster's serving pods fetch their
model from the registry when they start, so a reboot that left it down
would leave them stuck in their init container (it did once: serving was
down for 6 hours until the stack was started by hand). Docker doesn't honor
`depends_on` when it restarts containers itself; the MLflow server retries
its database until Postgres is up, and the one-shot `s3-init` isn't needed
again (the buckets persist in `mlflow/seaweedfs/`).

**Configuration.** Credentials default to local-development values in
`docker-compose.yml`; copy `.env.example` to `.env` to override them. They
are not meant for anything reachable beyond your machine.

**Using the stack from the host.** With the stack up, `uv run
scripts/training/train.py mlflow=server` logs to it via `localhost:5000`
(the `mlflow/server.yaml` config); the default `mlflow=local` still uses the
SQLite file and needs no containers.

**Large artifacts from outside the stack.** For an artifact of 500 MB or
more, the MLflow server (3.16, on an S3 store) advertises presigned
multipart transfers: the client sends or fetches the parts straight from the
object store, at the address the server knows it by, `seaweedfs:8333`, which
resolves only inside the compose network. A client outside it -- the host, a
Kubernetes pod -- sets `MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=false` and
`MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false`, so these go through the
server like every smaller artifact; `scripts/serving/promote.py` sets both.
Without them an upload fails on the unresolvable name and a download keeps
retrying it. `train.py mlflow=server` from the host with a checkpoint that
large (Hankel DMD's 865 MB) needs them exported too.

The default image is CPU-only; `app-gpu` adds the CUDA build of JAX (see
"GPU image").

`scripts/` is located relative to the working directory (`/app` in the
container), not relative to the installed package, since in the image the
package lives in `site-packages`. Keep bind-mount source directories tracked
(`.gitkeep`): Docker creates missing ones as root, which the non-root
container user can't write to.
