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
against real data. No surrogate model or training loop exists yet --
`scripts/training/train.py` is still just a smoke test of the Hydra/MLflow
plumbing it will grow into.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
uv run pre-commit install
```

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
  training/          mlflow_utils, tracking
  utils/             logging_config, parallel
scripts/             CLI entry points, same split (plus viz/)
  data/              explore_data, convert_to_zarr, compute_stats
  analysis/          check_*.py, run_all_checks, benchmark_backends
  viz/               make_video, make_all_videos
  training/          train
tests/               mirrors src/ and scripts/ (data/, analysis/, training/, utils/, viz/)
dvc.yaml, dvc.lock  data pipeline (raw -> zarr -> stats) and its pinned hashes
data/raw.dvc         DVC pointer to the raw .npy files (.dvc/ holds the remote config)
configs/             Hydra tree: config.yaml + data/, dataset/, normalization/, mlflow/ groups
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
`spectral`, `dmd`, `pod`, `spod`, `parallel`, `mlflow_utils`,
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
division of labor. A training stage in `dvc.yaml` (depending on the zarr
store, the stats and the config params) is the natural next step once a real
training loop exists: `dvc repro`/`dvc exp run` would then guarantee fresh
inputs by construction, with MLflow still doing the tracking. An enforcing
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
(12 cores, 16 GB; wall seconds, median of n runs after an untimed warmup,
backends interleaved so the OS page cache doesn't favor one; it times the real
scripts as subprocesses, so Ray's startup is included):

| workload | sequential | processes | ray |
|---|---|---|---|
| `compute_stats` (7 datasets, n=3) | 19.8 | **7.8** (2.5x) | 11.7 (1.7x) |
| `convert_to_zarr` (9 files, n=2) | 49.8 | **12.1** (4.1x) | 16.9 (2.9x) |

| analysis suite, 48 jobs | workers | wall (s) | vs sequential |
|---|---|---|---|
| sequential (n=1) | 1 | 451.0 | 1.0x |
| processes (n=2) | 5 | 163.2 | 2.8x |
| ray, memory-aware (n=2) | 5 | 166.6 | 2.7x |
| ray, memory-aware (n=2) | 12 | **148.1** | **3.0x** |

What this says, honestly:

- **For the preprocessing, Ray is slower than a process pool**, by its fixed
  ~4 s runtime startup, on a workload that finishes in 8-12 s. The work is
  7-9 uneven tasks on one machine, which is exactly what a process pool is for;
  the speedup is capped well below 12x by the task count and the size spread
  (1248 vs ~900 steps). Default stays `processes`.
- **For the suite, memory, not cores, is the constraint**, and that's where Ray
  earns its place. The six checks differ hugely in peak memory (measured on the
  largest dataset): `check_autocorrelation` 2.2 GB, the other five 0.2-0.4 GB.
  Run 12 at a time and 8 autocorrelation jobs at once need ~16 GB: the first
  attempt at this benchmark exhausted the machine's RAM and had to be killed.
  A thread/process pool runs `--workers` jobs regardless, so it has to be sized
  by hand (5 workers above). Ray schedules against a declared per-job memory
  (`JOB_MEMORY_GB` in `run_all_checks.py`: 2.5 GB for autocorrelation, 0.5 GB
  for the rest), so it can be given all 12 workers and still never overcommit
  memory — and it's the fastest configuration (148 s) because of it. At the
  same worker count the two backends tie (166.6 vs 163.2 s, within the run-to-run
  spread of n=2).
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

## Training config (Hydra)

```bash
uv run scripts/training/train.py
uv run scripts/training/train.py data=re16k seed=123
```

The exploration/analysis scripts above stay on plain argparse + PyYAML
(`configs/analysis/split.yaml`, `configs/analysis/grid.yaml`) — they are finished, standalone
tools and don't need config composition. New training/model code uses
[Hydra](https://hydra.cc) instead, since that will need config groups (model,
optimizer, trainer, ...) that compose, with CLI overrides and later multirun
sweeps. `configs/config.yaml` is the root config (currently `data`, `dataset`
and `seed`); `configs/data/re16k.yaml` holds the zarr store path and the
dataset-level split (`train_datasets`, `val_dataset`, `test_dataset` -- see
"Train / val / test split" above), selected via the `data` default.

Model code will be in [JAX](https://jax.readthedocs.io). `jax` runs on CPU
here; there's an NVIDIA GPU on this machine but no CUDA-enabled `jaxlib`
installed yet.

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

No model exists yet, so `scripts/training/train.py` currently only resolves
the config, confirms every train/val dataset is reachable (shape/dtype,
sample count, one sample's shapes), and logs the run to MLflow (see below),
as a smoke test of the plumbing it will grow into the real training loop on
top of -- it deliberately never opens `test_dataset`, not even for a shape
check. Each run's resolved config and logs are written to
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
dataset before building the normalizer, and fails with a pointer to
`compute_stats.py` if the file is missing.

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
`mlflow.log_params` expects (e.g. `data.dataset`, `dataset.window`); the
dataset shape/dtype and both splits' sample counts are also logged. There
are no metrics yet, since there is no training loop to produce them.

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

## Docker

The project ships as a container image so the exact environment (Python
3.14, the locked dependency set, the bundled ffmpeg) is reproducible on any
machine, and later stages (training jobs, serving) have a unit to deploy.

- **`Dockerfile`** — multi-stage. `builder` installs the locked dependencies
  with uv (in their own cached layer, so source edits don't reinstall
  everything), then the project non-editable. `runtime` (default) copies only
  the finished virtualenv plus `scripts/` and `configs/` into a fresh
  `python:3.14-slim`: no uv, no dev tools, runs as a non-root user.
  `test` adds the dev group and `tests/` and runs pytest.
- **Data is never in the image.** The zarr store is ~9 GB and changes
  independently of the code, so `data/`, `reports/`, and `outputs/` are bind
  mounts. `.dockerignore` keeps them out of the build context too.
- **`docker-compose.yml`** — the app container plus a full MLflow tracking
  stack (below).

```bash
docker compose build
docker compose up -d                          # tracking stack; UI on http://localhost:5000
docker compose run --rm app                   # train.py, logging to the stack
docker compose run --rm --no-deps app python scripts/analysis/run_all_checks.py
docker compose run --rm test                  # pytest inside the image
docker compose down                           # stop; all state stays in ./mlflow
```

(`--no-deps` skips starting the tracking stack for scripts that don't log to it.)

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

The image is CPU-only, like the local setup (no CUDA build of JAX yet).

`scripts/` is located relative to the working directory (`/app` in the
container), not relative to the installed package, since in the image the
package lives in `site-packages`. Keep bind-mount source directories tracked
(`.gitkeep`): Docker creates missing ones as root, which the non-root
container user can't write to.
