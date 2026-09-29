# mhd_surrogate_modeling_2

Surrogate modeling of a magnetohydrodynamic (MHD) flow. This second project is
independent work on the same data as the first one (mhd_surrogate_modeling).

The data is a 2D slice of a 3D direct numerical simulation (DNS) of an MHD
flow. Working with a 2D slice is itself a research hypothesis: the imposed
magnetic field drives the flow toward a near-uniform state along one of the
three spatial axes, so a 2D slice is treated as a reasonable stand-in for the
full 3D field. There are 9 datasets (`re16k_t400_0.npy` through
`re16k_t400_10.npy`, excluding two known-bad ones; same setup, different
simulation parameters). The analysis suite (see "Running the suite across
all datasets") now runs across all 9, to support the eventual decision on
final test set size and which dataset(s) to train on; the training config
(`configs/data/`) still targets just `re16k_t400_0` for now, since that
decision hasn't been made yet.

## Status

Data exploration phase. No models are implemented yet.

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
  data/              splitting, dataset, grid
  analysis/          fields, summary, spectral
  training/          mlflow_utils
  utils/             logging_config, parallel
scripts/             CLI entry points, same split (plus viz/)
  data/              explore_data, convert_to_zarr, split_data
  analysis/          check_*.py, run_all_checks
  viz/               make_video, make_all_videos
  training/          train
tests/               mirrors src/ and scripts/ (data/, analysis/, training/, utils/, viz/)
configs/             Hydra tree: config.yaml + data/, dataset/, mlflow/ groups
  analysis/          plain-YAML configs (split.yaml, grid.yaml)
```

New code goes under the subpackage/subdirectory matching its pipeline stage,
with its tests in the mirrored `tests/` subdirectory. `configs/analysis/` is
the home for the plain-YAML (non-Hydra) configs the data/analysis scripts
read directly — including `split.yaml`, which `scripts/data/split_data.py`
consumes — while everything Hydra composes for training lives in the rest of
`configs/`. Scripts are run from the repo root (`uv run
scripts/analysis/check_split.py`), since data and config paths are relative
to it.

## Tests

```bash
uv run pytest
```

Unit tests live in `tests/`, covering the pure computational logic: the
`src/mhd_surrogate/` modules (`splitting`, `grid`, `fields`, `dataset`,
`summary`, `parallel`, `mlflow_utils`, `logging_config`) plus the
computational core functions inside the `check_*.py`/`make_video.py`
scripts (e.g. `per_timestep_stats`, `field_acf`, `spectrum_sum`,
`divergence_stats`, `vorticity_stats`, `compute_field`) — the
`scripts/**/*.py` files aren't part of the installed package, so
`tests/conftest.py` adds each `scripts/` subdirectory to `sys.path` to import
them directly.
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

### Data contract tests

`tests/data/test_data_contract.py` is different from the rest of `tests/`: it
runs against the real zarr store (the datasets and store path listed in
`configs/analysis/split.yaml`) rather than synthetic data, checking objective,
storage-agnostic structural invariants — every configured dataset exists,
shape is `(T, 2, Nx, Ny)`, dtype is `float32`, all values are finite, values
stay within a broad sanity bound (`MAX_ABS_VALUE = 100`, meant to catch
corrupted data, not enforce a tight physical range), spatial shape is
consistent across datasets, and there are enough time steps for the
configured split (`test_fraction`/`buffer_steps`). It does **not** check
statistical representativeness or physical plausibility — that stays with
`check_split.py` and the other `check_*.py` scripts.

It reads the zarr store path from config rather than a hardcoded local one,
and reads in chunks rather than loading full arrays, so it keeps working
unchanged if the store moves from local disk to object storage (S3/GCS)
later — zarr supports both through the same API. Since the ~9GB store isn't
checked into git, these tests skip themselves automatically when it isn't
present locally, including in CI.

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

Raw `.npy` files are not tracked in git (see `.gitignore`). Copy them into
`data/raw/` before running any scripts:

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
  snapshots. This is consistent with the stationarity found in the train/test
  split and autocorrelation checks: no early-time transient is visible,
  though a slower ~500-700 step oscillation is present (see Train/test
  split).

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

Writes all simulations in `data/raw/` into a single zarr store at
`data/processed/re16k_t400.zarr` (also gitignored). Since each simulation has
a different number of time steps on the same spatial grid, they are stored as
separate arrays within one group rather than stacked into one array:

```python
import zarr

root = zarr.open_group("data/processed/re16k_t400.zarr", mode="r")
root["re16k_t400_0"]  # shape (1248, 2, 1151, 127)
```

## Train+val / test split

Split parameters live in `configs/analysis/split.yaml` (which datasets, test fraction,
buffer). The split is time-based (trailing): the last `test_fraction` of each
dataset's time steps become the held-out test set, since these are temporally
autocorrelated snapshots and a random split would leak information across
the boundary. The rest — still called "train" in the split manifest — is the
train+val region analyzed below; the actual train/val boundary within it
isn't decided yet.

**Analysis scripts never read the test region.** Every `scripts/analysis/*.py`
script only ever reads `[0, train_end)`, i.e. the train+val region — not just
its own summary statistics being excluded, the test region's raw values are
never loaded at all. Looking at the held-out set, even just its aggregate
statistics, is data snooping: it lets the choices made while developing the
analysis and later the model be (even unconsciously) informed by data the
final evaluation is supposed to be blind to. This didn't matter much while
still exploring what the data looks like at all, but the tooling now exists
to actually train and compare models, so it's worth doing properly before
that starts, rather than after. `check_split.py` used to compare train
against test directly for exactly this purpose (below); that comparison is
gone for now, and will return as train-vs-val once the split within
train+val is decided.

`test_fraction` (0.4) is a conservative placeholder, not a derived value —
pending an actual decision on dataset/split sizing (see `CLAUDE.md`), once
the analysis below has been reviewed in full.

`buffer_steps` (20) drops that many snapshots from the *end of the train+val
region*, so its last snapshot is separated from the first test snapshot; the
test set keeps its full `test_fraction`. Unlike `test_fraction`, this one is
derived, from the autocorrelation check below: the direct (monotone) field
decorrelation crosses zero within 6-11 steps and drops below 0.05 within
6-10 steps, checked across all 9 datasets, so 20 clears that with margin.
The recurring quasi-periodic component found in every dataset (period
~24-40 steps) is not removed by any practical buffer size -- see "Implication
for the split" in the autocorrelation section for why 20 rather than
something larger. For `re16k_t400_0` (1248 steps) this gives train+val
`[0, 729)`, buffer `[729, 749)`, test `[749, 1248)`.

```bash
uv run scripts/data/split_data.py   # writes data/processed/splits/split_manifest.json
uv run scripts/analysis/check_split.py  # describes the train+val region
```

`check_split.py` computes, per time step, the spatial mean/std/min/max of
each channel plus (when there are 2 channels, i.e. velocity components) a
kinetic energy proxy per direction (`0.5*u_x^2`, `0.5*u_y^2`) and in total,
and the spatial `u_x`-`u_y` correlation, over the train+val region. It
produces two plots per dataset: every series over time (`<name>_split_check.png`)
and a value histogram per channel (`<name>_split_hist.png`) — so a drift or
trend within the region would be visible rather than hidden inside a single
aggregate number.

For `re16k_t400_0`'s train+val region: `u_x` mean 1.0, std 0.824, range
[-3.23, 4.64]; `u_y` mean ~0 (-0.0003), std 0.421, range [-2.97, 2.63] --
`u_x` (streamwise) carries most of the flow's energy and variance, as
expected. Kinetic energy is 0.840 (`u_x`) + 0.089 (`u_y`) = 0.929 total,
with small std relative to the mean (~2%) -- consistent with the "slow
energy variation" found in the autocorrelation section being a real but
modest-amplitude effect. `u_x`-`u_y` correlation is essentially zero
(0.007 +- 0.036), i.e. the two velocity components are spatially
uncorrelated on average, as expected for a shear-dominated channel flow.

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
  and ~22 (`u_y`) — a short held-out window would hold just a handful of
  effectively independent samples of this slow variation, worth keeping in
  mind once the train/val/test sizes are decided. This estimate is itself
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

## Frequency analysis

The autocorrelation check above found a persistent quasi-periodic
component but could only infer its period indirectly, from zero crossings.
The four scripts below read it off directly as spectral peaks, from
several independent angles, toward eventually sizing train/val/test off
the slowest well-characterized frequency (see the Train+val / test split
section above). They are exploratory and, unlike the check scripts above,
not wired into `run_all_checks.py`. The shared FFT machinery
(`power_spectrum`, `welch_spectrum`, `dominant_periods`) lives in
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

### Where this leaves the split-sizing question

Four independent analyses (pooled autocorrelation, per-point, domain-mean,
and joint space-time spectra) now agree on the same ~24-40 step
(dataset-dependent) oscillation as the flow's best-characterized frequency
-- confirmed, not just detected once. It is a broad, quasi-periodic
feature rather than a pure tone, which is itself useful to know: it argues
for buffer/margin choices with headroom rather than a razor-precise
period. Kinetic energy's slow variation remains a genuine open question --
its own characterizing timescale exceeds what a single train+val window
can resolve, which matters for deciding how much data a final split needs,
not just how it should be divided. The actual train/val/test size decision
is still pending, per `CLAUDE.md`.

## Running the suite across all datasets

`configs/analysis/split.yaml` now lists all 9 available datasets (`re16k_t400_0`
through `re16k_t400_10`, excluding the two known-bad ones), so
`uv run scripts/data/split_data.py` builds a manifest covering all of them.

Each of the five `check_*.py` scripts accepts `--dataset NAME` (repeatable;
default: every dataset in the manifest) and now writes a JSON summary of its
key numbers per dataset to `reports/summaries/<dataset>__<script>.json`
(gitignored, like the figures), in addition to its existing printed output
and plots — this is what makes cross-dataset comparison possible instead of
having to read 45 separate walls of text.

```bash
uv run scripts/analysis/run_all_checks.py
uv run scripts/analysis/run_all_checks.py --dataset re16k_t400_0 --dataset re16k_t400_1
uv run scripts/analysis/run_all_checks.py --script check_split.py --workers 4
```

`scripts/analysis/run_all_checks.py` runs every (script, dataset) pair — 45 by
default — and builds a comparison table (printed and written to
`reports/summaries/comparison.csv`) from the resulting JSON summaries, with
one row per dataset and a handful of headline train+val numbers (step
counts, divergence residual, enstrophy, the `u_x` field decorrelation time
and effective sample size). The full detail stays in the individual JSON
files; the table is meant for a quick side-by-side look, not the final word.

**Parallelization:** each (script, dataset) pair is independent, so this
dispatches them as separate `python check_*.py --dataset X` subprocesses via
`mhd_surrogate.utils.parallel` (a thread pool where the threads just block on
`subprocess.run`; the real numpy/FFT work happens in the child processes, on
separate cores, with full process isolation — also used by
`make_all_videos.py` below). All 45 jobs (9 datasets, on the ~60% train+val
region only) completed in ~67s wall time on a 12-core machine, down from the
~113s measured pre-overhaul against the full-length arrays — consistent
with reading and processing noticeably less data per job. A distributed
framework like Ray was considered but is not warranted for a workload this
size (a minute, one machine); it would earn its keep once training actually
needs a cluster, distributed GPUs, or data beyond single-machine scale.

**First cross-dataset result (train+val, post-overhaul):** the divergence
residual clusters tightly across all 9 datasets (0.4165-0.4328), and so does
enstrophy (26.4-28.0) — consistent with the `re16k_t400_0` numbers above
being representative of this simulation family rather than a fluke of one
run. The `check_split`/`check_vorticity` flags this section used to mention
are gone along with the train-vs-test comparison that produced them (see
the Train+val / test split section above); this table now only describes
train+val, so it has nothing to flag against. The tooling working
end-to-end across all 9 datasets, on the new split, is confirmed; the
dataset-selection/split-size decision itself is a separate step, still
pending, from reviewing this table.

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
and `seed`); `configs/data/` holds data-source configs (zarr store + array
name), selected via the `data` default.

Model code will be in [JAX](https://jax.readthedocs.io). `jax` runs on CPU
here; there's an NVIDIA GPU on this machine but no CUDA-enabled `jaxlib`
installed yet.

`configs/dataset/` holds sample-windowing configs (`window`/`horizon`/
`stride`/the split manifest path), selected via the `dataset` default.
`src/mhd_surrogate/data/dataset.py`'s `WindowedDataset` turns one train/test
region into fixed-size samples: `window` consecutive time steps as input,
the following `horizon` steps as target, a new sample every `stride` steps.
It reads directly from the zarr array (no data is preloaded into memory) and
never lets a sample cross the train/test boundary, since it's built from one
region's `[start, end)` range in the split manifest. `windowed.yaml`'s
`window=4, horizon=1, stride=1` are placeholder defaults (short history,
one-step-ahead prediction), not tied to any model yet. Values are returned
as-is, float32; normalization is not implemented yet.

No model exists yet, so `scripts/training/train.py` currently only resolves the
config, confirms the configured dataset is reachable (shape/dtype), builds
the train/test `WindowedDataset`s (sample count, one sample's shapes), and
logs the run to MLflow (see below), as a smoke test of the plumbing it will
grow into the real training loop on top of. Each run's resolved config and
logs are written to `outputs/<date>/<time>/` (gitignored, like the other run
artifacts). `hydra.job.chdir` is set to `false` so the working directory
stays the repo root; without it, Hydra's default of chdir-ing into the run
directory would break every relative path used throughout this project
(`data/raw/...`, `configs/...`, etc.).

### Experiment tracking (MLflow)

```bash
uv run scripts/training/train.py
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db  # view runs at http://127.0.0.1:5000
# or, with the Docker stack running (see "Docker"): train.py mlflow=server
```

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
- A one-shot `s3-init` container creates the artifact bucket.
- The server's extra dependencies (`psycopg2`, `boto3`) live in the
  `server` extra in `pyproject.toml`; training clients don't need them.

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
