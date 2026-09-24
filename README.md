# mhd_surrogate_modeling_2

Surrogate modeling of a magnetohydrodynamic (MHD) flow. This second project is
independent work on the same data as the first one (mhd_surrogate_modeling).

The data is a 2D slice of a 3D direct numerical simulation (DNS) of an MHD
flow. Working with a 2D slice is itself a research hypothesis: the imposed
magnetic field drives the flow toward a near-uniform state along one of the
three spatial axes, so a 2D slice is treated as a reasonable stand-in for the
full 3D field. The project currently focuses on a single dataset
(`re16k_t400_0.npy`); companion datasets `re16k_t400_1.npy` through
`re16k_t400_10.npy` (same setup, different simulation parameters) are planned
for later robustness checks and comparisons, not part of the initial scope.

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

## Tests

```bash
uv run pytest
```

Unit tests live in `tests/`, covering the pure computational logic: the
`src/mhd_surrogate/` modules (`splitting`, `grid`, `fields`, `dataset`) plus
the computational core functions inside the `check_*.py`/`make_video.py`
scripts (e.g. `per_timestep_stats`, `field_acf`, `spectrum_sum`,
`divergence_stats`, `vorticity_stats`, `compute_field`) — the
`scripts/*.py` files aren't part of the installed package, so
`tests/conftest.py` adds `scripts/` to `sys.path` to import them directly.
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

`tests/test_data_contract.py` is different from the rest of `tests/`: it
runs against the real zarr store (the datasets and store path listed in
`configs/split.yaml`) rather than synthetic data, checking objective,
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

The domain lengths are in `configs/grid.yaml` and are used by every
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
uv run scripts/explore_data.py
```

Prints shape, dtype, value range, and NaN/Inf counts for each `.npy` file in
`data/raw/`, and saves a preview plot per file to `reports/figures/`.

## Converting to zarr

```bash
uv run scripts/convert_to_zarr.py
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

## Train/test split

Split parameters live in `configs/split.yaml` (which datasets, test fraction,
buffer). The split is time-based (trailing): the last `test_fraction` of each
dataset's time steps become the test set, since these are temporally
autocorrelated snapshots and a random split would leak information between
train and test.

`buffer_steps` (10) drops that many snapshots from the *end of the training
region*, so the last training snapshot is separated from the first test
snapshot; the test set keeps its full `test_fraction`. For `re16k_t400_0`
(1248 steps) this gives train `[0, 988)`, buffer `[988, 998)`, test
`[998, 1248)`. The value follows the autocorrelation check (the pooled field
autocorrelation falls below 0.05 after ~8 steps) and is in snapshot steps.
It does not remove the recurring quasi-periodic correlation at longer lags
(see the autocorrelation section). All analysis scripts read the regions
from the split manifest and exclude the buffer from both train and test.

```bash
uv run scripts/split_data.py   # writes data/processed/splits/split_manifest.json
uv run scripts/check_split.py  # sanity-checks train vs. test statistics
```

`check_split.py` computes, per time step, the spatial mean/std/min/max of
each channel plus (when there are 2 channels, i.e. velocity components) a
kinetic energy proxy per direction (`0.5*u_x^2`, `0.5*u_y^2`) and in total,
and the spatial `u_x`-`u_y` correlation. It compares train vs. test aggregate statistics for all of
these and produces two plots per dataset: every series over time with the
split boundary marked (`<name>_split_check.png`), and a train-vs-test value
histogram per channel (`<name>_split_hist.png`) — so a drift, trend, or
distribution shift concentrated in the held-out tail would be visible
rather than hidden inside a single aggregate number.

For `re16k_t400_0`, the raw channel means/std/histograms match closely
between train and test (mean shift ~0 std devs, std within ~3%). The energy
proxy, however, is flagged (mean shift ~1.5 std devs, std ~34% lower in
test): the flow has a slow oscillation (period ~500-700 steps) with a
pronounced high-energy excursion around t=600-700 that falls inside the
training region; the trailing test region sits at a lower point of that
cycle, though it matches the last third of train (t=700-988) reasonably
well. This is a real feature of the dynamics, not a computation artifact —
worth keeping in mind when interpreting test-set performance later, since
the current trailing split under-represents that higher-energy regime. The
per-direction energies show this comes entirely from `u_x` (~1.17 std devs
shift); `u_y` energy is essentially unchanged (~0.10).
`u_x`-`u_y` correlation is not flagged (~0.04 std devs shift).

## Incompressibility check

```bash
uv run scripts/check_divergence.py [--dx DX --dy DY]
```

Computes `div(u) = du_x/dx + du_y/dy` per time step (second-order central
differences, layout assumed `(T, 2, Nx, Ny)` with `x` = axis 2) and plots the
RMS divergence, the RMS normalized by the RMS of the two derivative terms,
and divergence maps at the first/middle/last step. The long axis (axis 2) is
the streamwise x direction. Grid spacing is derived from the domain lengths
in `configs/grid.yaml` (see the Grid section): `dx = 25/1150`, `dy = 2/126`.
`--dx/--dy` override the config (e.g. `--dx 1 --dy 1` for grid units).

For `re16k_t400_0` the normalized divergence is ~0.41 (train and test alike,
stationary in time; RMS divergence ~1.05). In grid units (`dx = dy = 1`) it
was ~0.53, and swapping the two axes was much worse (~0.93 in grid units),
confirming `x` = axis 2. The divergence maps show large-scale structure tied
to the flow features rather than grid-scale noise.

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
uv run scripts/check_vorticity.py [--dx DX --dy DY]
```

Computes the out-of-plane vorticity `w = du_y/dx - du_x/dy` (same central
differences, layout and `configs/grid.yaml` spacing as the divergence check)
and the enstrophy proxy `0.5*w^2` (spatial mean per time step, matching the
`0.5*u^2` energy convention). Only the z-component exists in a 2D slice, so
this is a 2D enstrophy, not the full 3D one. It prints train vs. test
statistics for the mean vorticity and the enstrophy, plots both over time
with the train/test boundary, and maps the vorticity at the first/middle/last
step.

Vorticity scales with 1/length and enstrophy with 1/length^2, so the absolute
values below depend on the domain lengths in `configs/grid.yaml`. The
`du_x/dy` term inherits the y-interpolation caveat (see the Grid section):
the thin wall layers, which dominate the vorticity extremes, are the part of
the field most affected by the linear interpolation onto the uniform y grid.

For `re16k_t400_0` the maps show shear layers and jets
near the inlet (x < ~150), large coherent vortices of roughly channel-width
size downstream, and thin high-vorticity layers along both y walls, which
dominate the extremes. Mean enstrophy is ~27.5 in train and ~26.8 in test
(~3% lower), with slow variations over time. The spatially averaged
vorticity is a regular oscillation (period ~25-30 steps, amplitude ~0.03; see
the autocorrelation section)
with a small mean (~0.012 train, ~0.005 test), tiny compared with the local
vorticity magnitude of order 10-20. Both quantities are flagged using the
same thresholds as `check_split.py` (mean shifts of 0.48 and 0.33 std devs),
but those thresholds are tight because the temporal fluctuations of these
spatial means are small relative to their level, so the actual differences
are small. The enstrophy is consistent with the milder end of the
energy finding in the split check: higher in the ~400-700 stretch, lower in
the trailing test region.

## Spatial power spectrum

```bash
uv run scripts/check_spectrum.py [--dx DX --dy DY]
```

Computes 1D power spectra E(k) of `u_x` and `u_y` along x (axis 2) and along
y (axis 3), each averaged over the other spatial axis and over time, for the
train and test time steps separately. The domain is not periodic (inlet,
walls in y), so each line has its mean removed and a Hann window applied
before the FFT. Spectra are one-sided and normalized so that E(k) integrated
over the angular wavenumber equals the window-weighted variance; this was
verified on synthetic sine waves (integral 0.500 for a unit-amplitude sine,
peak at the expected wavenumber). Output: a log-log plot
(`<name>_spectrum.png`) and a train-vs-test summary.

Wavenumbers use the domain lengths in `configs/grid.yaml` (`k = 2*pi /
wavelength` in physical units; see the Grid section). Because the flow is not
homogeneous in x (the inlet region differs from the developed region), the x
spectrum averages over a non-stationary signal. In y the data was linearly
interpolated from a non-uniform DNS grid, which acts as a smoothing filter
and gives a piecewise-linear profile, so the high-k part of the y spectrum
cannot be attributed purely to the flow.

For `re16k_t400_0`:

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
- **Train vs. test:** y spectra agree within ~1-7% at all scales, and the
  fitted slopes agree within ~0.1. The x spectra differ mainly at the lowest
  wavenumbers (test/train power 0.66 for `u_x` and 0.59 for `u_y` in the
  lowest band, ~1.2-1.3 in the next band), where the test peak appears
  slightly shifted toward higher k. These bands contain few wavenumber bins
  and ~8-10 structures per domain length, so with ~250 temporally correlated
  test steps this is plausibly statistical noise plus the slow energy
  variation found in the split check, though this was not tested.

## Temporal autocorrelation

```bash
uv run scripts/check_autocorrelation.py [--max-lag N]
```

Two analyses, with lags in snapshot steps (the physical time between
snapshots is not stored in the data, so no physical time scales here):

1. **Field autocorrelation:** the pointwise fluctuation `u' = u - <u>_t`
   (about each region's time-mean field) correlated with itself at lag `tau`,
   pooled over all grid points via an FFT along time, per channel and
   separately for train and test.
2. **Scalar autocorrelation** over the whole series: spatial means of `u_x`
   and `u_y` and the per-direction kinetic energy, with the approximate
   `+-1.96/sqrt(N)` white-noise band.

The estimators were validated on synthetic AR(1) data (measured
autocorrelation matches `phi^lag` to within ~0.005 at every lag, and is
insensitive to a mean offset and scale). The reported lags, integral time
`tau_int` and `N_eff = N/(2*tau_int)` are rough: `tau_int` is truncated at
the first zero crossing, which ignores the negative lobe and the recurring
oscillations described below, and the tail of the estimate is noisy.

For `re16k_t400_0`:

- **Fast decorrelation, then oscillation.** The field autocorrelation
  falls below 1/e after ~5-6 steps and crosses zero after ~7-9 steps
  (`u_x`: rho(1) = 0.90, `u_y`: 0.82), with a negative lobe (down to -0.4 to
  -0.55 near lag 12-14). It then keeps oscillating with a period of ~25
  steps (peaks near lags 26, 51, 76, 99, 121, 147, still ~0.3 at lag ~147), so
  the flow contains a persistent quasi-periodic component. The spatial mean
  of `u_y` is close to a pure oscillator with a period of ~28 steps, with the
  oscillation amplitude modulated in time.
- **Train vs. test:** the initial decay is essentially identical (rho(1)
  0.896 vs. 0.885 for `u_x`, 0.824 vs. 0.816 for `u_y`; same 1/e lags). The
  later oscillatory recurrences are stronger in test (e.g. `u_y` ~0.42 vs.
  ~0.22 near lag 26). With only 250 test steps and each region's own mean
  removed, this may be sampling noise or a real difference in how coherent
  the periodic component is; not determined.
- **Slow energy variation.** The kinetic energy per direction has much
  longer memory (1/e at ~32 steps for `u_x`, ~23 for `u_y`; the `u_x` energy
  only crosses zero at lag ~318), consistent with the slow oscillation found
  in the split check. Its `N_eff` over the whole series is only ~10 (`u_x`)
  and ~30 (`u_y`), so a 250-step test window holds just a couple of
  effectively independent samples of the slow variation. As a
  back-of-envelope estimate, accounting for the autocorrelation puts the
  train/test `u_x` energy shift flagged in the split check at roughly 1.5
  standard errors of the difference of means (the split check's 1.17 is in
  units of the time series' own standard deviation, a different scale):
  suggestive, but not clearly distinguishable from sampling variability of
  a slowly varying signal. This
  estimate is crude (`N_eff` of about 2 in the test window is too small for
  a normal approximation to be reliable).
- **Implication for the split:** snapshots are highly correlated at short
  lags, so a random split would leak information, as assumed. The direct
  (monotone) correlation drops below 0.05 after ~8 steps, which suggests a
  buffer of roughly 10 steps between train and test would remove it; the
  recurring oscillatory correlation at longer lags reflects a persistent
  periodic component of the dynamics and is not something a buffer removes.
  A 10-step buffer is configured in `configs/split.yaml`.

Only temporal autocorrelation is covered; spatial autocorrelation (integral
length scales) and the enstrophy autocorrelation are not.

## Video

```bash
uv run scripts/make_video.py --dataset re16k_t400_0
uv run scripts/make_video.py --dataset re16k_t400_0 --field speed --stride 2 --fps 30
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
level. This is a subjective, qualitative check, not a measurement, but it
lines up with the quantitative train-vs-test comparisons in the sections
above.

## Training config (Hydra)

```bash
uv run scripts/train.py
uv run scripts/train.py data=re16k seed=123
```

The exploration/analysis scripts above stay on plain argparse + PyYAML
(`configs/split.yaml`, `configs/grid.yaml`) — they are finished, standalone
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
`src/mhd_surrogate/dataset.py`'s `WindowedDataset` turns one train/test
region into fixed-size samples: `window` consecutive time steps as input,
the following `horizon` steps as target, a new sample every `stride` steps.
It reads directly from the zarr array (no data is preloaded into memory) and
never lets a sample cross the train/test boundary, since it's built from one
region's `[start, end)` range in the split manifest. `windowed.yaml`'s
`window=4, horizon=1, stride=1` are placeholder defaults (short history,
one-step-ahead prediction), not tied to any model yet. Values are returned
as-is, float32; normalization is not implemented yet.

No model exists yet, so `scripts/train.py` currently only resolves the
config, confirms the configured dataset is reachable (shape/dtype), and
builds the train/test `WindowedDataset`s (sample count, one sample's
shapes), as a smoke test of the plumbing it will grow into the real training
loop on top of. Each run's resolved config and logs are written to
`outputs/<date>/<time>/` (gitignored, like the other run artifacts).
`hydra.job.chdir` is set to `false` so the working directory stays the repo
root; without it, Hydra's default of chdir-ing into the run directory would
break every relative path used throughout this project (`data/raw/...`,
`configs/...`, etc.).

**Dependency note:** `hydra-core` is pinned to the `1.4.0.dev9` pre-release.
The latest stable release (1.3.7) is broken on Python 3.14 (this project's
Python version) — an upstream bug
([facebookresearch/hydra#3121](https://github.com/facebookresearch/hydra/issues/3121)):
its CLI parser fails under Python 3.14's stricter `argparse` validation
before any user code runs. `1.4.0.dev9` fixes it and was verified to resolve
config and run cleanly via a plain `uv sync` (no `--prerelease` flag needed,
since the version is pinned exactly). Swap for the stable 1.4.0 release once
it ships.
