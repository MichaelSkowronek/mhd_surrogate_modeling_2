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
```

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
the streamwise x direction. Grid spacing is not stored in the data, so it is
derived from the domain lengths in `configs/grid.yaml` (`lx: 30`, `ly: 2`,
both **provisional**; see below), giving `dx = 30/1150`, `dy = 2/126`.
`--dx/--dy` override the config (e.g. `--dx 1 --dy 1` for grid units).

For `re16k_t400_0` the normalized divergence is then ~0.37 (train and test
alike, stationary in time; ~0.53 in unit grid spacing). The x length of 30 is
not known independently: a least-squares fit of the `dx/dy` ratio on 13
snapshots implies `Lx` ~ 30 for `Ly` = 2, which is consistent but not
confirmed. Swapping the axes is much worse (~0.93). The divergence maps show
large-scale structure tied to the flow features rather than grid-scale noise.

So the 2D field is not exactly incompressible, which is expected: the
quasi-2D hypothesis is only approximate, so `du_z/dz` in the slice does not
vanish, and this residual (~37% of the derivative magnitude) is a rough
measure of how far the slice is from ideal 2D. Whether it is "good enough"
is a modeling judgement; a non-uniform grid or a discretization that differs
from central differences would also contribute and cannot be separated here.

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

**Provisional:** vorticity scales with 1/length and enstrophy with 1/length^2,
so the absolute values below depend on the unconfirmed `lx`/`ly` in
`configs/grid.yaml`; a non-uniform grid (e.g. refined near the walls) would
also affect them. Rerun once the real grid is known.

For `re16k_t400_0` (with `lx=30`, `ly=2`) the maps show shear layers and jets
near the inlet (x < ~150), large coherent vortices of roughly channel-width
size downstream, and thin high-vorticity layers along both y walls, which
dominate the extremes. Mean enstrophy is ~26.2 in train and ~25.5 in test
(~3% lower), with slow variations over time. The spatially averaged
vorticity is a regular oscillation (period ~25-30 steps, amplitude ~0.03; see
the autocorrelation section)
with a small mean (~0.012 train, ~0.005 test), tiny compared with the local
vorticity magnitude of order 10-20. Both quantities are flagged using the
same thresholds as `check_split.py` (mean shifts of 0.45 and 0.33 std devs),
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

**Provisional:** wavenumber labels use the unconfirmed `lx`/`ly` in
`configs/grid.yaml`. Fitted slopes are unaffected by the length scale (only
the k axis shifts). Because the flow is not homogeneous in x (the inlet
region differs from the developed region), the x spectrum averages over a
non-stationary signal.

For `re16k_t400_0`:

- **x direction:** the spectrum peaks at k ~ 1.7-1.9 (wavelength ~3.4-3.75,
  about 1.7-1.9 channel widths for `ly=2`), matching the size of the coherent
  vortices in the vorticity maps. Above the peak it decays as a power law
  with slope ~ -3 over k ~ 3-60 (log-log fit: -3.0 for `u_x`, -2.5 to -2.9 for
  `u_y` depending on the range). It flattens at k > ~100; the cause was not
  investigated (grid-scale content, leakage from the wall layers, or
  numerical noise are all possible).
- **y direction:** no interior peak; the spectrum decreases monotonically
  from the lowest resolved wavenumber (dominated by the cross-stream
  profile and wall layers), with slope ~ -3.6 for `u_x` and ~ -3.2 for `u_y`
  over k ~ 10-60. `u_x` has roughly 10x the power of `u_y` at low k, closing
  to a factor of a few at high k (read from the plot).
- The slope of ~ -3 in x is the classic 2D enstrophy-cascade value, but it
  is also what smooth fields dominated by isolated vortices and shear layers
  give, and the 1D spectra of a bounded, inhomogeneous domain are not a
  clean test, so this is consistent with rather than evidence for such a
  cascade.
- **Train vs. test:** y spectra agree within ~1-7% at all scales, and the
  fitted slopes agree within ~0.1. The x spectra differ mainly at the lowest
  wavenumbers (test/train power 0.65 for `u_x` and 0.59 for `u_y` in the
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
