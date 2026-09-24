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

Split parameters live in `configs/split.yaml` (which datasets, test fraction).
The split is time-based (trailing): the last `test_fraction` of each
dataset's time steps become the test set, since these are temporally
autocorrelated snapshots and a random split would leak information between
train and test.

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
cycle, though it matches the last third of train (t=700-998) reasonably
well. This is a real feature of the dynamics, not a computation artifact —
worth keeping in mind when interpreting test-set performance later, since
the current trailing split under-represents that higher-energy regime. The
per-direction energies show this comes entirely from `u_x` (~1.15 std devs
shift); `u_y` energy is essentially unchanged (~0.08).
`u_x`-`u_y` correlation is not flagged (~0.03 std devs shift).
