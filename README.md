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
