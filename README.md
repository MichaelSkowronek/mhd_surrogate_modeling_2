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

## Exploring the data

```bash
uv run scripts/explore_data.py
```

Prints shape, dtype, value range, and NaN/Inf counts for each `.npy` file in
`data/raw/`, and saves a preview plot per file to `reports/figures/`.
