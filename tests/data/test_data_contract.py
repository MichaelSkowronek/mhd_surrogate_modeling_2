"""Data contract checks against the real zarr store.

Unlike the rest of tests/, these run against the actual dataset rather than
synthetic data, so they need data/processed/<...>.zarr to exist locally
(built by scripts/data/convert_to_zarr.py) -- they're skipped automatically
otherwise, including in CI, since the ~9GB store isn't checked into git.

The zarr store path and dataset list come from configs/analysis/split.yaml rather
than a hardcoded local path, so these checks work unchanged if the store
later moves to object storage (zarr supports remote stores through the same
API), and they read in chunks rather than loading full arrays, so they stay
cheap to run against a remote store too.

These assert only objective, storage-agnostic structural properties (shape,
dtype, finiteness, a broad sanity value range, enough time steps for the
configured split) -- not statistical representativeness or physical
plausibility, which stay human judgement calls made via check_split.py and
the other check_*.py scripts (see their README sections).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
import zarr

from mhd_surrogate.data.splitting import trailing_split

SPLIT_CONFIG = Path("configs/analysis/split.yaml")
CHUNK_T = 64
# A broad sanity bound, not a tight physical one: catches corrupted data
# (overflow, wrong dtype reinterpretation, garbage), not legitimate
# variation between simulation runs.
MAX_ABS_VALUE = 100.0


def _load_split_config() -> dict:
    return yaml.safe_load(SPLIT_CONFIG.read_text())


def _store_available() -> bool:
    try:
        config = _load_split_config()
    except FileNotFoundError:
        return False
    return Path(config["zarr_store"]).exists()


pytestmark = pytest.mark.skipif(
    not _store_available(),
    reason="zarr store not present locally (build it with scripts/data/convert_to_zarr.py)",
)


@pytest.fixture(scope="module")
def split_config() -> dict:
    return _load_split_config()


@pytest.fixture(scope="module")
def root(split_config):
    return zarr.open_group(store=split_config["zarr_store"], mode="r")


@pytest.fixture(scope="module")
def dataset_names(split_config) -> list[str]:
    return split_config["datasets"]


def test_configured_datasets_exist(root, dataset_names):
    for name in dataset_names:
        assert name in root, f"{name} missing from the zarr store"


def test_shape_and_dtype(root, dataset_names):
    for name in dataset_names:
        arr = root[name]
        assert arr.ndim == 4, f"{name}: expected (T, C, Nx, Ny), got shape {arr.shape}"
        assert arr.shape[1] == 2, f"{name}: expected 2 velocity channels, got {arr.shape[1]}"
        assert arr.dtype == np.float32, f"{name}: expected float32, got {arr.dtype}"


def test_spatial_shape_is_consistent_across_datasets(root, dataset_names):
    shapes = {name: root[name].shape[2:] for name in dataset_names}
    assert len(set(shapes.values())) == 1, f"inconsistent spatial shapes: {shapes}"


def test_no_nan_or_inf(root, dataset_names):
    for name in dataset_names:
        arr = root[name]
        for start in range(0, arr.shape[0], CHUNK_T):
            block = arr[start : start + CHUNK_T]
            assert np.isfinite(block).all(), (
                f"{name}: non-finite values in steps [{start}, {start + CHUNK_T})"
            )


def test_values_within_sanity_bounds(root, dataset_names):
    for name in dataset_names:
        arr = root[name]
        for start in range(0, arr.shape[0], CHUNK_T):
            block = arr[start : start + CHUNK_T]
            max_abs = np.abs(block).max()
            assert max_abs < MAX_ABS_VALUE, (
                f"{name}: |value|={max_abs} exceeds sanity bound {MAX_ABS_VALUE}"
            )


def test_enough_steps_for_configured_split(root, dataset_names, split_config):
    for name in dataset_names:
        n_steps = root[name].shape[0]
        # Raises ValueError if there aren't enough steps for test_fraction/buffer_steps.
        trailing_split(n_steps, split_config["test_fraction"], split_config["buffer_steps"])
