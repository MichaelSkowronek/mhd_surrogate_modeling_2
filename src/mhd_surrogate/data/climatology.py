"""The real flow's climatology of its domain-mean velocity: the reference of
the climatological drift band (CLAUDE.md, "Data analysis";
`evaluation/usable.py`).

A rollout from any start of a dataset runs past the frames its truth
covers, so its bulk flow (each channel's domain-mean velocity) can't be
compared with the truth at the same frames, except from the end of the
context. It is compared with the range the real flow shows instead: every
`window_steps`-frame window of the training datasets (sliding by `stride`)
has, per channel, a mean and a std over time, and

- `level` is the mean of the windows' means (the flow's bulk level),
- `unit` the median of their stds (the band's unit: how much the domain
  mean varies within a window, ~0.36% of the bulk flow for u_x),
- `slow_std` the median std of their `block_steps`-frame block means (the
  slow-variability ratio's reference), and
- `max_window_offset` and `std_ratio_range` how far the windows themselves
  stray from `level` and `unit` (the basis the band's limits are set from;
  reported, not used by the check).

From the training datasets only, like the normalization stats (the
`compute_climatology` DVC stage), so the validation rollouts the band
judges are out of sample.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import zarr
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mhd_surrogate.utils.parallel import map_tasks

DEFAULT_CHUNK_T = 32


def domain_mean_series(arr, chunk_t: int = DEFAULT_CHUNK_T) -> np.ndarray:
    """Each frame's spatial mean per channel of a (T, C, Nx, Ny) array,
    (T, C) in float64, read `chunk_t` frames at a time."""
    return np.concatenate(
        [
            np.asarray(arr[start : start + chunk_t], dtype=np.float64).mean(axis=(2, 3))
            for start in range(0, arr.shape[0], chunk_t)
        ]
    )


def dataset_domain_means(zarr_store: str, name: str, chunk_t: int = DEFAULT_CHUNK_T) -> np.ndarray:
    """`domain_mean_series` of dataset `name` in the zarr group at
    `zarr_store`; opens the store itself, so it can run as a parallel task."""
    root = zarr.open_group(store=str(zarr_store), mode="r")
    return domain_mean_series(root[name], chunk_t)


def store_domain_means(
    zarr_store: Path | str,
    names: Sequence[str],
    chunk_t: int = DEFAULT_CHUNK_T,
    backend: str = "processes",
    workers: int | None = None,
) -> dict[str, np.ndarray]:
    """Per-dataset domain means for `names`, one parallel task each, in
    `names` order."""
    tasks = [(str(zarr_store), name, chunk_t) for name in names]
    return dict(zip(names, map_tasks(dataset_domain_means, tasks, backend, workers), strict=True))


def block_std(series: np.ndarray, block_steps: int) -> np.ndarray:
    """The std of the means of the full `block_steps`-step blocks of a
    (T, C) series, per channel (a trailing partial block is left out)."""
    n = len(series) // block_steps
    if n < 2:
        raise ValueError(f"{len(series)} steps hold fewer than 2 blocks of {block_steps}")
    blocks = np.asarray(series[: n * block_steps], dtype=np.float64)
    return blocks.reshape(n, block_steps, -1).mean(axis=1).std(axis=0)


def window_statistics(
    series: np.ndarray, window_steps: int, stride: int, block_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Means, stds over time and `block_std`s of every `window_steps`-step
    window of a (T, C) series, from step 0 every `stride` steps: three
    (windows, C) arrays."""
    if window_steps > len(series):
        raise ValueError(f"a series of {len(series)} steps has no window of {window_steps}")
    windows = [
        series[s : s + window_steps] for s in range(0, len(series) - window_steps + 1, stride)
    ]
    return (
        np.array([w.mean(axis=0) for w in windows]),
        np.array([w.std(axis=0) for w in windows]),
        np.array([block_std(w, block_steps) for w in windows]),
    )


class Climatology(BaseModel):
    """The domain-mean velocity's climatology (see the module docstring),
    one entry per channel in `channel_names`, from `n_windows` windows of
    `source_datasets`."""

    model_config = ConfigDict(frozen=True)

    channel_names: list[str]
    window_steps: int = Field(gt=0)
    stride: int = Field(gt=0)
    block_steps: int = Field(gt=0)
    level: list[float]
    unit: list[float]
    slow_std: list[float]
    max_window_offset: list[float]
    std_ratio_range: list[tuple[float, float]]
    n_windows: int = Field(gt=0)
    source_datasets: list[str]

    @model_validator(mode="after")
    def _check_consistent(self) -> Climatology:
        n = len(self.channel_names)
        if n == 0:
            raise ValueError("channel_names must not be empty")
        fields = ("level", "unit", "slow_std", "max_window_offset", "std_ratio_range")
        for name in fields:
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have one entry per channel ({n})")
        if not all(math.isfinite(v) and v > 0 for v in (*self.unit, *self.slow_std)):
            raise ValueError("unit and slow_std must be finite and > 0")
        if not self.source_datasets:
            raise ValueError("source_datasets must not be empty")
        return self

    def check_source(self, train_datasets: Sequence[str], test_dataset: str) -> None:
        """Raise ValueError unless computed from exactly the training
        datasets, and never from the held-out test dataset."""
        if test_dataset in self.source_datasets:
            raise ValueError(f"the climatology used the test dataset {test_dataset!r}")
        if set(train_datasets) != set(self.source_datasets):
            raise ValueError(
                f"the climatology was computed from {sorted(self.source_datasets)}, but the "
                f"training datasets are {sorted(train_datasets)}"
            )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> Climatology:
        return cls.model_validate_json(Path(path).read_text())


def climatology_from_series(
    series_by_dataset: Mapping[str, np.ndarray],
    channel_names: Sequence[str],
    window_steps: int,
    stride: int,
    block_steps: int,
) -> Climatology:
    """The `Climatology` of per-dataset (T, C) domain-mean series, their
    windows pooled in the mapping's order (so it reproduces bit for bit)."""
    stats = [
        window_statistics(s, window_steps, stride, block_steps) for s in series_by_dataset.values()
    ]
    means, stds, slow = (np.concatenate(parts) for parts in zip(*stats))
    level = means.mean(axis=0)
    unit = np.median(stds, axis=0)
    ratios = stds / unit
    return Climatology(
        channel_names=list(channel_names),
        window_steps=window_steps,
        stride=stride,
        block_steps=block_steps,
        level=level.tolist(),
        unit=unit.tolist(),
        slow_std=np.median(slow, axis=0).tolist(),
        max_window_offset=(np.abs(means - level) / unit).max(axis=0).tolist(),
        std_ratio_range=[
            (float(lo), float(hi)) for lo, hi in zip(ratios.min(axis=0), ratios.max(axis=0))
        ],
        n_windows=len(means),
        source_datasets=list(series_by_dataset),
    )
