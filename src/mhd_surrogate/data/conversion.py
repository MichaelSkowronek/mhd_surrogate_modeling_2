"""Convert raw (T, C, H, W) .npy simulations into one zarr group.

Each simulation becomes its own array (they have different T on one spatial
grid). The arrays are independent, so they're written by parallel tasks
(`utils.parallel.map_tasks`); the group and its attributes are created up
front by the caller's process, and each task only creates and fills its own
child array, so tasks never touch shared metadata.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import zarr

from mhd_surrogate.utils.parallel import map_tasks

log = logging.getLogger(__name__)


def convert_array(src_path: str, out_store: str, chunk_t: int, overwrite: bool) -> str:
    """Write one .npy file as the array `<stem>` in the zarr group `out_store`.

    Reads and writes `chunk_t` time steps at a time, so the source is
    memory-mapped and never fully loaded. Returns the array's name.
    """
    path = Path(src_path)
    src = np.load(path, mmap_mode="r")
    name = path.stem
    n_steps = src.shape[0]

    log.info("%s: %s %s -> %s/%s", name, src.shape, src.dtype, out_store, name)
    root = zarr.open_group(store=out_store, mode="a")
    arr = root.create_array(
        name=name,
        shape=src.shape,
        dtype=src.dtype,
        chunks=(min(chunk_t, n_steps),) + src.shape[1:],
        overwrite=overwrite,
    )
    for start in range(0, n_steps, chunk_t):
        end = min(start + chunk_t, n_steps)
        arr[start:end] = src[start:end]

    arr.attrs["source_file"] = path.name
    arr.attrs["n_steps"] = n_steps
    return name


def convert_to_zarr(
    paths: Sequence[Path],
    out_path: Path,
    chunk_t: int,
    overwrite: bool,
    description: str,
    channel_names: Sequence[str],
    excluded_files: Sequence[str],
    backend: str = "processes",
    workers: int | None = None,
) -> list[str]:
    """Write every file in `paths` into the zarr group at `out_path`, one
    parallel task per file; returns the array names in `paths` order.
    """
    root = zarr.open_group(store=str(out_path), mode="w" if overwrite else "a")
    root.attrs["description"] = description
    root.attrs["channel_names"] = list(channel_names)
    root.attrs["excluded_files"] = list(excluded_files)

    tasks = [(str(p), str(out_path), chunk_t, overwrite) for p in paths]
    return map_tasks(convert_array, tasks, backend, workers)
