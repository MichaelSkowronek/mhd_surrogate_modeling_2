"""Grid spacing for the 2D slice, derived from the domain lengths in a config."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_GRID_CONFIG = Path("configs/grid.yaml")


def grid_spacing(nx: int, ny: int, config_path: Path = DEFAULT_GRID_CONFIG) -> tuple[float, float]:
    """Uniform (dx, dy) for an nx x ny grid including both endpoints."""
    config = yaml.safe_load(config_path.read_text())
    return config["lx"] / (nx - 1), config["ly"] / (ny - 1)
