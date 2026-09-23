"""Time-based train/test splitting for MHD simulation time series."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    n_steps: int
    test_fraction: float
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def trailing_split(n_steps: int, test_fraction: float) -> Split:
    """Hold out the trailing `test_fraction` of time steps as the test set.

    A time-based split avoids leakage between temporally autocorrelated
    snapshots, unlike a random split.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2 to split, got {n_steps}")

    train_end = round(n_steps * (1 - test_fraction))
    train_end = min(max(train_end, 1), n_steps - 1)
    return Split(
        n_steps=n_steps,
        test_fraction=test_fraction,
        train_start=0,
        train_end=train_end,
        test_start=train_end,
        test_end=n_steps,
    )
