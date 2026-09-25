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


def trailing_split(n_steps: int, test_fraction: float, buffer_steps: int = 0) -> Split:
    """Hold out the trailing `test_fraction` of time steps as the test set.

    A time-based split avoids leakage between temporally autocorrelated
    snapshots, unlike a random split. `buffer_steps` further drops that many
    steps from the end of the training region, so the last training snapshot
    is separated from the first test snapshot; the test set keeps its full
    `test_fraction`.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if buffer_steps < 0:
        raise ValueError(f"buffer_steps must be >= 0, got {buffer_steps}")
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2 to split, got {n_steps}")

    test_start = round(n_steps * (1 - test_fraction))
    test_start = min(max(test_start, 1), n_steps - 1)
    train_end = test_start - buffer_steps
    if train_end < 1:
        raise ValueError(
            f"buffer_steps={buffer_steps} leaves no training steps "
            f"(test starts at step {test_start})"
        )
    return Split(
        n_steps=n_steps,
        test_fraction=test_fraction,
        train_start=0,
        train_end=train_end,
        test_start=test_start,
        test_end=n_steps,
    )
