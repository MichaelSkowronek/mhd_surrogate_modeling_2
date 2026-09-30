"""Time-based train/val/test splitting for MHD simulation time series."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    n_steps: int
    val_steps: int
    test_steps: int
    buffer_steps: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


def trailing_split(n_steps: int, val_steps: int, test_steps: int, buffer_steps: int = 0) -> Split:
    """Hold out the trailing `test_steps` snapshots as the test set, and the
    `val_steps` snapshots immediately before that (minus a buffer) as
    validation. Both are fixed absolute sizes, not fractions of `n_steps`:
    they should each cover the same number of periods of the flow's
    dominant oscillation, which doesn't scale with a given dataset's total
    length (see configs/analysis/split.yaml).

    A time-based split avoids leakage between temporally autocorrelated
    snapshots, unlike a random split. `buffer_steps` drops that many
    snapshots from the end of both train and val, so neither's last
    snapshot is adjacent to the region that immediately follows it (train ->
    val, and val -> test) -- val and test themselves keep their full
    requested size.
    """
    if val_steps < 1:
        raise ValueError(f"val_steps must be >= 1, got {val_steps}")
    if test_steps < 1:
        raise ValueError(f"test_steps must be >= 1, got {test_steps}")
    if buffer_steps < 0:
        raise ValueError(f"buffer_steps must be >= 0, got {buffer_steps}")

    test_start = n_steps - test_steps
    val_end = test_start - buffer_steps
    val_start = val_end - val_steps
    train_end = val_start - buffer_steps

    if train_end < 1:
        raise ValueError(
            f"val_steps={val_steps}, test_steps={test_steps}, buffer_steps={buffer_steps} "
            f"leave no training steps for n_steps={n_steps}"
        )
    return Split(
        n_steps=n_steps,
        val_steps=val_steps,
        test_steps=test_steps,
        buffer_steps=buffer_steps,
        train_start=0,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=n_steps,
    )
