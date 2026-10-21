import logging

import numpy as np
import pytest

from mhd_surrogate.models.snapshots import (
    gram_matrix,
    load_frames,
    normalize_in_place,
    project_basis,
    truncated_pod,
)

FRAME = (2, 3, 4)


def random_frames(n=10, seed=0):
    rng = np.random.default_rng(seed)
    channel_scale = np.array([3.0, 0.5])[:, None, None]
    return (rng.normal(size=FRAME) + channel_scale * rng.normal(size=(n, *FRAME))).astype(
        np.float32
    )


def test_load_frames_stacks_every_dataset_in_order_whatever_the_chunk_size():
    a, b = random_frames(5, seed=1), random_frames(3, seed=2)

    frames = load_frames({"a": a, "b": b}, chunk_t=2, label="test")

    assert frames.shape == (8, 24)
    np.testing.assert_array_equal(frames, np.concatenate([a, b]).reshape(8, -1))


def test_normalized_snapshots_have_zero_mean_and_unit_rms_per_channel():
    raw = random_frames()
    frames = raw.reshape(len(raw), -1).copy()

    mean, scale = normalize_in_place(frames, FRAME, chunk_t=3)

    per_channel = frames.reshape(len(raw), 2, -1)
    assert per_channel.mean(axis=0) == pytest.approx(np.zeros((2, 12)), abs=1e-6)
    assert np.sqrt((per_channel**2).mean(axis=(0, 2))) == pytest.approx([1.0, 1.0])
    restored = frames.reshape(raw.shape) * scale[None, :, None, None] + mean[None]
    np.testing.assert_allclose(restored, raw, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("column_block", [5, 24, 100])
def test_gram_matrix_and_basis_do_not_depend_on_the_device_block_size(column_block):
    frames = random_frames().reshape(10, -1)
    rows = np.array([0, 2, 3, 7])
    coefficients = np.random.default_rng(3).normal(size=(4, 2)).astype(np.float32)

    gram = gram_matrix(frames, column_block)
    basis = project_basis(frames, rows, coefficients, column_block)

    np.testing.assert_allclose(gram, frames.astype(np.float64) @ frames.T, rtol=1e-5)
    np.testing.assert_allclose(basis, frames[rows].T @ coefficients, rtol=1e-4, atol=1e-5)


def test_truncated_pod_gives_the_singular_values_of_the_snapshots():
    frames = random_frames().reshape(10, -1).astype(np.float64)

    s2, s, v = truncated_pod(frames @ frames.T, rank=3, label="test")

    expected = np.linalg.svd(frames, compute_uv=False)
    assert np.sqrt(s2) == pytest.approx(expected, rel=1e-6)
    assert s == pytest.approx(expected[:3], rel=1e-6)
    assert v.shape == (10, 3)


def test_truncated_pod_caps_the_rank_at_the_usable_singular_values(caplog):
    rng = np.random.default_rng(0)
    frames = rng.normal(size=(10, 2)) @ rng.normal(size=(2, 30))  # rank 2

    with caplog.at_level(logging.WARNING):
        _, s, v = truncated_pod(frames @ frames.T, rank=5, label="test")

    assert len(s) == 2 and v.shape == (10, 2)
    assert "only 2 usable" in caplog.text
