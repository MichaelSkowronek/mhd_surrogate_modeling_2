import numpy as np
import pytest

from mhd_surrogate.models.codec import load_codec
from mhd_surrogate.models.pod_codec import PODCodec

FRAME = (2, 6, 4)
STATE = int(np.prod(FRAME))


def subspace_datasets(n_datasets=2, length=12, dim=4, seed=0):
    """Frames whose fluctuations lie in a fixed `dim`-dimensional subspace
    (a random map of random coefficients), with a mean field and different
    channel amplitudes added. Returns the datasets and a function making
    more frames from the same subspace."""
    rng = np.random.default_rng(seed)
    embed = rng.normal(size=(STATE, dim))
    mean = rng.normal(size=FRAME)
    channel_scale = np.array([3.0, 0.5])[:, None, None]

    def frames(n):
        fluct = (embed @ rng.normal(size=(dim, n))).T.reshape(n, *FRAME)
        return (mean + channel_scale * fluct).astype(np.float32)

    return {f"d{i}": frames(length) for i in range(n_datasets)}, frames


def test_frames_in_the_span_of_the_training_frames_round_trip_exactly():
    datasets, more_frames = subspace_datasets()
    codec = PODCodec(rank=4)
    codec.fit(datasets)
    unseen = more_frames(5)

    latents = codec.encode(unseen)

    assert latents.shape == (5, 4) and codec.latent_shape == (4,)
    np.testing.assert_allclose(codec.decode(latents), unseen, rtol=1e-4, atol=1e-4)
    assert codec.fit_info["explained_variance"] == pytest.approx(1.0)


def test_training_latents_are_uncorrelated_with_decreasing_mode_variances():
    datasets, _ = subspace_datasets()
    codec = PODCodec(rank=4)
    codec.fit(datasets)

    latents = codec.encode(np.concatenate(list(datasets.values()))).astype(np.float64)

    np.testing.assert_allclose(codec.basis.T @ codec.basis, np.eye(4), atol=1e-5)
    products = latents.T @ latents
    variances = np.diag(products)
    np.testing.assert_allclose(products, np.diag(variances), atol=1e-3 * variances[0])
    assert (np.diff(variances) < 0).all()


def test_truncation_drops_exactly_the_trailing_modes_energy():
    datasets, _ = subspace_datasets()
    full, truncated = PODCodec(rank=4), PODCodec(rank=2)
    full.fit(datasets)
    truncated.fit(datasets)
    frames = np.concatenate(list(datasets.values()))

    # In normalized units, where the POD is optimal.
    scale = truncated.scale[:, None, None]
    residual = (frames - truncated.decode(truncated.encode(frames))) / scale
    variances = (full.encode(frames).astype(np.float64) ** 2).sum(axis=0)

    assert (residual.astype(np.float64) ** 2).sum() == pytest.approx(variances[2:].sum(), rel=1e-3)
    assert truncated.fit_info["explained_variance"] == pytest.approx(
        variances[:2].sum() / variances.sum(), rel=1e-4
    )


def test_rank_is_capped_at_the_number_of_usable_singular_values():
    datasets, _ = subspace_datasets(dim=3)
    codec = PODCodec(rank=10)
    codec.fit(datasets)

    assert codec.latent_shape == (3,)
    assert codec.fit_info["rank"] == 3
    assert codec.parameter_count() == STATE * 3


def test_result_does_not_depend_on_chunk_or_device_block_size():
    datasets, more_frames = subspace_datasets()
    small, large = PODCodec(rank=4, chunk_t=3, column_block=5), PODCodec(rank=4)
    small.fit(datasets)
    large.fit(datasets)
    unseen = more_frames(7)

    np.testing.assert_allclose(
        small.decode(small.encode(unseen)), large.decode(large.encode(unseen)), atol=1e-4
    )


def test_checkpoint_round_trips_through_the_codec_registry(tmp_path):
    datasets, more_frames = subspace_datasets()
    codec = PODCodec(rank=3, chunk_t=5)
    codec.fit(datasets)
    codec.save(tmp_path)

    loaded = load_codec(tmp_path)

    unseen = more_frames(4)
    assert isinstance(loaded, PODCodec)
    assert (loaded.encode(unseen) == codec.encode(unseen)).all()
    assert (loaded.decode(codec.encode(unseen)) == codec.decode(codec.encode(unseen))).all()
    assert loaded.chunk_t == 5 and loaded.fit_info == codec.fit_info
    assert loaded.datasets == ["d0", "d1"]


def test_use_before_fit_and_empty_datasets_are_rejected():
    for call in (
        lambda c: c.encode(np.zeros((1, *FRAME))),
        lambda c: c.decode(np.zeros((1, 3))),
        lambda c: c.save("unused"),
        lambda c: c.parameter_count(),
        lambda c: c.latent_shape,
    ):
        with pytest.raises(RuntimeError, match="not fitted"):
            call(PODCodec())
    with pytest.raises(ValueError):
        PODCodec().fit({})
    with pytest.raises(ValueError):
        PODCodec().fit({"a": np.zeros((0, *FRAME))})
