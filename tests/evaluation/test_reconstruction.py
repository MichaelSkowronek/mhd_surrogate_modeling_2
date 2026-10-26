import numpy as np
import pytest

from mhd_surrogate.evaluation.reconstruction import reconstruct, reconstruction_scores

SHAPE = (2, 8, 6)
DX = DY = 0.1


class Scaling:
    """A codec whose reconstruction is `factor` times the frame."""

    def __init__(self, factor):
        self.factor = factor

    def encode(self, frames):
        return np.asarray(frames).reshape(len(frames), -1)

    def decode(self, latents):
        return self.factor * latents.reshape(-1, *SHAPE)


def series(n=256, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)[:, None, None]
    x = np.arange(SHAPE[1])[None, :, None]
    u_x = np.sin(0.5 * x - 0.2 * t) + 0.1 * rng.normal(size=(n, *SHAPE[1:]))
    u_y = np.cos(2 * np.pi * t / 25) + 0.1 * rng.normal(size=(n, *SHAPE[1:]))
    return np.stack([u_x, u_y], axis=1).astype(np.float32)


def test_a_perfect_reconstruction_scores_perfectly():
    truth = series()

    scores = reconstruction_scores(Scaling(1.0), truth, np.array([1.0, 2.0]), DX, DY)

    assert scores["reconstruction_rmse"] == 0
    for name in (
        "energy_rel_error",
        "enstrophy_rel_error",
        "spectrum_x_lsd",
        "u_x_mean_offset",
        "u_y_mean_offset",
    ):
        assert scores[name] == pytest.approx(0, abs=1e-6), name
    assert scores["divergence_ratio"] == pytest.approx(1)
    assert scores["u_x_mean_std_ratio"] == pytest.approx(1)
    assert scores["u_y_mean_std_ratio"] == pytest.approx(1)


def test_a_halved_reconstruction_has_the_known_errors():
    truth = series()
    scale = np.array([1.0, 2.0])

    scores = reconstruction_scores(Scaling(0.5), truth, scale, DX, DY)

    # The error is half the frame; energy and enstrophy are quadratic.
    expected = 0.5 * np.sqrt(np.mean((truth / scale[None, :, None, None]) ** 2))
    assert scores["reconstruction_rmse"] == pytest.approx(expected, rel=1e-5)
    assert scores["energy_rel_error"] == pytest.approx(0.75, rel=1e-5)
    assert scores["enstrophy_rel_error"] == pytest.approx(0.75, rel=1e-5)
    assert scores["u_x_mean_std_ratio"] == pytest.approx(0.5, rel=1e-5)
    assert scores["u_y_mean_std_ratio"] == pytest.approx(0.5, rel=1e-5)


def test_reconstruction_does_not_depend_on_the_chunk_size():
    truth = series(n=10)

    assert (reconstruct(Scaling(0.5), truth, chunk_t=3) == reconstruct(Scaling(0.5), truth)).all()
