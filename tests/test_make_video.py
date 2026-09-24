import numpy as np
import pytest
from make_video import compute_field


def _block():
    rng = np.random.default_rng(0)
    return rng.standard_normal((3, 2, 4, 5))


def test_compute_field_u_x_and_u_y_select_the_right_channel():
    block = _block()
    np.testing.assert_array_equal(compute_field(block, "u_x", dx=1.0, dy=1.0), block[:, 0])
    np.testing.assert_array_equal(compute_field(block, "u_y", dx=1.0, dy=1.0), block[:, 1])


def test_compute_field_speed_is_non_negative_and_matches_formula():
    block = _block()
    speed = compute_field(block, "speed", dx=1.0, dy=1.0)
    expected = np.sqrt(block[:, 0] ** 2 + block[:, 1] ** 2)
    np.testing.assert_allclose(speed, expected)
    assert (speed >= 0).all()


def test_compute_field_vorticity_matches_solid_body_rotation():
    nx, ny = 6, 5
    x = np.arange(nx)[:, None] * np.ones((1, ny))
    y = np.ones((nx, 1)) * np.arange(ny)[None, :]
    block = np.stack([-y, x])[None]

    w = compute_field(block, "vorticity", dx=1.0, dy=1.0)

    np.testing.assert_allclose(w, 2.0)


def test_compute_field_rejects_unknown_field():
    with pytest.raises(ValueError):
        compute_field(_block(), "pressure", dx=1.0, dy=1.0)
