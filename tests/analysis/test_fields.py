import numpy as np

from mhd_surrogate.analysis.fields import vorticity


def test_vorticity_of_solid_body_rotation_is_constant():
    """u_x = -y, u_y = x is a rigid rotation with vorticity 2 everywhere.

    Both fields are linear in x/y, so central (and one-sided, at the
    boundaries) finite differences are exact, not just approximate.
    """
    nx, ny = 6, 5
    x = np.arange(nx)[:, None] * np.ones((1, ny))
    y = np.ones((nx, 1)) * np.arange(ny)[None, :]
    ux, uy = -y, x
    block = np.stack([ux, uy])[None]  # (1, 2, nx, ny)

    w = vorticity(block, dx=1.0, dy=1.0)

    assert w.shape == (1, nx, ny)
    np.testing.assert_allclose(w, 2.0)


def test_vorticity_scales_inversely_with_spacing():
    nx, ny = 6, 5
    x = np.arange(nx)[:, None] * np.ones((1, ny))
    y = np.ones((nx, 1)) * np.arange(ny)[None, :]
    block = np.stack([-y, x])[None]

    w_unit = vorticity(block, dx=1.0, dy=1.0)
    w_scaled = vorticity(block, dx=2.0, dy=2.0)

    np.testing.assert_allclose(w_scaled, w_unit / 2.0)


def test_vorticity_of_uniform_flow_is_zero():
    block = np.ones((3, 2, 4, 4))  # constant u_x = u_y = 1 everywhere
    w = vorticity(block, dx=1.0, dy=1.0)
    np.testing.assert_allclose(w, 0.0)
