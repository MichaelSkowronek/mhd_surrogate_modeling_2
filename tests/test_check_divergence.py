import numpy as np
from check_divergence import divergence_stats


def _linear_velocity_field(nx: int, ny: int, ux_coeff: float, uy_coeff: float):
    x = np.arange(nx)[:, None] * np.ones((1, ny))
    y = np.ones((nx, 1)) * np.arange(ny)[None, :]
    ux, uy = ux_coeff * x, uy_coeff * y
    return np.stack([ux, uy])[None]  # (1, 2, nx, ny)


def test_divergence_is_zero_for_divergence_free_field():
    # u_x = x, u_y = -y -> du_x/dx + du_y/dy = 1 - 1 = 0 everywhere.
    block = _linear_velocity_field(10, 8, ux_coeff=1.0, uy_coeff=-1.0)

    rms_div, rel_div, snapshots = divergence_stats(
        block, dx=1.0, dy=1.0, chunk_t=1, snapshot_steps=[0]
    )

    np.testing.assert_allclose(rms_div, 0.0, atol=1e-10)
    np.testing.assert_allclose(rel_div, 0.0, atol=1e-10)
    np.testing.assert_allclose(snapshots[0], 0.0, atol=1e-10)


def test_divergence_matches_known_nonzero_value():
    # u_x = x, u_y = y -> du_x/dx + du_y/dy = 1 + 1 = 2 everywhere.
    block = _linear_velocity_field(10, 8, ux_coeff=1.0, uy_coeff=1.0)

    rms_div, rel_div, _ = divergence_stats(block, dx=1.0, dy=1.0, chunk_t=1, snapshot_steps=[0])

    np.testing.assert_allclose(rms_div, 2.0)
    # normalized = rms(div) / rms(du_x/dx, du_y/dy) = 2 / sqrt((1^2+1^2)) = sqrt(2)
    np.testing.assert_allclose(rel_div, np.sqrt(2.0), rtol=1e-6)


def test_divergence_scales_inversely_with_spacing():
    block = _linear_velocity_field(10, 8, ux_coeff=1.0, uy_coeff=1.0)

    rms_unit, _, _ = divergence_stats(block, dx=1.0, dy=1.0, chunk_t=1, snapshot_steps=[0])
    rms_scaled, _, _ = divergence_stats(block, dx=2.0, dy=2.0, chunk_t=1, snapshot_steps=[0])

    np.testing.assert_allclose(rms_scaled, rms_unit / 2.0)
