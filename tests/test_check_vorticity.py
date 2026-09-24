import numpy as np
from check_vorticity import vorticity_stats


def test_vorticity_stats_solid_body_rotation():
    # u_x = -y, u_y = x: constant vorticity w = 2, so enstrophy = 0.5*2^2 = 2.
    nx, ny = 6, 5
    x = np.arange(nx)[:, None] * np.ones((1, ny))
    y = np.ones((nx, 1)) * np.arange(ny)[None, :]
    frame = np.stack([-y, x])
    arr = np.stack([frame, frame, frame])  # 3 identical time steps

    mean_vorticity, enstrophy, snapshots = vorticity_stats(
        arr, dx=1.0, dy=1.0, chunk_t=2, snapshot_steps=[0, 2]
    )

    np.testing.assert_allclose(mean_vorticity, 2.0)
    np.testing.assert_allclose(enstrophy, 2.0)
    assert set(snapshots) == {0, 2}
    np.testing.assert_allclose(snapshots[0], 2.0)


def test_vorticity_stats_chunking_does_not_change_result():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((11, 2, 4, 5))

    whole = vorticity_stats(arr, dx=0.1, dy=0.2, chunk_t=100, snapshot_steps=[0, 10])
    chunked = vorticity_stats(arr, dx=0.1, dy=0.2, chunk_t=3, snapshot_steps=[0, 10])

    np.testing.assert_allclose(whole[0], chunked[0])
    np.testing.assert_allclose(whole[1], chunked[1])
    for t in (0, 10):
        np.testing.assert_allclose(whole[2][t], chunked[2][t])
