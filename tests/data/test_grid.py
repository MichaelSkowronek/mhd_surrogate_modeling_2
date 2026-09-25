import pytest

from mhd_surrogate.data.grid import grid_spacing


def test_grid_spacing_divides_length_by_intervals_not_points(tmp_path):
    config_path = tmp_path / "grid.yaml"
    config_path.write_text("lx: 10.0\nly: 4.0\n")

    dx, dy = grid_spacing(nx=11, ny=5, config_path=config_path)

    assert dx == pytest.approx(1.0)  # 10.0 / (11 - 1)
    assert dy == pytest.approx(1.0)  # 4.0 / (5 - 1)


def test_grid_spacing_matches_committed_re16k_grid(tmp_path):
    config_path = tmp_path / "grid.yaml"
    config_path.write_text("lx: 25.0\nly: 2.0\n")

    dx, dy = grid_spacing(nx=1151, ny=127, config_path=config_path)

    assert dx == pytest.approx(25.0 / 1150)
    assert dy == pytest.approx(2.0 / 126)
