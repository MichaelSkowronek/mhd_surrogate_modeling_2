import numpy as np
import pytest
from make_forecast_video import forecast_limits


def test_limits_of_a_signed_field_are_symmetric_and_the_diff_matches_their_size():
    sample = np.linspace(-1.0, 3.0, 1001)  # 1st/99th percentiles: -0.96, 2.96

    vmin, vmax, diff = forecast_limits(sample, symmetric=True)

    assert (vmin, vmax) == pytest.approx((-2.96, 2.96))
    assert diff == pytest.approx(2.96)


def test_limits_of_an_unsigned_field_follow_the_percentiles_and_diff_is_half_the_range():
    sample = np.linspace(0.0, 10.0, 1001)  # 1st/99th percentiles: 0.1, 9.9

    vmin, vmax, diff = forecast_limits(sample, symmetric=False)

    assert (vmin, vmax) == pytest.approx((0.1, 9.9))
    assert diff == pytest.approx(4.9)
