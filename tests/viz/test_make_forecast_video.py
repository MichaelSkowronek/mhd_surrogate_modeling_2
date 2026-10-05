import numpy as np
import pytest
from make_forecast_video import check_dataset, forecast_limits

CONFIG = {"train_datasets": ["a", "b"], "val_dataset": "v", "test_dataset": "t"}


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


def test_check_dataset_refuses_the_test_dataset_and_unknown_names():
    check_dataset("v", CONFIG)
    check_dataset("b", CONFIG)
    with pytest.raises(SystemExit, match="test dataset"):
        check_dataset("t", CONFIG)
    with pytest.raises(SystemExit, match="not a validation or training"):
        check_dataset("x", CONFIG)
