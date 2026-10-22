import numpy as np
import pytest
from check_parity import relative_difference, rmse


def test_relative_difference_is_the_largest_gap_in_truth_stds():
    truth = np.array([-2.0, 2.0])  # std 2
    served = np.array([1.0, 1.0, 1.0])
    offline = np.array([1.0, 0.5, 1.0])

    assert relative_difference(served, offline, truth) == pytest.approx(0.25)
    assert relative_difference(served, served, truth) == 0.0


def test_rmse_is_in_truth_stds():
    truth = np.array([-2.0, 2.0])

    assert rmse(truth + 1.0, truth) == pytest.approx(0.5)
    assert rmse(truth, truth) == 0.0
