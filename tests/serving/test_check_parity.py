import numpy as np
import pytest
from check_parity import rmse


def test_rmse_is_in_truth_stds():
    truth = np.array([-2.0, 2.0])

    assert rmse(truth + 1.0, truth) == pytest.approx(0.5)
    assert rmse(truth, truth) == 0.0
