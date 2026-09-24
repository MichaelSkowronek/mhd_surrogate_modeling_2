import numpy as np
import pytest
from check_autocorrelation import decorrelation_metrics, field_acf, series_acf


def _ar1(rng: np.random.Generator, n_steps: int, phi: float, shape_tail: tuple = ()) -> np.ndarray:
    """AR(1) process x_t = phi*x_{t-1} + noise, unit variance, known autocorrelation phi^lag."""
    x = np.empty((n_steps,) + shape_tail)
    x[0] = rng.standard_normal(shape_tail)
    noise = rng.standard_normal((n_steps,) + shape_tail) * np.sqrt(1 - phi**2)
    for t in range(1, n_steps):
        x[t] = phi * x[t - 1] + noise[t]
    return x


def test_field_acf_matches_ar1_theoretical_autocorrelation():
    rng = np.random.default_rng(0)
    n_steps, nx, ny, phi = 3000, 8, 6, 0.85
    # A mean offset and a per-channel scale must not affect the autocorrelation.
    data = np.stack(
        [_ar1(rng, n_steps, phi, (nx, ny)), 3.0 + 2.0 * _ar1(rng, n_steps, phi, (nx, ny))],
        axis=1,
    ).astype(np.float32)

    acf = field_acf(data, max_lag=20, slab=3)
    theory = phi ** np.arange(21)

    np.testing.assert_allclose(acf[0], theory, atol=0.05)
    np.testing.assert_allclose(acf[1], theory, atol=0.05)


def test_series_acf_matches_ar1_theoretical_autocorrelation():
    rng = np.random.default_rng(1)
    n_steps, phi = 5000, 0.7
    series = np.stack([_ar1(rng, n_steps, phi), _ar1(rng, n_steps, phi)], axis=1)

    acf = series_acf(series, max_lag=15)
    theory = phi ** np.arange(16)

    np.testing.assert_allclose(acf[0], theory, atol=0.05)
    np.testing.assert_allclose(acf[1], theory, atol=0.05)


def test_decorrelation_metrics_thresholds_and_integral_time():
    rho = np.array([1.0, 0.5, 0.2, 0.05, -0.1, -0.2])
    m = decorrelation_metrics(rho, n_steps=100)

    assert m["lag1"] == pytest.approx(0.5)
    assert m["one_over_e"] == 2  # first lag with rho <= 1/e ~= 0.368
    assert m["below_0.05"] == 3
    assert m["zero"] == 4
    assert m["tau_int"] == pytest.approx(0.5 + 0.5 + 0.2 + 0.05)
    assert m["n_eff"] == pytest.approx(100 / (2 * m["tau_int"]))


def test_decorrelation_metrics_when_series_never_decorrelates():
    rho = np.array([1.0, 0.9, 0.8, 0.7])
    m = decorrelation_metrics(rho, n_steps=10)
    assert m["zero"] is None
    assert m["tau_int"] is None
    assert m["n_eff"] is None
