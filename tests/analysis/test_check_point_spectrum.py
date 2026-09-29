import numpy as np
import pytest
from check_point_spectrum import dominant_periods, power_spectrum


def test_power_spectrum_recovers_amplitude_and_period_of_a_sine():
    """A unit-amplitude sine has variance 0.5 and all its power at period T0."""
    n = 1000
    t0 = 50  # chosen so n/t0 is an integer: the frequency lands exactly on a bin
    t = np.arange(n)
    series = np.sin(2 * np.pi * t / t0)

    omega, power = power_spectrum(series)

    domega = omega[1] - omega[0]
    integral = power[1:].sum() * domega
    assert integral == pytest.approx(0.5, rel=0.02)

    peak = np.argmax(power[1:]) + 1
    period = 2 * np.pi / omega[peak]
    assert period == pytest.approx(t0, rel=0.02)


def test_dominant_periods_ranks_two_known_tones_by_power():
    n = 2000
    t_slow, t_fast = 200, 25  # both divide n exactly
    t = np.arange(n)
    series = 1.0 * np.sin(2 * np.pi * t / t_slow) + 0.4 * np.sin(2 * np.pi * t / t_fast)

    omega, power = power_spectrum(series)
    peaks = dominant_periods(omega, power, n_peaks=2)

    assert len(peaks) == 2
    assert peaks[0]["period"] == pytest.approx(t_slow, rel=0.02)
    assert peaks[1]["period"] == pytest.approx(t_fast, rel=0.02)
    assert peaks[0]["power_fraction"] > peaks[1]["power_fraction"]


def test_dominant_periods_excludes_zero_frequency_bin():
    # A signal with a large mean offset would dominate the raw FFT at omega=0
    # if it weren't detrended inside power_spectrum; the peak found here must
    # still be the oscillation, not the (already-removed) DC component.
    n = 500
    t0 = 20
    t = np.arange(n)
    series = 100.0 + np.sin(2 * np.pi * t / t0)

    omega, power = power_spectrum(series)
    peaks = dominant_periods(omega, power, n_peaks=1)

    assert peaks[0]["period"] == pytest.approx(t0, rel=0.02)
