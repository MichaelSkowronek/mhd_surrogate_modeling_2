import numpy as np
import pytest

from mhd_surrogate.analysis.spectral import (
    dominant_periods,
    format_peaks,
    interpolated_peak,
    power_spectrum,
    welch_spectrum,
)


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


def test_welch_spectrum_recovers_known_period():
    n = 800
    t0 = 40  # divides nperseg (200) exactly, for a crisp peak
    t = np.arange(n)
    series = np.sin(2 * np.pi * t / t0)

    omega, power = welch_spectrum(series, nperseg=200, noverlap=100)

    peak = np.argmax(power[1:]) + 1
    period = 2 * np.pi / omega[peak]
    assert period == pytest.approx(t0, rel=0.02)


def test_welch_spectrum_reduces_variance_of_a_white_noise_estimate():
    """Welch's method trades frequency resolution for a lower-variance PSD
    estimate: averaging n_segments independent (non-overlapping) segments
    should reduce the variance of the power estimate at a given frequency by
    roughly 1/n_segments relative to a single periodogram over one segment.
    """
    rng = np.random.default_rng(0)
    nperseg, n_segments = 64, 8
    n = nperseg * n_segments
    n_trials = 200
    bin_idx = 10  # an arbitrary interior frequency bin

    periodogram_vals = []
    welch_vals = []
    for _ in range(n_trials):
        series = rng.standard_normal(n)
        _, p = power_spectrum(series[:nperseg])
        periodogram_vals.append(p[bin_idx])
        _, w = welch_spectrum(series, nperseg=nperseg, noverlap=0)
        welch_vals.append(w[bin_idx])

    periodogram_var = np.var(periodogram_vals)
    welch_var = np.var(welch_vals)
    # Generous vs. the ~8x reduction expected in theory for independent segments.
    assert welch_var < periodogram_var / 3


def test_welch_spectrum_rejects_noverlap_not_less_than_nperseg():
    with pytest.raises(ValueError):
        welch_spectrum(np.zeros(100), nperseg=50, noverlap=50)


def test_welch_spectrum_rejects_series_shorter_than_nperseg():
    with pytest.raises(ValueError):
        welch_spectrum(np.zeros(30), nperseg=50, noverlap=0)


def test_format_peaks_formats_period_and_power_fraction():
    peaks = [
        {"period": 24.567, "omega": 0.25, "power_fraction": 0.314},
        {"period": 8.0, "omega": 0.9, "power_fraction": 0.05},
    ]
    assert format_peaks(peaks) == "T=24.6 (31.4%), T=8.0 (5.0%)"


def test_format_peaks_empty_list():
    assert format_peaks([]) == ""


@pytest.mark.parametrize("period", [30.0, 36.0, 41.5])
def test_interpolated_peak_recovers_a_period_that_falls_between_bins(period):
    """With 200-step segments the bins are at periods 200/k (40, 33.3, ...),
    so these tones all fall between bins; the refined peak is within 3%."""
    t = np.arange(200)
    omega, power = power_spectrum(np.sin(2 * np.pi * t / period))

    peak_omega, peak_power = interpolated_peak(omega, power)

    assert 2 * np.pi / peak_omega == pytest.approx(period, rel=0.03)
    assert peak_power >= power[1:].max() * 0.99


def test_interpolated_peak_does_not_refine_at_the_edges_of_the_spectrum():
    omega = np.arange(6.0)
    rising = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])  # peak at the last bin
    falling = np.array([9.0, 5.0, 4.0, 3.0, 2.0, 1.0])  # peak at the first bin after 0

    assert interpolated_peak(omega, rising) == (5.0, 5.0)
    assert interpolated_peak(omega, falling) == (1.0, 5.0)
