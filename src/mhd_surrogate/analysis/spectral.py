"""Temporal power spectrum estimators, shared by scripts/analysis/check_point_spectrum.py
and check_spatial_mean_spectrum.py.

Two estimators: a plain periodogram (`power_spectrum`, one FFT over the
whole series) and Welch's method (`welch_spectrum`, the average of
overlapping segments' periodograms). A single periodogram bin is a
high-variance estimate (effectively 2 degrees of freedom, i.e. its own value
is not strong evidence against being a random fluctuation of white noise);
Welch's method trades frequency resolution for a much lower-variance
estimate by averaging, so a peak that survives in Welch too is real, not an
artifact of that variance.
"""

from __future__ import annotations

import numpy as np


def power_spectrum(series: np.ndarray, spacing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectrum of a 1D real time series via FFT.

    Detrends (removes the mean) and applies a Hann window to limit spectral
    leakage (the series is a finite, non-periodic excerpt), then computes a
    one-sided periodogram normalized so that integrating `power` over the
    angular frequency axis returns the window-weighted variance of the
    series -- same convention as check_spectrum.py's spatial spectra.

    Returns (omega, power), each shape (n//2+1,): angular frequency in
    radians per snapshot step (the physical time step is not stored in the
    data) and power spectral density.
    """
    n = series.shape[0]
    window = np.hanning(n)
    detrended = series - series.mean()
    transform = np.fft.rfft(detrended * window)
    power = np.abs(transform) ** 2 * spacing / (window**2).sum() / (2 * np.pi)
    last = -1 if n % 2 == 0 else None
    power[1:last] *= 2
    omega = 2 * np.pi * np.fft.rfftfreq(n, d=spacing)
    return omega, power


def welch_spectrum(
    series: np.ndarray, nperseg: int, noverlap: int, spacing: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Welch's method: average the periodogram (via `power_spectrum`, so each
    segment is independently detrended and Hann-windowed) over overlapping
    segments of length `nperseg`, stepping by `nperseg - noverlap`.

    Trades frequency resolution (bins are `nperseg`-wide, not
    `series`-length-wide) for a lower-variance power spectral density
    estimate -- see this module's docstring for why that matters here.

    Returns (omega, power), each shape (nperseg // 2 + 1,); same
    units/normalization as `power_spectrum`.
    """
    if noverlap >= nperseg:
        raise ValueError(f"noverlap ({noverlap}) must be < nperseg ({nperseg})")
    n = series.shape[0]
    if n < nperseg:
        raise ValueError(f"series length ({n}) must be >= nperseg ({nperseg})")

    step = nperseg - noverlap
    starts = range(0, n - nperseg + 1, step)
    omega = None
    powers = []
    for start in starts:
        omega, power = power_spectrum(series[start : start + nperseg], spacing)
        powers.append(power)
    return omega, np.mean(powers, axis=0)


def dominant_periods(omega: np.ndarray, power: np.ndarray, n_peaks: int) -> list[dict]:
    """Top `n_peaks` local-maximum peaks of `power`, ranked by power, excluding
    the zero-frequency bin (omega[0]).

    A local maximum is a bin whose power exceeds both immediate neighbors;
    the highest-frequency bin (Nyquist) is never counted, since it has no
    right neighbor to compare against. Returns a list of
    {"period", "omega", "power_fraction"} (power_fraction relative to the
    total power excluding the zero-frequency bin), most dominant first.
    """
    total = power[1:].sum()
    is_peak = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
    peak_idx = np.flatnonzero(is_peak) + 1
    ranked = peak_idx[np.argsort(power[peak_idx])[::-1]][:n_peaks]
    return [
        {
            "period": 2 * np.pi / omega[i],
            "omega": omega[i],
            "power_fraction": power[i] / total,
        }
        for i in ranked
    ]


def format_peaks(peaks: list[dict]) -> str:
    return ", ".join(f"T={p['period']:.1f} ({p['power_fraction']:.1%})" for p in peaks)
