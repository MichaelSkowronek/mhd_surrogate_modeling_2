"""Long-horizon physics diagnostics: does a forecast stay on the attractor?

Compare scalar summaries of the predicted and true series that a model can
get wrong even while its pointwise error is no worse than the unavoidable
decorrelation: kinetic energy, enstrophy, how incompressible the field stays
and whether the small scales survive (smoothing toward the mean kills them).

The spatial spectra check each snapshot's structure; the temporal ones check
the dynamics. The EDA's most robust feature is a coherent, domain-wide
oscillation of ~24-40 steps that shows up as a single sharp peak in the Welch
spectrum of the *domain-averaged* u_y (30-51% of its power in every dataset),
so the temporal scores are computed on the domain-averaged velocity: a model
can have the right spatial statistics at every instant and still drift, damp
or mistime that oscillation. (Welch with the EDA's segment length, since a
held-out series is only ~840 steps long: ~7 segments, so the estimates are
noisy and the frequency resolution coarse.)

Everything works on (T, 2, Nx, Ny) arrays processed `chunk_t` steps at a time,
so a long series is never held in memory in float64 at once.
"""

from __future__ import annotations

import numpy as np

from mhd_surrogate.analysis.fields import divergence, vorticity
from mhd_surrogate.analysis.spectral import interpolated_peak, spectrum_sum, welch_spectrum

DEFAULT_CHUNK_T = 32
# Welch segment length in steps, as in the EDA's temporal spectra
# (scripts/analysis/check_spatial_mean_spectrum.py); overlap defaults to half.
NPERSEG = 200
# Channel order of the data; u_y carries the coherent oscillation.
U_Y = 1
# Spectral power below this fraction of the true spectrum's peak counts as
# zero. A relative floor (not an absolute epsilon) so a flat forecast, whose
# power is ~0 everywhere, scores a bounded "several decades off" rather than a
# number set by an arbitrary constant.
SPECTRUM_FLOOR_RATIO = 1e-10


def kinetic_energy(block: np.ndarray) -> np.ndarray:
    """Spatial mean of 0.5 |u|^2 per step, shape (t,)."""
    return 0.5 * (block**2).sum(axis=1).mean(axis=(1, 2))


def enstrophy(block: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Spatial mean of 0.5 w^2 per step, shape (t,)."""
    return 0.5 * (vorticity(block, dx, dy) ** 2).mean(axis=(1, 2))


def rms_divergence(block: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Spatial RMS of the divergence per step, shape (t,)."""
    return np.sqrt((divergence(block, dx, dy) ** 2).mean(axis=(1, 2)))


def series_summary(series, dx: float, dy: float, chunk_t: int = DEFAULT_CHUNK_T) -> dict:
    """Per-step energy, enstrophy and RMS divergence (arrays of shape (T,)),
    the per-step domain-averaged velocity (T, C), plus the time-averaged 1D
    spectra along x and y, shape (C, K) each."""
    n = series.shape[0]
    energy, ens, div, mean_velocity = [], [], [], []
    sums = {"x": 0.0, "y": 0.0}
    for start in range(0, n, chunk_t):
        block = np.asarray(series[start : start + chunk_t], dtype=np.float64)
        energy.append(kinetic_energy(block))
        ens.append(enstrophy(block, dx, dy))
        div.append(rms_divergence(block, dx, dy))
        mean_velocity.append(block.mean(axis=(2, 3)))
        sums["x"] = sums["x"] + spectrum_sum(block, 2, dx)
        sums["y"] = sums["y"] + spectrum_sum(block, 3, dy)
    return {
        "energy": np.concatenate(energy),
        "enstrophy": np.concatenate(ens),
        "rms_divergence": np.concatenate(div),
        "mean_velocity": np.concatenate(mean_velocity),
        "spectrum_x": sums["x"] / n,
        "spectrum_y": sums["y"] / n,
    }


def log_spectral_distance(predicted: np.ndarray, true: np.ndarray) -> float:
    """Mean |log10(P_pred / P_true)| over channels and nonzero wavenumbers of
    (C, K) spectra: 0 for a perfect spectrum, 0.3 for a factor-2 error.

    Power below `SPECTRUM_FLOOR_RATIO` of each channel's true peak is raised
    to that floor in both spectra, which bounds the distance of a spectrum
    that is (nearly) empty to about 10.
    """
    predicted, true = predicted[:, 1:], true[:, 1:]
    floor = SPECTRUM_FLOOR_RATIO * true.max(axis=1, keepdims=True)
    ratio = np.log10(np.maximum(predicted, floor) / np.maximum(true, floor))
    return float(np.abs(ratio).mean())


def temporal_scores(
    predicted: np.ndarray, true: np.ndarray, nperseg: int = NPERSEG, noverlap: int | None = None
) -> dict[str, float]:
    """Scores comparing the Welch spectra of two (T, C) domain-averaged
    velocity series:

    - `temporal_u_x_lsd`, `temporal_u_y_lsd`: log-spectral distance per channel.
    - `u_y_period_error`: relative error of the dominant period of u_y (the
      EDA's coherent oscillation), peak refined between bins; nan if the
      forecast has no oscillation (zero power).
    - `u_y_peak_power_ratio`: predicted over true power at that peak, below 1
      when the oscillation is damped.
    """
    noverlap = nperseg // 2 if noverlap is None else noverlap
    omega, pred_power = None, []
    true_power = []
    for c in range(predicted.shape[1]):
        omega, p = welch_spectrum(predicted[:, c], nperseg, noverlap)
        pred_power.append(p)
        true_power.append(welch_spectrum(true[:, c], nperseg, noverlap)[1])
    pred_power, true_power = np.stack(pred_power), np.stack(true_power)

    pred_omega, pred_peak = interpolated_peak(omega, pred_power[U_Y])
    true_omega, true_peak = interpolated_peak(omega, true_power[U_Y])
    if pred_peak <= SPECTRUM_FLOOR_RATIO * true_peak:
        period_error = float("nan")  # no oscillation at all: its period is undefined
    else:
        # period = 2 pi / omega, so the relative period error is |true/pred - 1|.
        period_error = float(abs(true_omega / pred_omega - 1.0))
    return {
        "temporal_u_x_lsd": log_spectral_distance(pred_power[:1], true_power[:1]),
        "temporal_u_y_lsd": log_spectral_distance(pred_power[U_Y:], true_power[U_Y:]),
        "u_y_period_error": period_error,
        "u_y_peak_power_ratio": float(pred_peak / true_peak),
    }


def compare_diagnostics(
    prediction,
    target,
    dx: float,
    dy: float,
    chunk_t: int = DEFAULT_CHUNK_T,
    nperseg: int = NPERSEG,
) -> dict[str, float]:
    """Flat dict of long-horizon scores (ready for `mlflow.log_metrics`):

    - `energy_rel_error`, `enstrophy_rel_error`: |mean_pred - mean_true| /
      mean_true of the time-averaged value.
    - `divergence_ratio`: predicted over true time-mean RMS divergence (the
      DNS field is only approximately divergence-free on this grid, so the
      target's own value is the reference, not zero).
    - `spectrum_x_lsd`, `spectrum_y_lsd`: log-spectral distance of the
      time-averaged spectra.
    - the `temporal_scores` of the domain-averaged velocity (Welch segments of
      `nperseg` steps).
    """
    pred = series_summary(prediction, dx, dy, chunk_t)
    true = series_summary(target, dx, dy, chunk_t)

    def rel_error(key: str) -> float:
        return float(abs(pred[key].mean() - true[key].mean()) / true[key].mean())

    return {
        "energy_rel_error": rel_error("energy"),
        "enstrophy_rel_error": rel_error("enstrophy"),
        "divergence_ratio": float(pred["rms_divergence"].mean() / true["rms_divergence"].mean()),
        "spectrum_x_lsd": log_spectral_distance(pred["spectrum_x"], true["spectrum_x"]),
        "spectrum_y_lsd": log_spectral_distance(pred["spectrum_y"], true["spectrum_y"]),
        **temporal_scores(pred["mean_velocity"], true["mean_velocity"], nperseg),
    }
