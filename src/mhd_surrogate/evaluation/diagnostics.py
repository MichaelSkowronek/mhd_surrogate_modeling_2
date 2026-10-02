"""Long-horizon physics diagnostics: does a forecast stay on the attractor?

Compare scalar summaries of the predicted and true series that a model can
get wrong even while its pointwise error is no worse than the unavoidable
decorrelation: kinetic energy, enstrophy, how incompressible the field stays
and whether the small scales survive (smoothing toward the mean kills them).

Everything works on (T, 2, Nx, Ny) arrays processed `chunk_t` steps at a time,
so a long series is never held in memory in float64 at once.
"""

from __future__ import annotations

import numpy as np

from mhd_surrogate.analysis.fields import divergence, vorticity
from mhd_surrogate.analysis.spectral import spectrum_sum

DEFAULT_CHUNK_T = 32
# Floor so an empty spectral bin doesn't turn the log ratio into inf/nan.
SPECTRUM_FLOOR = 1e-30


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
    plus the time-averaged 1D spectra along x and y, shape (C, K) each."""
    n = series.shape[0]
    energy, ens, div = [], [], []
    sums = {"x": 0.0, "y": 0.0}
    for start in range(0, n, chunk_t):
        block = np.asarray(series[start : start + chunk_t], dtype=np.float64)
        energy.append(kinetic_energy(block))
        ens.append(enstrophy(block, dx, dy))
        div.append(rms_divergence(block, dx, dy))
        sums["x"] = sums["x"] + spectrum_sum(block, 2, dx)
        sums["y"] = sums["y"] + spectrum_sum(block, 3, dy)
    return {
        "energy": np.concatenate(energy),
        "enstrophy": np.concatenate(ens),
        "rms_divergence": np.concatenate(div),
        "spectrum_x": sums["x"] / n,
        "spectrum_y": sums["y"] / n,
    }


def log_spectral_distance(predicted: np.ndarray, true: np.ndarray) -> float:
    """Mean |log10(P_pred / P_true)| over channels and nonzero wavenumbers of
    (C, K) spectra: 0 for a perfect spectrum, 0.3 for a factor-2 error."""
    ratio = np.log10((predicted[:, 1:] + SPECTRUM_FLOOR) / (true[:, 1:] + SPECTRUM_FLOOR))
    return float(np.abs(ratio).mean())


def compare_diagnostics(
    prediction, target, dx: float, dy: float, chunk_t: int = DEFAULT_CHUNK_T
) -> dict[str, float]:
    """Flat dict of long-horizon scores (ready for `mlflow.log_metrics`):

    - `energy_rel_error`, `enstrophy_rel_error`: |mean_pred - mean_true| /
      mean_true of the time-averaged value.
    - `divergence_ratio`: predicted over true time-mean RMS divergence (the
      DNS field is only approximately divergence-free on this grid, so the
      target's own value is the reference, not zero).
    - `spectrum_x_lsd`, `spectrum_y_lsd`: log-spectral distance of the
      time-averaged spectra.
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
    }
