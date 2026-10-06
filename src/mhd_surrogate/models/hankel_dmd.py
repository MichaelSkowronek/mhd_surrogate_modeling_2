"""Time-delay (Hankel) DMD: DMD on the last `delays` frames instead of one.

Plain DMD (`dmd.py`) advances a single frame, so everything it knows about
where the flow is going has to be in that frame. Stacking the last `d`
frames into the state lets the linear operator use the recent history too --
the phase of an oscillation, say, which a single snapshot of a standing
pattern doesn't determine. It answers the question the forecast protocol's
80-step context was sized for: does more context help? (`window` = `d`.)

POD first, then delays. Delay-embedding the raw frames would make the state
d x ~292k entries and its POD basis d x ~880 MB at rank 750 (~70 GB at
d = 80). Instead each frame is first reduced to its coefficients a_t in the
plain-DMD POD basis U (`spatial_rank` modes), and the delays are taken of
those: h_t = [a_t, a_{t-1}, ..., a_{t-d+1}], at most 80 x 750 = 60k entries.
Only U is stored at full size. The cost is that the delays see each frame's
`spatial_rank`-mode projection, not the frame: at rank 750 that keeps 98% of
the variance, and what it drops is small-scale content no linear model
carries forward anyway.

Fit. Everything comes from the one Gram matrix plain DMD already computes:
- U and its singular values from the "before" frames, as in DMD.
- Every frame's coefficients a = Z U = G[:, before] V S^-1 (no second pass
  over the frames).
- The delay states' Gram matrix as a sum of shifted blocks of a a^T,
  G_H[i, j] = sum_{k<d} (a a^T)[t_i - k, t_j - k], so the (pairs x d*r_s)
  delay matrix is never formed. Delay states never reach across a dataset
  boundary: each dataset loses its first d - 1 frames as starting points.
- Projected exact DMD on the delay states, truncated to `rank`: the delay
  basis U_H = H V_H S_H^-1, stored as d blocks of (spatial_rank x rank), and
  the reduced operator A~ = S_H^-1 V_H^T (H^T H') V_H S_H^-1.

With `delays=1` this is plain DMD at `min(rank, spatial_rank)` exactly: U is
the POD of the "before" frames, so the coefficients of those frames are
V S, already orthogonal, and the delay POD just picks their leading columns.

Forecast. The last `d` context frames are projected onto U, stacked newest
first and projected onto U_H; the result is evolved with A~'s
eigendecomposition (with `stabilize`, as in DMD), and each forecast state's
newest block is mapped back through U to a frame.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np

from mhd_surrogate.models.base import CHECKPOINT_META
from mhd_surrogate.models.dmd import (
    DEFAULT_CHUNK_T,
    DEFAULT_COLUMN_BLOCK,
    DEFAULT_RANK,
    DMD,
    eigendynamics,
    pair_indices,
    truncated_pod,
)

log = logging.getLogger(__name__)

DEFAULT_DELAYS = 8


def delay_pair_indices(lengths: list[int], delays: int) -> tuple[np.ndarray, np.ndarray]:
    """Row indices (into the stacked frames of datasets with these lengths)
    of the newest frame of the "before" and "after" delay state of every
    snapshot pair: a state needs its `delays - 1` predecessors in the same
    dataset, and the "after" state must be in it too."""
    before = []
    offset = 0
    for length in lengths:
        before.append(np.arange(offset + delays - 1, offset + length - 1))
        offset += length
    before = np.concatenate(before)
    return before, before + 1


def delay_gram(products: np.ndarray, rows: np.ndarray, cols: np.ndarray, delays: int) -> np.ndarray:
    """Inner products of the delay states whose newest frames are `rows` and
    `cols`, from the frames' inner products `products`: the sum, over the
    delays, of the blocks shifted back by that many frames."""
    gram = np.zeros((len(rows), len(cols)))
    for k in range(delays):
        gram += products[np.ix_(rows - k, cols - k)]
    return gram


class HankelDMD(DMD):
    name = "hankel_dmd"

    def __init__(
        self,
        rank: int = DEFAULT_RANK,
        delays: int = DEFAULT_DELAYS,
        spatial_rank: int = DEFAULT_RANK,
        stabilize: bool = True,
        chunk_t: int = DEFAULT_CHUNK_T,
        column_block: int = DEFAULT_COLUMN_BLOCK,
    ) -> None:
        if delays < 1:
            raise ValueError(f"delays must be >= 1, got {delays}")
        super().__init__(spatial_rank, stabilize, chunk_t, column_block)
        self.rank = rank
        self.delays = delays
        self.spatial_rank = spatial_rank
        self.window = delays
        self.hankel_basis: np.ndarray | None = None  # (delays, spatial_rank, rank), U_H

    # -- fitting -------------------------------------------------------------

    def fit(self, datasets: Mapping[str, Any]) -> None:
        lengths = [array.shape[0] for array in datasets.values()]
        if not lengths or min(lengths) < self.delays + 1:
            raise ValueError(
                f"HankelDMD.fit with delays={self.delays} needs datasets with at least "
                f"{self.delays + 1} frames each"
            )
        frames, lengths = self._load(datasets)
        self._normalize_in_place(frames)
        before, _ = pair_indices(lengths)

        gram = self._gram(frames)
        s2, s, v = truncated_pod(
            gram[np.ix_(before, before)], self.spatial_rank, f"{self.name} spatial"
        )
        self.basis = self._basis(frames, before, v / s)
        del frames
        # Every frame's POD coefficients, a = Z U = Z X^T V S^-1, from the Gram matrix.
        coefficients = gram[:, before] @ (v / s)
        del gram

        delay_before, delay_after = delay_pair_indices(lengths, self.delays)
        products = coefficients @ coefficients.T
        h2, h, w = truncated_pod(
            delay_gram(products, delay_before, delay_before, self.delays),
            self.rank,
            f"{self.name} delay",
        )
        r = len(h)
        cross = delay_gram(products, delay_before, delay_after, self.delays)
        reduced = (w / h).T @ cross @ (w / h)
        eigenvalues, eigenvectors, n_unstable = eigendynamics(reduced, self.stabilize)

        # U_H = H V_H S_H^-1, one (spatial_rank x rank) block per delay.
        self.hankel_basis = np.stack(
            [coefficients[delay_before - k].T @ (w / h) for k in range(self.delays)]
        ).astype(np.float32)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.datasets = list(datasets)
        self.fit_info = {
            "rank": float(r),
            "spatial_rank": float(len(s)),
            "delays": float(self.delays),
            # Of the frames' variance, kept by U; of the delay states' (in
            # U's coordinates), kept by U_H.
            "spatial_explained_variance": float(s2[: len(s)].sum() / s2.sum()),
            "delay_explained_variance": float(h2[:r].sum() / h2.sum()),
            "unstable_modes": float(n_unstable),
            "n_pairs": float(len(delay_before)),
        }
        log.info("%s: %s", self.name, ", ".join(f"{k}={v:.4g}" for k, v in self.fit_info.items()))

    # -- forecasting ---------------------------------------------------------

    def predict(self, context: np.ndarray, n_steps: int) -> np.ndarray:
        if self.hankel_basis is None:
            raise RuntimeError("HankelDMD is not fitted")
        if len(context) < self.delays:
            raise ValueError(f"context has {len(context)} frames, need delays={self.delays}")
        frame_shape = self.mean.shape
        recent = np.asarray(context[len(context) - self.delays :])
        z = ((recent - self.mean[None]) / self.scale[None, :, None, None]).reshape(self.delays, -1)
        # Delay block k holds the frame k steps before the newest one.
        coefficients = (z @ self.basis)[::-1]
        projected = np.einsum("kij,ki->j", self.hankel_basis, coefficients)
        amplitudes = np.linalg.solve(self.eigenvectors, projected)
        powers = self.eigenvalues[None, :] ** np.arange(1, n_steps + 1)[:, None]
        reduced = (powers * amplitudes[None, :]) @ self.eigenvectors.T
        newest = (reduced @ self.hankel_basis[0].T).real.astype(np.float32)
        frames = (newest @ self.basis.T).reshape(n_steps, *frame_shape)
        return frames * self.scale[None, :, None, None] + self.mean[None]

    # -- checkpoint ----------------------------------------------------------

    def save(self, directory: Path) -> None:
        if self.hankel_basis is None:
            raise RuntimeError("HankelDMD is not fitted")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "hankel_dmd.npz",
            mean=self.mean,
            scale=self.scale,
            basis=self.basis,
            hankel_basis=self.hankel_basis,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
        )
        meta = {
            "name": self.name,
            "rank": self.rank,
            "delays": self.delays,
            "spatial_rank": self.spatial_rank,
            "stabilize": self.stabilize,
            "chunk_t": self.chunk_t,
            "column_block": self.column_block,
            "datasets": self.datasets,
            "fit_info": self.fit_info,
        }
        (directory / CHECKPOINT_META).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> Self:
        directory = Path(directory)
        meta = json.loads((directory / CHECKPOINT_META).read_text())
        model = cls(
            meta["rank"],
            meta["delays"],
            meta["spatial_rank"],
            meta["stabilize"],
            meta["chunk_t"],
            meta["column_block"],
        )
        with np.load(directory / "hankel_dmd.npz") as arrays:
            model.mean = arrays["mean"]
            model.scale = arrays["scale"]
            model.basis = arrays["basis"]
            model.hankel_basis = arrays["hankel_basis"]
            model.eigenvalues = arrays["eigenvalues"]
            model.eigenvectors = arrays["eigenvectors"]
        model.datasets = meta["datasets"]
        model.fit_info = meta["fit_info"]
        return model
