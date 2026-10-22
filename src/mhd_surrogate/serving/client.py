"""A client for the forecast service (`serving/app.py`).

    client = ForecastClient("http://localhost:8000")
    info = client.model_info()
    forecast = client.forecast(context, n_steps=837)   # (members, 837, C, Nx, Ny)

`forecast_blocks` streams instead, a block at a time, for forecasts too
large to hold. A stream that ends before every member's `n_steps` frames
arrived (the server failed mid-forecast, the connection dropped) raises
`ForecastError`, as does an error response.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import httpx
import numpy as np

from mhd_surrogate.serving.schemas import PROBE_HEADER, ModelInfo
from mhd_surrogate.serving.stream import MEDIA_TYPE, encode, read_arrays

# A forecast block can take a while on a CPU; this bounds the wait for each
# one (httpx's timeouts are per read, not per response).
DEFAULT_TIMEOUT_S = 300.0


class ForecastError(RuntimeError):
    """An error response (`status`, the server's `detail`), or a forecast
    stream that ended early (`status` 200)."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class ForecastClient:
    """The forecast service at `base_url`, or behind `http` (any
    `httpx.Client` with the service as its base URL, e.g. a FastAPI
    `TestClient`)."""

    def __init__(
        self,
        base_url: str | None = None,
        http: httpx.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ):
        if (base_url is None) == (http is None):
            raise ValueError("give either base_url or http")
        self.http = http or httpx.Client(base_url=base_url, timeout=timeout)

    def model_info(self) -> ModelInfo:
        response = self.http.get("/v1/model")
        _raise_for_error(response)
        return ModelInfo.model_validate_json(response.content)

    def forecast_blocks(
        self,
        context: np.ndarray,
        n_steps: int,
        seeds: Sequence[int] = (0,),
        block_steps: int | None = None,
        probe: bool = False,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """(member, block) pairs of the forecast of `n_steps` frames from
        `context` (float32 frames, raw units), member by member: a
        stochastic model's sample for each of `seeds`, a deterministic
        model's single forecast. A `probe` (a synthetic check) is kept out
        of the service's ML metrics."""
        params: dict = {"n_steps": n_steps, "seeds": list(seeds)}
        if block_steps is not None:
            params["block_steps"] = block_steps
        headers = {"Content-Type": MEDIA_TYPE}
        if probe:
            headers[PROBE_HEADER] = "1"
        with self.http.stream(
            "POST",
            "/v1/forecast",
            params=params,
            content=encode(np.asarray(context, dtype=np.float32)),
            headers=headers,
        ) as response:
            if response.status_code != 200:
                response.read()
                _raise_for_error(response)
            members = int(response.headers["X-Members"])
            received = 0
            try:
                for block in read_arrays(response.iter_bytes()):
                    yield received // n_steps, block
                    received += len(block)
            except ValueError as error:  # a stream cut inside a record
                raise ForecastError(200, f"the forecast stream broke off: {error}") from None
            if received != members * n_steps:
                raise ForecastError(
                    200, f"the forecast stream ended after {received} of {members * n_steps} frames"
                )

    def forecast(
        self,
        context: np.ndarray,
        n_steps: int,
        seeds: Sequence[int] = (0,),
        block_steps: int | None = None,
        probe: bool = False,
    ) -> np.ndarray:
        """The whole forecast, shape (members, n_steps, C, Nx, Ny)."""
        members: dict[int, list[np.ndarray]] = {}
        for member, block in self.forecast_blocks(context, n_steps, seeds, block_steps, probe):
            members.setdefault(member, []).append(block)
        return np.stack([np.concatenate(blocks) for _, blocks in sorted(members.items())])

    def close(self) -> None:
        self.http.close()


def _raise_for_error(response: httpx.Response) -> None:
    if response.status_code >= 400:
        try:
            detail = response.json()["detail"]
        except ValueError, KeyError, TypeError:
            detail = response.text
        raise ForecastError(response.status_code, str(detail))
