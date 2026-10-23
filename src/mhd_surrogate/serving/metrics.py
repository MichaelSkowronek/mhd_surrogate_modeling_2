"""The forecast service's Prometheus metrics (`GET /metrics`).

Service metrics -- requests and their latency by route and status,
forecasts in flight, their duration and frames, forecasts that failed after
their response started -- and two ML ones. A served forecast has no truth
to be scored against, so these are proxies:

- **Input drift**: each context's per-channel mean and standard deviation,
  in units of the training data's (`reference`, which promotion attaches to
  the registered version): how far requests are from the data the model was
  fitted on. Training and validation contexts (4-frame windows) stay within
  +-0.025 stds of the mean and at 0.83-1.22 times the std.
- **Forecast energy**: each streamed block's mean kinetic energy over the
  context's, the quantity the selection rule's stability check bounds
  (README, "How candidates are ranked"): a blow-up shows as a growing
  ratio. The validation truth's 100-step blocks stay at 1.00-1.05 times its
  context's; Hankel DMD's forecast decays to 0.72 (it damps).

Synthetic requests (`X-Probe`, the deployment's smoke check) are left out
of both: their made-up contexts would read as drift. The buckets resolve
those ranges. The shift histogram, signed, has no
`_sum` (Prometheus' rule for histograms that can see negative values): its
quantiles come from the buckets, as every histogram's do. Each app has a
registry of its own, so several apps in one process (the tests) don't share
counters.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

import numpy as np
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from pydantic import BaseModel

from mhd_surrogate.evaluation.quantities import kinetic_energy

ENERGY_BUCKETS = (0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)
SHIFT_BUCKETS = (-1.0, -0.5, -0.25, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
SCALE_BUCKETS = (0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
DURATION_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)


class Reference(BaseModel):
    """The training data's per-channel mean and standard deviation (the
    normalization statistics, `data/normalization.py`)."""

    channel_names: list[str]
    mean: list[float]
    std: list[float]


class ServeMetrics:
    """The metrics of one app serving `model` (the `ModelInfo` fields
    `name`, `version`, `model_id`, `model_type`, `checkpoint_sha256`)."""

    def __init__(self, model: dict[str, str | None], reference: Reference | None):
        self.registry = CollectorRegistry()
        self.reference = reference
        options = {"registry": self.registry}
        self.requests = Counter(
            "mhd_serve_requests", "HTTP requests", ["route", "method", "status"], **options
        )
        self.latency = Histogram(
            "mhd_serve_request_duration_seconds",
            "Time to a response's start (a forecast's first byte)",
            ["route"],
            buckets=LATENCY_BUCKETS,
            **options,
        )
        self.inflight = Gauge("mhd_forecasts_inflight", "Forecasts being streamed", **options)
        self.duration = Histogram(
            "mhd_forecast_duration_seconds",
            "A forecast stream's duration, all of its members",
            buckets=DURATION_BUCKETS,
            **options,
        )
        self.frames = Counter("mhd_forecast_frames", "Forecast frames streamed", **options)
        self.failed = Counter(
            "mhd_forecasts_failed", "Forecasts that failed after their response started", **options
        )
        self.energy = Histogram(
            "mhd_forecast_energy_ratio",
            "A forecast block's mean kinetic energy over its context's",
            buckets=ENERGY_BUCKETS,
            **options,
        )
        self.shift = Histogram(
            "mhd_context_mean_shift",
            "A context's mean minus the training data's, in training stds, per channel",
            ["channel"],
            buckets=SHIFT_BUCKETS,
            **options,
        )
        self.scale = Histogram(
            "mhd_context_std_ratio",
            "A context's standard deviation over the training data's, per channel",
            ["channel"],
            buckets=SCALE_BUCKETS,
            **options,
        )
        # A bare logged model has no version: an empty label, not None.
        info = {key: "" if value is None else str(value) for key, value in model.items()}
        Info("mhd_model", "The model served", **options).info(info)

    def observe_context(self, frames: np.ndarray) -> float | None:
        """Record the drift of the context `frames` (the ones the model
        reads) and return their mean kinetic energy, the reference for the
        forecast's blocks; None for no frames."""
        if not len(frames):
            return None
        if self.reference is not None:
            mean = np.asarray(self.reference.mean)
            std = np.asarray(self.reference.std)
            channel_mean = frames.mean(axis=(0, 2, 3))
            channel_std = frames.std(axis=(0, 2, 3))
            for c, channel in enumerate(self.reference.channel_names):
                self.shift.labels(channel).observe((channel_mean[c] - mean[c]) / std[c])
                self.scale.labels(channel).observe(channel_std[c] / std[c])
        return float(kinetic_energy(frames).mean())

    def observe_block(self, block: np.ndarray, context_energy: float | None) -> None:
        self.frames.inc(len(block))
        if context_energy and len(block):
            self.energy.observe(float(kinetic_energy(block).mean()) / context_energy)

    def exposition(self) -> tuple[bytes, str]:
        """The metrics in Prometheus' text format, and its media type."""
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


class RequestMetrics:
    """ASGI middleware counting requests by route (the matched route's
    path, so the label set stays bounded), method and status, and timing
    them to the response's start. Plain ASGI rather than Starlette's
    BaseHTTPMiddleware, which would pass every streamed forecast chunk
    through a queue of its own. (`route`, not `endpoint`: the Prometheus
    operator labels every scraped series with the `endpoint` it came from,
    which would push this one aside as `exported_endpoint`.)"""

    def __init__(self, app: Callable[..., Awaitable[None]], metrics: ServeMetrics):
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        started = False

        def record(status: int) -> None:
            matched = scope.get("route")
            route = matched.path if matched is not None else "unmatched"
            self.metrics.requests.labels(route, scope["method"], str(status)).inc()
            self.metrics.latency.labels(route).observe(time.perf_counter() - start)

        async def recorded(message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                record(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, recorded)
        except Exception:
            if not started:
                record(500)
            raise
