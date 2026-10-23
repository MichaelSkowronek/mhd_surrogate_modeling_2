import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mhd_surrogate.serving.metrics import Reference, RequestMetrics, ServeMetrics

MODEL = {
    "name": "m",
    "version": "1",
    "model_id": "m-1",
    "model_type": "t",
    "checkpoint_sha256": "s",
}
REFERENCE = Reference(channel_names=["u_x", "u_y"], mean=[1.0, 0.0], std=[2.0, 0.5])


def value(metrics: ServeMetrics, name: str, **labels) -> float:
    return metrics.registry.get_sample_value(name, labels or None)


def frames(u_x: float, u_y: float, n: int = 2) -> np.ndarray:
    out = np.empty((n, 2, 3, 2), dtype=np.float32)
    out[:, 0], out[:, 1] = u_x, u_y
    return out


def test_a_context_is_measured_against_the_reference():
    metrics = ServeMetrics(MODEL, REFERENCE)
    context = frames(2.0, 0.25)
    context[1, 1] = 0.75  # u_y: mean 0.5, std 0.25

    energy = metrics.observe_context(context)

    assert energy == pytest.approx(np.mean([0.5 * (4 + 0.0625), 0.5 * (4 + 0.5625)]))
    # A histogram that can see negative values has no _sum: read its buckets.
    shift = "mhd_context_mean_shift_bucket"
    assert value(metrics, shift, channel="u_x", le="0.25") == 0  # shift 0.5
    assert value(metrics, shift, channel="u_x", le="0.5") == 1
    assert value(metrics, shift, channel="u_y", le="0.5") == 0  # shift 1.0
    assert value(metrics, shift, channel="u_y", le="1.0") == 1
    scale = {c: value(metrics, "mhd_context_std_ratio_sum", channel=c) for c in ["u_x", "u_y"]}
    assert scale == pytest.approx({"u_x": 0.0, "u_y": 0.5})


def test_without_a_reference_only_the_energy_is_measured():
    metrics = ServeMetrics(MODEL, None)

    assert metrics.observe_context(frames(1.0, 1.0)) == pytest.approx(1.0)
    assert value(metrics, "mhd_context_mean_shift_count", channel="u_x") is None
    assert metrics.observe_context(frames(1.0, 1.0, n=0)) is None


def test_a_block_is_counted_and_its_energy_compared_with_the_contexts():
    metrics = ServeMetrics(MODEL, None)

    metrics.observe_block(frames(2.0, 0.0, n=3), context_energy=1.0)
    metrics.observe_block(frames(2.0, 0.0, n=2), context_energy=None)

    assert value(metrics, "mhd_forecast_frames_total") == 5
    assert value(metrics, "mhd_forecast_energy_ratio_count") == 1
    assert value(metrics, "mhd_forecast_energy_ratio_sum") == pytest.approx(2.0)


def test_the_exposition_names_the_model():
    content, media_type = ServeMetrics(MODEL, None).exposition()

    assert media_type.startswith("text/plain")
    assert b'mhd_model_info{checkpoint_sha256="s",model_id="m-1"' in content


def test_a_model_without_a_version_is_served_with_an_empty_version_label():
    content, _ = ServeMetrics({**MODEL, "version": None}, None).exposition()

    assert b'version=""' in content


def test_requests_are_counted_by_route_and_status_including_errors():
    metrics = ServeMetrics(MODEL, None)
    app = FastAPI()
    app.add_middleware(RequestMetrics, metrics=metrics)

    @app.get("/items/{item}")
    def item(item: int) -> dict[str, int]:
        return {"item": item}

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("boom")

    # As a context manager, the client runs the app's lifespan, which the
    # middleware passes through untouched.
    with TestClient(app, raise_server_exceptions=False) as client:
        statuses = [client.get(path).status_code for path in ["/items/1", "/items/2", "/nope"]]
        assert statuses == [200, 200, 404]
        assert client.get("/boom").status_code == 500

    requests = "mhd_serve_requests_total"
    assert value(metrics, requests, route="/items/{item}", method="GET", status="200") == 2
    assert value(metrics, requests, route="unmatched", method="GET", status="404") == 1
    assert value(metrics, requests, route="/boom", method="GET", status="500") == 1
    latency = "mhd_serve_request_duration_seconds_count"
    assert value(metrics, latency, route="/items/{item}") == 2
