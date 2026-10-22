import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from mhd_surrogate.evaluation.rollout import rollout_blocks
from mhd_surrogate.serving.app import create_app
from mhd_surrogate.serving.client import ForecastClient, ForecastError
from mhd_surrogate.serving.fetch import FetchedModel
from mhd_surrogate.serving.schemas import ServeSettings
from mhd_surrogate.serving.stream import STREAM_MEDIA_TYPE, encode

FRAME = (2, 5, 4)
FETCHED = FetchedModel(
    model_uri="models:/surrogate/1",
    name="surrogate",
    version="1",
    model_id="m-1",
    model_type="shift",
    checkpoint_sha256="abc",
    source_commit=None,
    validated_steps=50,
    frame_shape=FRAME,
)


class Shift:
    """A stochastic model: the last frame plus `seed + step` at each step."""

    window = 1
    stochastic = True

    def predict(self, context, n_steps, seed=0):
        steps = np.arange(1, n_steps + 1, dtype=np.float32)[:, None, None, None]
        return context[-1] + seed + steps


class StopsAfterOneBlock(Shift):
    stochastic = False

    def predict_blocks(self, context, n_steps, block_steps):
        yield np.repeat(context[-1:], block_steps, axis=0)
        raise RuntimeError("diverged")


def client_for(model) -> ForecastClient:
    app = create_app(model, FETCHED, ServeSettings(model_dir="."))
    return ForecastClient(http=TestClient(app))


def context() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(2, *FRAME)).astype(np.float32)


def mocked(handler) -> ForecastClient:
    return ForecastClient(
        http=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x")
    )


def test_the_client_reads_the_model_info():
    info = client_for(Shift()).model_info()

    assert (info.name, info.version, info.window, info.stochastic) == ("surrogate", "1", 1, True)
    assert info.limits.max_steps == 50


def test_forecast_assembles_each_members_blocks():
    model, frames = Shift(), context()

    forecast = client_for(model).forecast(frames, 5, seeds=[3, 7], block_steps=2)

    assert forecast.shape == (2, 5, *FRAME)
    for member, seed in enumerate((3, 7)):
        expected = np.concatenate(list(rollout_blocks(model, frames, 5, 2, seed=seed)))
        np.testing.assert_array_equal(forecast[member], expected)


def test_forecast_blocks_yields_each_block_with_its_member():
    blocks = list(client_for(Shift()).forecast_blocks(context(), 3, seeds=[0, 1], block_steps=2))

    assert [(member, len(block)) for member, block in blocks] == [(0, 2), (0, 1), (1, 2), (1, 1)]


def test_an_error_response_raises_with_the_servers_detail():
    with pytest.raises(ForecastError, match="n_steps above 50") as error:
        client_for(Shift()).forecast(context(), 51)
    assert error.value.status == 422


def test_a_forecast_that_stops_early_raises():
    with pytest.raises(ForecastError, match="ended after 2 of 6 frames"):
        client_for(StopsAfterOneBlock()).forecast(context(), 6, block_steps=2)


def test_a_stream_broken_inside_a_record_raises():
    record = encode(np.zeros((3, *FRAME), np.float32))

    def handler(request):
        headers = {"X-Members": "1", "Content-Type": STREAM_MEDIA_TYPE}
        return httpx.Response(200, content=record[:-8], headers=headers)

    with pytest.raises(ForecastError, match="broke off"):
        mocked(handler).forecast(context(), 3)


def test_an_error_without_a_json_detail_raises_with_the_body():
    client = mocked(lambda request: httpx.Response(502, text="bad gateway"))

    with pytest.raises(ForecastError, match="502: bad gateway"):
        client.model_info()


def test_the_client_takes_a_base_url_or_an_http_client():
    client = ForecastClient("http://localhost:1")
    assert str(client.http.base_url) == "http://localhost:1"
    client.close()
    with pytest.raises(ValueError, match="either base_url or http"):
        ForecastClient()
    with pytest.raises(ValueError, match="either base_url or http"):
        ForecastClient("http://localhost:1", http=httpx.Client())
