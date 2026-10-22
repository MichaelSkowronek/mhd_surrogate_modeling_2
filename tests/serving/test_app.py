import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mhd_surrogate.evaluation.rollout import rollout_blocks
from mhd_surrogate.models.baselines import Persistence
from mhd_surrogate.serving import app as app_module
from mhd_surrogate.serving.app import app_from_env, create_app
from mhd_surrogate.serving.fetch import FetchedModel, fetch
from mhd_surrogate.serving.schemas import ServeSettings
from mhd_surrogate.serving.stream import MEDIA_TYPE, encode, read_arrays

FRAME = (2, 5, 4)
FETCHED = FetchedModel(
    model_uri="models:/surrogate/3",
    name="surrogate",
    version="3",
    model_id="m-1",
    model_type="persistence",
    checkpoint_sha256="abc",
    source_commit="c0ffee",
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


class FailsAfterOneBlock:
    """Streams one block of its forecast, then fails."""

    window = 1

    def predict(self, context, n_steps):  # pragma: no cover - streamed only
        raise NotImplementedError

    def predict_blocks(self, context, n_steps, block_steps):
        yield np.repeat(context[-1:], block_steps, axis=0)
        raise RuntimeError("diverged")


def serve(model=None, fetched=FETCHED, **settings) -> TestClient:
    app = create_app(model or Persistence(), fetched, ServeSettings(model_dir=".", **settings))
    return TestClient(app)


def context(n_frames: int = 3) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(n_frames, *FRAME)).astype(np.float32)


def post(client: TestClient, body: bytes, content_type: str = MEDIA_TYPE, **params):
    return client.post(
        "/v1/forecast", params=params, content=body, headers={"Content-Type": content_type}
    )


def test_healthz_says_ok():
    assert serve().get("/healthz").json() == {"status": "ok"}


def test_model_info_describes_the_model_and_the_limits(monkeypatch):
    monkeypatch.setenv("MHD_GIT_COMMIT", "1234")

    info = serve(max_members=4).get("/v1/model").json()

    assert info == {
        "name": "surrogate",
        "version": "3",
        "model_id": "m-1",
        "model_type": "persistence",
        "checkpoint_sha256": "abc",
        "window": 1,
        "frame_shape": list(FRAME),
        "stochastic": False,
        "source_commit": "c0ffee",
        "image_commit": "1234",
        "limits": {
            "max_steps": 50,
            "max_members": 4,
            "max_context_frames": 80,
            "default_block_steps": 100,
            "max_block_steps": 500,
        },
    }


def test_max_steps_is_the_validated_horizon_unless_set_and_required():
    assert serve(max_steps=7).get("/v1/model").json()["limits"]["max_steps"] == 7
    ungated = FETCHED.model_copy(update={"validated_steps": None})
    assert serve(fetched=ungated, max_steps=7).get("/v1/model").json()["limits"]["max_steps"] == 7
    with pytest.raises(ValueError, match="set MHD_SERVE_MAX_STEPS"):
        serve(fetched=ungated)


def test_openapi_documents_the_forecast_endpoint():
    schema = serve().get("/openapi.json").json()

    assert MEDIA_TYPE in schema["paths"]["/v1/forecast"]["post"]["requestBody"]["content"]


def test_a_deterministic_model_streams_one_member_in_blocks():
    frames = context()

    response = post(serve(), encode(frames), n_steps=7, block_steps=3, seeds=[0, 1, 2])

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-npy-stream"
    assert {k: response.headers[k] for k in ["x-members", "x-steps", "x-block-steps"]} == {
        "x-members": "1",
        "x-steps": "7",
        "x-block-steps": "3",
    }
    assert (response.headers["x-model-name"], response.headers["x-model-version"]) == (
        "surrogate",
        "3",
    )
    blocks = list(read_arrays([response.content]))
    assert [len(b) for b in blocks] == [3, 3, 1]
    assert all((b == frames[-1]).all() and b.dtype == np.float32 for b in blocks)


def test_a_stochastic_model_streams_each_seeds_member_as_rollout_blocks_makes_it():
    model, frames = Shift(), context()

    response = post(serve(model), encode(frames), n_steps=5, block_steps=2, seeds=[4, 9])

    assert response.headers["x-members"] == "2"
    expected = [b for seed in (4, 9) for b in rollout_blocks(model, frames, 5, 2, seed=seed)]
    got = list(read_arrays([response.content]))
    assert len(got) == len(expected) == 6
    for block, want in zip(got, expected, strict=True):
        np.testing.assert_array_equal(block, want)


def test_the_block_size_defaults_to_the_servers():
    response = post(serve(default_block_steps=4), encode(context()), n_steps=10)

    assert response.headers["x-block-steps"] == "4"
    assert [len(b) for b in read_arrays([response.content])] == [4, 4, 2]


@pytest.mark.parametrize(
    ("body", "content_type", "params", "status", "detail"),
    [
        (encode(context()), MEDIA_TYPE, {"n_steps": 51}, 422, "n_steps above 50"),
        (encode(context()), MEDIA_TYPE, {"n_steps": 1, "seeds": list(range(17))}, 422, "seeds"),
        (encode(context()), MEDIA_TYPE, {"n_steps": 1, "block_steps": 501}, 422, "block_steps"),
        (encode(context()), MEDIA_TYPE, {"n_steps": 0}, 422, "greater than or equal to 1"),
        (encode(context()), MEDIA_TYPE, {"n_steps": 1, "seeds": [1, 1]}, 422, "distinct"),
        (encode(context()), MEDIA_TYPE, {"n_steps": 1, "seeds": [-1]}, 422, "between 0"),
        (encode(context()), "application/json", {"n_steps": 1}, 415, "application/x-npy"),
        (b"garbage", MEDIA_TYPE, {"n_steps": 1}, 400, "isn't an .npy array"),
        (encode(context().astype(np.float64)), MEDIA_TYPE, {"n_steps": 1}, 422, "float32"),
        (encode(np.zeros((3, 2, 5, 5), np.float32)), MEDIA_TYPE, {"n_steps": 1}, 422, "shape"),
        (encode(np.zeros((2, 5, 4), np.float32)), MEDIA_TYPE, {"n_steps": 1}, 422, "shape"),
        (encode(context(0)), MEDIA_TYPE, {"n_steps": 1}, 422, "1 to 80 frames, not 0"),
        (encode(context(81)), MEDIA_TYPE, {"n_steps": 1}, 422, "1 to 80 frames, not 81"),
        (
            encode(np.full((1, *FRAME), np.nan, np.float32)),
            MEDIA_TYPE,
            {"n_steps": 1},
            422,
            "finite",
        ),
    ],
)
def test_bad_requests_are_refused_with_the_reason(body, content_type, params, status, detail):
    client = serve()

    response = post(client, body, content_type, **params)

    assert response.status_code == status
    assert detail in str(response.json()["detail"])
    assert client.app.state.slots.free == 1


def test_a_context_beyond_the_size_limit_is_refused_before_it_is_read():
    client = serve(max_context_frames=2)
    oversized = encode(np.zeros((40, *FRAME), np.float32))

    response = post(client, oversized, n_steps=1)

    assert response.status_code == 413
    assert client.app.state.slots.free == 1


def test_a_chunked_context_beyond_the_size_limit_is_refused_while_read():
    client = serve(max_context_frames=2)
    oversized = encode(np.zeros((40, *FRAME), np.float32))

    response = client.post(
        "/v1/forecast",
        params={"n_steps": 1},
        content=iter([oversized[:3000], oversized[3000:]]),
        headers={"Content-Type": MEDIA_TYPE},
    )

    assert response.status_code == 413


def test_a_busy_server_says_when_to_retry():
    client = serve()
    client.app.state.slots.free = 0

    response = post(client, encode(context()), n_steps=1)

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    client.app.state.slots.free = 1
    assert post(client, encode(context()), n_steps=1).status_code == 200


def test_the_slot_is_freed_after_a_forecast_and_after_one_that_fails(caplog):
    assert [post(serve(), encode(context()), n_steps=2).status_code for _ in range(2)] == [200, 200]
    client = serve(FailsAfterOneBlock())

    with caplog.at_level(logging.ERROR, logger="mhd_surrogate.serving.app"):
        response = post(client, encode(context()), n_steps=6, block_steps=2)

    assert response.status_code == 200
    assert [len(b) for b in read_arrays([response.content])] == [2]
    assert "failed after its response started" in caplog.text
    assert client.app.state.slots.free == 1


def test_app_from_env_serves_the_fetched_model(stores, promoted, tmp_path, monkeypatch):
    model, _ = promoted(scores={"gate.rollout_stable_steps": 300.0})
    fetch("models:/surrogate@champion", stores[1], tmp_path / "served")
    caches = []
    monkeypatch.setattr(app_module, "enable_compilation_cache", caches.append)
    monkeypatch.setenv("MHD_SERVE_MODEL_DIR", str(tmp_path / "served"))
    monkeypatch.setenv("MHD_SERVE_JAX_CACHE_DIR", str(tmp_path / "cache"))

    client = TestClient(app_from_env())

    assert caches == [tmp_path / "cache"]
    info = client.get("/v1/model").json()
    assert (info["name"], info["model_type"], info["window"]) == ("surrogate", "mean_field", 0)
    assert info["limits"]["max_steps"] == 300
    response = post(client, encode(np.zeros((0, *FRAME), np.float32)), n_steps=3)
    (forecast,) = read_arrays([response.content])
    assert (forecast == model.mean).all() and forecast.shape == (3, *FRAME)
