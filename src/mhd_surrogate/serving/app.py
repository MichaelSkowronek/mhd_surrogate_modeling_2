"""The forecast service: a FastAPI app over a fetched model.

- `POST /v1/forecast`: the context as an `.npy` body (`application/x-npy`,
  the last `window` frames are used), the parameters in the query
  (`schemas.ForecastParams`); the response streams the forecast as `.npy`
  records (`serving/stream.py`), member after member, each `block_steps`
  frames at a time (`evaluation/rollout.py`'s `rollout_blocks`, so a
  model's streamed forecast is its scored one), so the server holds one
  block, not the forecast. Headers give the members, steps and block size.
- `GET /v1/model`: what is served (`schemas.ModelInfo`).
- `GET /healthz`: liveness.

The model is the logged model `serving/fetch.py` fetched into
`MHD_SERVE_MODEL_DIR`, loaded by the image's own code
(`models/registry.py`), not by `mlflow.pyfunc.load_model`: a pyfunc's copy
of the package (`code_paths`) only goes on `sys.path`, behind the
`mhd_surrogate` the process has imported already, so the image's code would
load the checkpoint either way. The image must therefore read the served
checkpoint's format; `/v1/model` shows both commits.

The model is loaded before the server listens (`app_from_env`, which
`uvicorn --factory` calls), so a server that answers `/healthz` is ready to
forecast. Forecasts run in a worker thread, so health checks are answered
during one; `max_inflight` run at a time, more get 503 with `Retry-After`.

Run:
    MHD_SERVE_MODEL_DIR=models/served uv run uvicorn --factory \\
        mhd_surrogate.serving.app:app_from_env
"""

from __future__ import annotations

import importlib.metadata
import logging
import math
import os
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Annotated

import anyio.to_thread
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from mhd_surrogate.evaluation.protocol import ForecastModel, ensemble_size, is_stochastic
from mhd_surrogate.evaluation.rollout import rollout_blocks
from mhd_surrogate.models.registry import load_model
from mhd_surrogate.serving.fetch import MODEL_DIR, FetchedModel, checkpoint_path, load_fetched
from mhd_surrogate.serving.schemas import ForecastParams, Limits, ModelInfo, ServeSettings
from mhd_surrogate.serving.stream import MEDIA_TYPE, STREAM_MEDIA_TYPE, decode, encode
from mhd_surrogate.training.tracking import IMAGE_COMMIT_ENV
from mhd_surrogate.utils.jax_cache import enable_compilation_cache
from mhd_surrogate.utils.logging_config import setup_logging

log = logging.getLogger(__name__)

RETRY_AFTER_S = 5
# Room for an .npy header (NumPy pads it to a multiple of 64 bytes; a 4-D
# float32 array's is 128).
NPY_HEADER_BYTES = 4096


def app_from_env() -> FastAPI:
    """The app for the model fetched into `MHD_SERVE_MODEL_DIR`, configured
    by the `MHD_SERVE_*` environment (`schemas.ServeSettings`)."""
    settings = ServeSettings()
    setup_logging(settings.log_level)
    if settings.jax_cache_dir is not None:
        enable_compilation_cache(settings.jax_cache_dir)
    fetched = load_fetched(settings.model_dir)
    model = load_model(checkpoint_path(settings.model_dir / MODEL_DIR))
    log.info("serving %s version %s (%s)", fetched.name, fetched.version, fetched.model_id)
    return create_app(model, fetched, settings)


def create_app(model: ForecastModel, fetched: FetchedModel, settings: ServeSettings) -> FastAPI:
    """The app serving `model`, which `fetched` describes."""
    max_steps = settings.max_steps or fetched.validated_steps
    if max_steps is None:
        raise ValueError(f"{fetched.model_uri} wasn't gated: set MHD_SERVE_MAX_STEPS")
    limits = Limits(
        max_steps=max_steps,
        max_members=settings.max_members,
        max_context_frames=settings.max_context_frames,
        default_block_steps=settings.default_block_steps,
        max_block_steps=settings.max_block_steps,
    )
    info = ModelInfo(
        **fetched.model_dump(include=set(ModelInfo.model_fields)),
        window=model.window,
        stochastic=is_stochastic(model),
        image_commit=os.environ.get(IMAGE_COMMIT_ENV) or None,
        limits=limits,
    )
    slots = _Slots(settings.max_inflight)
    max_body = NPY_HEADER_BYTES + limits.max_context_frames * 4 * math.prod(info.frame_shape)
    app = FastAPI(
        title="MHD surrogate forecasts", version=importlib.metadata.version("mhd-surrogate")
    )
    app.state.slots = slots

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/model")
    def model_info() -> ModelInfo:
        return info

    @app.post(
        "/v1/forecast",
        response_class=StreamingResponse,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {MEDIA_TYPE: {"schema": {"type": "string", "format": "binary"}}},
            }
        },
        responses={
            200: {"content": {STREAM_MEDIA_TYPE: {}}, "description": "the forecast's blocks"},
            400: {"description": "the body isn't an .npy array"},
            413: {"description": "more context frames than the limit"},
            415: {"description": f"the body isn't {MEDIA_TYPE}"},
            422: {"description": "invalid parameters or context"},
            503: {"description": "busy: retry after Retry-After seconds"},
        },
    )
    async def forecast(
        request: Request, params: Annotated[ForecastParams, Query()]
    ) -> StreamingResponse:
        block_steps = params.block_steps or limits.default_block_steps
        _check_limits(params, block_steps, limits)
        media_type = request.headers.get("content-type", "").split(";")[0].strip()
        if media_type != MEDIA_TYPE:
            raise HTTPException(415, f"send the context as {MEDIA_TYPE}, not {media_type!r}")
        if not slots.acquire():
            raise HTTPException(
                503, "busy forecasting", headers={"Retry-After": str(RETRY_AFTER_S)}
            )
        try:
            context = _context(await _body(request, max_body), info, limits)
            members = params.seeds[: ensemble_size(model, len(params.seeds))]
            blocks = _blocks(model, context, params.n_steps, members, block_steps)
        except BaseException:
            slots.release()
            raise
        headers = {
            "X-Model-Name": info.name,
            "X-Model-Version": info.version or "",
            "X-Members": str(len(members)),
            "X-Steps": str(params.n_steps),
            "X-Block-Steps": str(block_steps),
        }
        return StreamingResponse(
            _records(blocks, slots), media_type=STREAM_MEDIA_TYPE, headers=headers
        )

    return app


class _Slots:
    """At most `n` forecasts at a time (used from the event loop only)."""

    def __init__(self, n: int):
        self.free = n

    def acquire(self) -> bool:
        if not self.free:
            return False
        self.free -= 1
        return True

    def release(self) -> None:
        self.free += 1


def _check_limits(params: ForecastParams, block_steps: int, limits: Limits) -> None:
    problems = [
        message
        for broken, message in [
            (params.n_steps > limits.max_steps, f"n_steps above {limits.max_steps}"),
            (len(params.seeds) > limits.max_members, f"more than {limits.max_members} seeds"),
            (block_steps > limits.max_block_steps, f"block_steps above {limits.max_block_steps}"),
        ]
        if broken
    ]
    if problems:
        raise HTTPException(422, "; ".join(problems))


async def _body(request: Request, max_bytes: int) -> bytes:
    """The request body, refused (413) beyond `max_bytes`."""
    too_large = HTTPException(413, f"the context is limited to {max_bytes} bytes")
    if int(request.headers.get("content-length", 0)) > max_bytes:
        raise too_large
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > max_bytes:
            raise too_large
    return bytes(body)


def _context(data: bytes, info: ModelInfo, limits: Limits) -> np.ndarray:
    """The context array of the body `data`, checked against the model."""
    try:
        context = decode(data)
    except ValueError as error:
        raise HTTPException(400, f"the body isn't an .npy array: {error}") from None
    if context.dtype != np.float32:
        raise HTTPException(422, f"the context must be float32, not {context.dtype}")
    if context.ndim != 4 or context.shape[1:] != info.frame_shape:
        raise HTTPException(
            422, f"the context must be frames of shape {info.frame_shape}, not {context.shape}"
        )
    if not info.window <= len(context) <= limits.max_context_frames:
        raise HTTPException(
            422,
            f"the context must have {info.window} to {limits.max_context_frames} frames, "
            f"not {len(context)}",
        )
    if not np.isfinite(context).all():
        raise HTTPException(422, "the context must be finite")
    return context


def _blocks(
    model: ForecastModel,
    context: np.ndarray,
    n_steps: int,
    seeds: Sequence[int],
    block_steps: int,
) -> Iterator[np.ndarray]:
    for seed in seeds:
        yield from rollout_blocks(model, context, n_steps, block_steps, seed=seed)


async def _records(blocks: Iterator[np.ndarray], slots: _Slots) -> AsyncIterator[bytes]:
    """The `.npy` records of `blocks`, each computed in a worker thread; the
    slot is released when the stream ends, fails or the client goes away.

    A failure mid-stream ends the stream early (the client counts the frames,
    `serving/client.py`): the response has started, so the status can't
    change any more."""
    try:
        while (record := await anyio.to_thread.run_sync(_next_record, blocks)) is not None:
            yield record
    except Exception:
        log.exception("the forecast failed after its response started")
    finally:
        blocks.close()
        slots.release()


def _next_record(blocks: Iterator[np.ndarray]) -> bytes | None:
    block = next(blocks, None)
    return None if block is None else encode(np.asarray(block, dtype=np.float32))
