"""The forecast API's request parameters, responses and settings.

Shared by the server (`serving/app.py`) and its client (`serving/client.py`),
so both validate against the same definitions.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Seeds are 32-bit, like the seeds `rollout.block_seed` draws.
MAX_SEED = 2**32 - 1
# Marks a synthetic request (a smoke check), kept out of the ML metrics.
PROBE_HEADER = "X-Probe"


class ForecastParams(BaseModel):
    """`POST /v1/forecast` query parameters: `n_steps` frames for each of
    `seeds` (a stochastic model's samples; a deterministic model forecasts
    one member whatever the seeds), streamed `block_steps` frames per
    record (the server's default if None)."""

    n_steps: int = Field(ge=1)
    seeds: list[int] = Field(default=[0], min_length=1)
    block_steps: int | None = Field(default=None, ge=1)

    @field_validator("seeds")
    @classmethod
    def _distinct_32_bit(cls, seeds: list[int]) -> list[int]:
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be distinct")
        if not all(0 <= seed <= MAX_SEED for seed in seeds):
            raise ValueError(f"seeds must be between 0 and {MAX_SEED}")
        return seeds


class Limits(BaseModel):
    """What one request may ask for: `max_steps` forecast steps (by default
    the steps the model's gate validated a rollout stable for),
    `max_members` seeds, `max_context_frames` context frames (extra leading
    frames are ignored, as under the forecast protocol), `max_block_steps`
    frames per streamed record (`default_block_steps` if not asked)."""

    max_steps: int
    max_members: int
    max_context_frames: int
    default_block_steps: int
    max_block_steps: int


class ModelInfo(BaseModel):
    """`GET /v1/model`: the model served (see `serving/fetch.py`'s
    `FetchedModel`), what it needs as context (`window` frames of
    `frame_shape`, float32, raw units), whether seeds give different
    members (`stochastic`), the commits of its training run and of the
    server's image, and the request limits."""

    name: str
    version: str | None
    model_id: str
    model_type: str
    checkpoint_sha256: str
    window: int
    frame_shape: tuple[int, int, int]
    stochastic: bool
    source_commit: str | None
    image_commit: str | None
    limits: Limits


class ServeSettings(BaseSettings):
    """The server's settings, from `MHD_SERVE_*` environment variables.

    `model_dir` is where `serving/fetch.py` fetched the model to.
    `max_steps` overrides the gate's validated horizon (required for a model
    without one). `max_inflight` forecasts run at a time; more get 503: a
    forecast keeps every CPU core busy, so a pod serves one at a time and
    scales by replicas. `jax_cache_dir` enables JAX's persistent compilation
    cache there (`utils/jax_cache.py`).
    """

    model_config = SettingsConfigDict(env_prefix="MHD_SERVE_")

    model_dir: Path
    max_steps: int | None = Field(default=None, ge=1)
    max_members: int = Field(default=16, ge=1)
    max_context_frames: int = Field(default=80, ge=0)
    default_block_steps: int = Field(default=100, ge=1)
    max_block_steps: int = Field(default=500, ge=1)
    max_inflight: int = Field(default=1, ge=1)
    jax_cache_dir: Path | None = None
    log_level: str = "INFO"
