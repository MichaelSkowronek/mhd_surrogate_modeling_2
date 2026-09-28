# syntax=docker/dockerfile:1
#
# Multi-stage image for the MHD surrogate project.
#
#   builder  resolves and installs the locked dependencies with uv, then the
#            project itself (non-editable) into /app/.venv
#   runtime  (default target) python-slim + the finished venv + scripts/ and
#            configs/; no uv, no build tooling, no dev dependencies, non-root
#   test     builder + dev group + tests/; `docker build --target test .`
#            runs pytest at build time
#
# Data is never baked into the image (multi-GB, and it changes independently
# of the code): mount data/ at run time, see docker-compose.yml.

ARG PYTHON_VERSION=3.14

FROM python:${PYTHON_VERSION}-slim AS builder

# Pinned to the uv version used locally / in CI.
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system

WORKDIR /app

# Layer 1: third-party dependencies only. This layer is rebuilt only when
# pyproject.toml or uv.lock change, not on every source edit.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --extra server --no-install-project

# Layer 2: the project itself, installed non-editable so the venv is
# self-contained and can be copied to the runtime stage as-is.
# (README.md is needed because pyproject.toml declares it as the readme.)
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra server --no-editable


FROM builder AS test

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra server --no-editable
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests
ENV PATH="/app/.venv/bin:$PATH"
CMD ["pytest"]


FROM python:${PYTHON_VERSION}-slim AS runtime

# Non-root user. UID/GID 1000 matches the usual first Linux user, so files
# written into bind-mounted directories (reports/, outputs/) stay owned by
# the host user. Override with --build-arg if yours differs.
ARG UID=1000
ARG GID=1000
RUN groupadd --gid ${GID} app && useradd --uid ${UID} --gid ${GID} --create-home app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    GIT_PYTHON_REFRESH=quiet
# GIT_PYTHON_REFRESH: MLflow imports GitPython to record the source commit;
# the slim image has no git binary, so silence its startup warning.

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv ./.venv
# scripts/ and configs/ stay side by side: Hydra resolves config_path
# relative to the script, and the analysis scripts use cwd-relative paths.
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app configs ./configs

# Mount points the app writes to, pre-created so a fresh named volume
# inherits the right ownership.
RUN mkdir -p data reports/figures reports/videos reports/summaries outputs \
    && chown -R app:app data reports outputs

USER ${UID}:${GID}
CMD ["python", "scripts/training/train.py"]
