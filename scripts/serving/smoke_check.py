"""Smoke-check a deployed forecast service.

Waits for `/healthz`, reads `/v1/model` (and checks it against `--expect-*`
if given: `--expect-model-uri` is what the deployment fetched,
`$MHD_MODEL_URI`, whose name and version, if it names one, must be served),
then forecasts `--steps` steps (at most the service's limit) of
`--members` seeds from a random context of the served model's shape, and
checks every member's frames arrived, shaped as the model's frames and
finite. Exits 1 on the first problem. Then, with `--set-alias`, it points
that alias of the served registered model at the version it checked: the
deployment runs it after every sync (an Argo CD PostSync hook,
deploy/k8s/smoke-check/) and moves `champion` that way, so the alias
always names the version verified in service. CI's deploy-smoke job runs
it against its throwaway cluster.

Usage:
    uv run scripts/serving/smoke_check.py --url http://localhost:8080 --expect-version 1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import httpx
import numpy as np

import mlflow
from mhd_surrogate.serving.client import ForecastClient
from mhd_surrogate.utils.logging_config import add_log_level_arg, setup_logging

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--members", type=int, default=2)
    parser.add_argument("--wait", type=float, default=120.0, help="seconds to wait for /healthz")
    parser.add_argument("--expect-model-uri", help="models:/<name>/<version> or @<alias>")
    parser.add_argument("--expect-name")
    parser.add_argument("--expect-version")
    parser.add_argument("--expect-sha", help="the checkpoint's checkpoint_sha256")
    parser.add_argument("--set-alias", help="the alias to point at the version, if it passes")
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="the registry --set-alias writes to (default: $MLFLOW_TRACKING_URI)",
    )
    add_log_level_arg(parser)
    return parser.parse_args()


def expected_from_uri(model_uri: str) -> dict[str, str]:
    """What a model URI says the service must serve: the name and version
    of `models:/<name>/<version>`, the name of `models:/<name>@<alias>`
    (the alias may have moved since), the logged model of
    `models:/<model id>`."""
    path = model_uri.removeprefix("models:/")
    if "@" in path:
        return {"name": path.split("@", 1)[0]}
    if "/" in path:
        name, version = path.split("/", 1)
        return {"name": name, "version": version}
    return {"model_id": path}


def wait_until_healthy(url: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while True:
        try:
            if httpx.get(f"{url}/healthz", timeout=5).status_code == 200:
                return
        except httpx.TransportError:
            pass
        if time.monotonic() > deadline:
            sys.exit(f"{url} isn't healthy after {seconds:g} s")
        time.sleep(2)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    wait_until_healthy(args.url, args.wait)
    client = ForecastClient(args.url)
    info = client.model_info()
    print(f"serving {info.name} version {info.version} ({info.model_id}, {info.model_type})")
    expectations = {
        "name": args.expect_name,
        "version": args.expect_version,
        "checkpoint_sha256": args.expect_sha,
    }
    if args.expect_model_uri:
        for field, value in expected_from_uri(args.expect_model_uri).items():
            expectations[field] = expectations.get(field) or value
    for field, expected in expectations.items():
        if expected is not None and getattr(info, field) != expected:
            sys.exit(f"{field} is {getattr(info, field)!r}, expected {expected!r}")

    steps = min(args.steps, info.limits.max_steps)
    context = np.random.default_rng(0).normal(size=(info.window, *info.frame_shape))
    seeds = list(range(min(args.members, info.limits.max_members)))
    # A probe: its random context stays out of the service's drift metrics.
    forecast = client.forecast(context.astype(np.float32), steps, seeds=seeds, probe=True)
    members = len(seeds) if info.stochastic else 1
    if forecast.shape != (members, steps, *info.frame_shape):
        sys.exit(f"the forecast has shape {forecast.shape}")
    if not np.isfinite(forecast).all():
        sys.exit("the forecast isn't finite")
    print(f"forecast {members} member(s) x {steps} steps of {info.frame_shape}: ok")

    if args.set_alias:
        if info.version is None or args.tracking_uri is None:
            sys.exit("--set-alias needs a registered version and --tracking-uri")
        registry = mlflow.MlflowClient(
            tracking_uri=args.tracking_uri, registry_uri=args.tracking_uri
        )
        registry.set_registered_model_alias(info.name, args.set_alias, info.version)
        print(f"{info.name} @{args.set_alias} -> version {info.version}")


if __name__ == "__main__":
    main()
