"""Fetch a registered model for the forecast server.

Resolves `--model-uri` on the tracking server, downloads the logged model
into `--dest`, checks its checkpoint's content hash and records what it is
(`serving.json`), for the server to load (src/mhd_surrogate/serving/fetch.py).
Run before the server: an init container in Kubernetes, the `serve-fetch`
service in docker-compose.yml.

Usage:
    uv run scripts/serving/fetch_model.py --model-uri models:/mhd-surrogate@champion \\
        --dest models/served
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml

from mhd_surrogate.serving.fetch import fetch
from mhd_surrogate.utils.logging_config import add_log_level_arg, setup_logging

DEFAULT_TRACKING = Path("configs/mlflow/server.yaml")

# As in promote.py: large artifacts through the tracking server, not by
# presigned URL to its object store, whose address resolves only inside the
# compose network (README, "Tracking stack").
os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "false")
os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model-uri",
        default=os.environ.get("MHD_MODEL_URI"),
        help="models:/<name>/<version>, models:/<name>@<alias> or models:/<model id> "
        "(default: $MHD_MODEL_URI)",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help=f"default: $MLFLOW_TRACKING_URI, else {DEFAULT_TRACKING}'s",
    )
    parser.add_argument("--dest", type=Path, required=True)
    add_log_level_arg(parser)
    args = parser.parse_args()
    if not args.model_uri:
        parser.error("give --model-uri or set MHD_MODEL_URI")
    return args


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    # MLflow downloads a model's files on more threads than urllib3 keeps
    # connections for, which it warns about for every file; harmless.
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
    tracking_uri = args.tracking_uri or yaml.safe_load(DEFAULT_TRACKING.read_text())["tracking_uri"]
    fetched = fetch(args.model_uri, tracking_uri, args.dest)
    print(fetched.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
