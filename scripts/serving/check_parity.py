"""Check that a running forecast service serves the model's own forecast.

Sends the validation dataset's context (its first `context_steps` frames,
configs/data/re16k.yaml) to the service, streams back the forecast, and
compares it with the forecast the same fetched model makes offline,
`evaluation/protocol.py`'s `forecast` -- the one its scores come from.
Both must run on the same JAX backend to agree beyond float rounding: the
service in the image runs on the CPU, so run this with `JAX_PLATFORMS=cpu`.
Exits 1 if they differ by more than `--tolerance` (relative to the truth's
standard deviation). The DVC `serve_parity` stage makes the same check in
process, on every change to the serving code (scripts/serving/serve_parity.py);
this one is for a deployed service, over HTTP. The test dataset is never
read.

Usage:
    JAX_PLATFORMS=cpu uv run scripts/serving/check_parity.py --model-dir models/served
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
import zarr

from mhd_surrogate.evaluation.protocol import forecast
from mhd_surrogate.models.registry import load_model
from mhd_surrogate.serving.client import ForecastClient
from mhd_surrogate.serving.fetch import MODEL_DIR, checkpoint_path, load_fetched
from mhd_surrogate.serving.parity import relative_difference
from mhd_surrogate.utils.logging_config import add_log_level_arg, setup_logging

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model-dir", type=Path, default=Path("models/served"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/re16k.yaml"))
    parser.add_argument("--block-steps", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    add_log_level_arg(parser)
    return parser.parse_args()


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)) / np.std(truth))


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    data = yaml.safe_load(args.data_config.read_text())
    series = zarr.open_group(store=data["zarr_store"], mode="r")[data["val_dataset"]]
    context_steps = data["context_steps"]

    fetched = load_fetched(args.model_dir)
    client = ForecastClient(args.url)
    info = client.model_info()
    if info.model_id != fetched.model_id:
        sys.exit(f"the service serves {info.model_id}, {args.model_dir} holds {fetched.model_id}")

    model = load_model(checkpoint_path(args.model_dir / MODEL_DIR))
    offline, truth = forecast(model, series, context_steps)
    log.info("requesting %d steps from %s", len(truth), args.url)
    context = np.asarray(series[:context_steps], dtype=np.float32)
    (served,) = client.forecast(context, len(truth), block_steps=args.block_steps)

    difference = relative_difference(served, offline, truth)
    print(f"{info.name} version {info.version} ({info.model_id}), {len(truth)} steps")
    print(f"max |served - offline| / std(truth): {difference:.3g}")
    print(
        f"RMSE / std(truth): served {rmse(served, truth):.6g}, offline {rmse(offline, truth):.6g}"
    )
    if difference > args.tolerance:
        sys.exit(f"the served forecast differs from the offline one by more than {args.tolerance}")
    print("parity: ok")


if __name__ == "__main__":
    main()
