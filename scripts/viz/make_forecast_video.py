"""Render a model's forecast next to the truth: truth, prediction and their
difference stacked full-width, plus the RMSE-vs-lead-time curve with a cursor.

The forecast is made exactly as it is scored (`evaluation/protocol.py`): the
first `context_steps` steps of the dataset are context, and frame k of the
video is lead time k + 1. Color scales are fixed for the whole video so decay
or blow-up stays visible: truth and prediction share the truth's 1st/99th
percentile range, and the difference panel is centered on zero with a
half-range as large as that, so an error the size of the flow's own
variability saturates it. The RMSE curve is the scored one (per-channel
training std as scale), over the whole forecast even when the video is cut
short with --max-steps.

The model comes from MLflow (--model-id, a logged model from
`training/mlflow_model.py`) or a checkpoint directory (--checkpoint). With
--log-to-mlflow (needs --model-id) the video is attached to the run that
logged the model, under `videos/`. The test dataset is refused: it is read
once, by the final evaluation, not here.

Requires no system ffmpeg install: encoding uses the static binary bundled
by the imageio-ffmpeg package.

Usage:
    uv run scripts/viz/make_forecast_video.py --model-id m-46cff265...
    uv run scripts/viz/make_forecast_video.py --checkpoint outputs/<date>/<time>/model \\
        --field u_y --max-steps 200
    uv run scripts/viz/make_forecast_video.py --model-id m-... --log-to-mlflow
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
import zarr  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

from mhd_surrogate.analysis.fields import FIELDS, compute_field  # noqa: E402
from mhd_surrogate.data.grid import grid_spacing  # noqa: E402
from mhd_surrogate.data.normalization import NormalizationStats  # noqa: E402
from mhd_surrogate.evaluation.metrics import rmse_per_step  # noqa: E402
from mhd_surrogate.evaluation.protocol import forecast  # noqa: E402
from mhd_surrogate.models.registry import load_model  # noqa: E402
from mhd_surrogate.utils.logging_config import add_log_level_arg, setup_logging  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_DATA_CONFIG = Path("configs/data/re16k.yaml")
DEFAULT_STATS = Path("data/processed/normalization_stats.json")
DEFAULT_OUT_DIR = Path("reports/videos")
DEFAULT_TRACKING_URI = "sqlite:///mlruns.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-id", help="MLflow logged model id (m-...)")
    source.add_argument("--checkpoint", type=Path, help="Checkpoint directory (model.json)")
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help="MLflow tracking URI for --model-id (default: $MLFLOW_TRACKING_URI or %(default)s)",
    )
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--dataset", default=None, help="Default: the validation dataset")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--std-mode", choices=["per_channel", "shared"], default="per_channel")
    parser.add_argument("--field", choices=sorted(FIELDS), default="vorticity")
    parser.add_argument("--max-steps", type=int, default=None, help="Default: whole forecast")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth lead time")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--color-sample", type=int, default=30, help="Frames sampled for limits")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--log-to-mlflow", action="store_true")
    add_log_level_arg(parser)
    return parser.parse_args()


def forecast_limits(truth_sample: np.ndarray, symmetric: bool) -> tuple[float, float, float]:
    """(vmin, vmax, diff_limit) for a sample of truth field frames.

    vmin/vmax are the 1st/99th percentiles (symmetric about zero for a
    signed field such as vorticity); the difference panel spans
    [-diff_limit, diff_limit] with diff_limit half that range, so the
    difference is drawn on the same scale as the field's own excursions.
    """
    lo, hi = np.percentile(truth_sample, [1, 99])
    if symmetric:
        limit = max(abs(lo), abs(hi))
        lo, hi = -limit, limit
    return float(lo), float(hi), float((hi - lo) / 2)


def check_dataset(name: str, data_config: dict) -> None:
    """Allowed: the validation and training datasets; never the test one."""
    if name == data_config["test_dataset"]:
        raise SystemExit(
            f"{name} is the test dataset: it is read once, by the final evaluation, not here"
        )
    allowed = [data_config["val_dataset"], *data_config["train_datasets"]]
    if name not in allowed:
        raise SystemExit(f"{name} is not a validation or training dataset ({allowed})")


def load_forecast_model(args: argparse.Namespace):
    """The model plus, for --model-id, the run that logged it."""
    if args.checkpoint is not None:
        return load_model(args.checkpoint), None
    import mlflow

    mlflow.set_tracking_uri(args.tracking_uri)
    logged = mlflow.get_logged_model(args.model_id)
    pyfunc = mlflow.pyfunc.load_model(f"models:/{args.model_id}")
    return pyfunc.unwrap_python_model().model, logged.source_run_id


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    if args.log_to_mlflow and args.model_id is None:
        raise SystemExit("--log-to-mlflow needs --model-id (the video goes to its run)")

    data_config = yaml.safe_load(args.data_config.read_text())
    dataset = args.dataset or data_config["val_dataset"]
    check_dataset(dataset, data_config)

    model, run_id = load_forecast_model(args)
    root = zarr.open_group(store=data_config["zarr_store"], mode="r")
    series = root[dataset]
    dx, dy = grid_spacing(series.shape[2], series.shape[3])
    scale = np.asarray(NormalizationStats.load(args.stats).effective_std(args.std_mode))

    prediction, truth = forecast(model, series, data_config["context_steps"])
    rmse = rmse_per_step(prediction, truth, scale)
    n_leads = len(rmse) if args.max_steps is None else min(args.max_steps, len(rmse))
    leads = np.arange(0, n_leads, args.stride)

    sample = leads[:: max(1, len(leads) // args.color_sample)]
    vmin, vmax, diff_limit = forecast_limits(
        compute_field(truth[sample], args.field, dx, dy), FIELDS[args.field]["symmetric"]
    )
    log.info(
        "%s on %s: %d frames, field=%s, color range [%.4g, %.4g], diff +-%.4g",
        model.name,
        dataset,
        len(leads),
        args.field,
        vmin,
        vmax,
        diff_limit,
    )

    out_path = args.out_dir / f"{model.name}_{dataset}_{args.field}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render(
        model.name,
        dataset,
        args,
        truth,
        prediction,
        rmse,
        leads,
        (vmin, vmax, diff_limit),
        dx,
        dy,
        out_path,
    )
    log.info("wrote %s", out_path)

    if args.log_to_mlflow:
        import mlflow

        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(out_path), artifact_path="videos")
        log.info("attached to run %s under videos/", run_id)


def render(name, dataset, args, truth, prediction, rmse, leads, limits, dx, dy, out_path) -> None:
    vmin, vmax, diff_limit = limits
    cmap = FIELDS[args.field]["cmap"]
    blank = np.zeros((truth.shape[3], truth.shape[2]))

    fig, axes = plt.subplots(4, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [1, 1, 1, 0.9]})
    images = []
    for ax, label, panel_cmap, lims in [
        (axes[0], "truth", cmap, (vmin, vmax)),
        (axes[1], "prediction", cmap, (vmin, vmax)),
        (axes[2], "prediction - truth", "RdBu_r", (-diff_limit, diff_limit)),
    ]:
        im = ax.imshow(blank, cmap=panel_cmap, vmin=lims[0], vmax=lims[1], aspect="auto")
        ax.set_ylabel(label)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
        images.append(im)

    curve_ax = axes[3]
    lead_axis = np.arange(1, len(rmse) + 1)
    curve_ax.plot(lead_axis[: leads[-1] + 1], rmse[: leads[-1] + 1], lw=1)
    curve_ax.set_xlim(1, leads[-1] + 1)
    curve_ax.set_xlabel("lead time (steps)")
    curve_ax.set_ylabel("RMSE / std")
    curve_ax.grid(alpha=0.3)
    cursor = curve_ax.axvline(1, color="k", lw=1)
    # Keep the curve's width aligned with the panels above (they have colorbars).
    fig.colorbar(images[0], ax=curve_ax, shrink=0.9, pad=0.01).ax.set_visible(False)
    title = fig.suptitle("")
    fig.tight_layout()

    writer = FFMpegWriter(fps=args.fps)
    with writer.saving(fig, str(out_path), dpi=120):
        for k in leads:
            true_field = compute_field(truth[k : k + 1], args.field, dx, dy)[0]
            pred_field = compute_field(prediction[k : k + 1], args.field, dx, dy)[0]
            images[0].set_data(np.rot90(true_field))
            images[1].set_data(np.rot90(pred_field))
            images[2].set_data(np.rot90(pred_field - true_field))
            cursor.set_xdata([k + 1, k + 1])
            title.set_text(f"{name} on {dataset}: {args.field}, lead {k + 1}, RMSE {rmse[k]:.3f}")
            writer.grab_frame()
    plt.close(fig)


if __name__ == "__main__":
    main()
