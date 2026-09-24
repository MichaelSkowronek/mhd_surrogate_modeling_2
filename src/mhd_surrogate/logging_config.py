"""Shared logging setup for the CLI scripts.

Status/progress/diagnostic messages (warnings, orchestration progress) go
through `logging`; the actual computed results (per-dataset stats,
comparison tables) stay as plain `print()`, since they are formatted report
output meant to be read directly or piped/redirected, not log records.
"""

from __future__ import annotations

import argparse
import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"


def add_log_level_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="default: %(default)s",
    )


def setup_logging(level: str = "INFO") -> None:
    # basicConfig sets the root logger, which every third-party library
    # (matplotlib, zarr, mlflow, ...) inherits from -- setting it to the
    # requested level directly leaks their internal logging as noise (e.g.
    # matplotlib logs an INFO line for every ffmpeg frame it writes). So the
    # root stays at WARNING or the requested level, whichever is quieter,
    # and only this project's own loggers ("__main__" for a script run
    # directly, "mhd_surrogate" for library code) get the requested level.
    #
    # force=True: reconfigure even if something already called basicConfig
    # (e.g. a library import), rather than logging's default no-op-if-already-
    # configured behavior.
    requested = getattr(logging, level)
    logging.basicConfig(
        level=max(requested, logging.WARNING), format=LOG_FORMAT, datefmt=DATE_FORMAT, force=True
    )
    logging.getLogger("__main__").setLevel(requested)
    logging.getLogger("mhd_surrogate").setLevel(requested)

    # print() output (report content) and logging (status/progress, on
    # stderr) interleave correctly in a terminal, but stdout is fully
    # block-buffered rather than line-buffered when redirected/piped, so
    # without this a script's own printed report can appear to arrive
    # "late" relative to its logged status lines once output isn't a tty.
    sys.stdout.reconfigure(line_buffering=True)
