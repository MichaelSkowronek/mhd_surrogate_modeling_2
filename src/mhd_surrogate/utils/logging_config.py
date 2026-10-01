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
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"
FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def add_log_level_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="default: %(default)s",
    )


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure console logging, and optionally a per-run log file.

    `level` is the console verbosity. With `log_file`, the file additionally
    captures everything this project's own loggers emit down to DEBUG
    regardless of `level` (it is the post-mortem record, e.g. for a run that
    diverged overnight), with full dates in the timestamps. Parent
    directories are created. Calling this again replaces the previous
    handlers, so the file is never written to twice.
    """
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
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    console.setLevel(requested)
    handlers: list[logging.Handler] = [console]

    own_level = requested
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, FILE_DATE_FORMAT))
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)
        # The loggers must pass DEBUG records for the file handler to see
        # them; the console handler's own level keeps the console at
        # `requested`. Third-party loggers are still gated by the root
        # level below, so their DEBUG noise stays out of the file too.
        own_level = logging.DEBUG

    logging.basicConfig(level=max(requested, logging.WARNING), handlers=handlers, force=True)
    logging.getLogger("__main__").setLevel(own_level)
    logging.getLogger("mhd_surrogate").setLevel(own_level)

    # print() output (report content) and logging (status/progress, on
    # stderr) interleave correctly in a terminal, but stdout is fully
    # block-buffered rather than line-buffered when redirected/piped, so
    # without this a script's own printed report can appear to arrive
    # "late" relative to its logged status lines once output isn't a tty.
    sys.stdout.reconfigure(line_buffering=True)
