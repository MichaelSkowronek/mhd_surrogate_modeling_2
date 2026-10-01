import argparse
import logging
from unittest.mock import patch

from mhd_surrogate.utils.logging_config import add_log_level_arg, setup_logging


def test_add_log_level_arg_default_and_choices():
    parser = argparse.ArgumentParser()
    add_log_level_arg(parser)

    args = parser.parse_args([])
    assert args.log_level == "INFO"

    args = parser.parse_args(["--log-level", "DEBUG"])
    assert args.log_level == "DEBUG"


def test_add_log_level_arg_rejects_unknown_level():
    parser = argparse.ArgumentParser(exit_on_error=False)
    add_log_level_arg(parser)
    try:
        parser.parse_args(["--log-level", "NOPE"])
        raised = False
    except SystemExit, argparse.ArgumentError:
        raised = True
    assert raised


def test_setup_logging_applies_requested_level_to_own_loggers():
    setup_logging("DEBUG")
    assert logging.getLogger("__main__").level == logging.DEBUG
    assert logging.getLogger("mhd_surrogate").level == logging.DEBUG

    setup_logging("ERROR")
    assert logging.getLogger("__main__").level == logging.ERROR
    assert logging.getLogger("mhd_surrogate").level == logging.ERROR


def test_setup_logging_never_makes_root_noisier_than_warning():
    """Third-party libraries (matplotlib, zarr, ...) inherit the root level,
    so a verbose request (DEBUG/INFO) must not leak their internal logging.
    """
    setup_logging("DEBUG")
    assert logging.getLogger().level == logging.WARNING

    setup_logging("INFO")
    assert logging.getLogger().level == logging.WARNING


def test_setup_logging_honors_a_stricter_than_warning_request():
    """An ERROR request should quiet third-party WARNINGs too, not just ours."""
    setup_logging("ERROR")
    assert logging.getLogger().level == logging.ERROR


def test_setup_logging_reconfigures_stdout_for_line_buffering():
    with patch("sys.stdout") as mock_stdout:
        setup_logging("INFO")
    mock_stdout.reconfigure.assert_called_once_with(line_buffering=True)


def _flush_root_handlers():
    for handler in logging.getLogger().handlers:
        handler.flush()


def test_log_file_captures_debug_while_console_stays_at_requested_level(tmp_path, capsys):
    log_file = tmp_path / "run.log"
    setup_logging("WARNING", log_file=log_file)
    log = logging.getLogger("mhd_surrogate.test_file")

    log.debug("fine detail")
    log.warning("something odd")
    _flush_root_handlers()

    text = log_file.read_text()
    assert "fine detail" in text
    assert "something odd" in text
    err = capsys.readouterr().err
    assert "something odd" in err
    assert "fine detail" not in err


def test_log_file_excludes_third_party_debug_noise(tmp_path):
    log_file = tmp_path / "run.log"
    setup_logging("INFO", log_file=log_file)

    logging.getLogger("some_third_party").debug("internal chatter")
    logging.getLogger("some_third_party").warning("real warning")
    _flush_root_handlers()

    text = log_file.read_text()
    assert "internal chatter" not in text
    assert "real warning" in text


def test_log_file_creates_missing_parent_directories(tmp_path):
    log_file = tmp_path / "nested" / "dir" / "run.log"
    setup_logging("INFO", log_file=log_file)
    logging.getLogger("mhd_surrogate.test_file").info("hello")
    _flush_root_handlers()
    assert log_file.exists()


def test_repeated_setup_does_not_duplicate_file_lines(tmp_path):
    log_file = tmp_path / "run.log"
    setup_logging("INFO", log_file=log_file)
    setup_logging("INFO", log_file=log_file)
    logging.getLogger("mhd_surrogate.test_file").info("only once")
    _flush_root_handlers()
    assert log_file.read_text().count("only once") == 1


def test_without_log_file_own_loggers_use_requested_level():
    setup_logging("INFO")
    assert logging.getLogger("mhd_surrogate").level == logging.INFO
    assert not any(isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers)
