"""Test config.

scripts/*.py are not part of the installed mhd_surrogate package, but their
computational core functions (as opposed to the CLI/plotting glue) are worth
unit testing directly, so make them importable as plain modules.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
