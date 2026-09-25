"""Test config.

scripts/**/*.py are not part of the installed mhd_surrogate package, but
their computational core functions (as opposed to the CLI/plotting glue) are
worth unit testing directly, so make each scripts/ subdirectory importable as
a plain module directory (e.g. `from check_split import per_timestep_stats`).
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

for subdir in sorted(SCRIPTS_DIR.iterdir()):
    if subdir.is_dir() and not subdir.name.startswith("__"):
        sys.path.insert(0, str(subdir))
