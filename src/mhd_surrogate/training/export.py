"""Write a run's scores as a DVC metrics file.

The DVC `train` stage (dvc.yaml) declares this file as its metrics, so
`dvc metrics show` / `dvc metrics diff` compare the canonical model's
scores across commits. Scores are rounded to `SIGNIFICANT_DIGITS` and written with sorted keys:
retraining the same model on the GPU changes the scores around the 8th
significant digit (float32 reductions aren't bitwise deterministic), and
without rounding every rerun would rewrite the git-tracked file with noise.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

SIGNIFICANT_DIGITS = 6


def write_metrics(path: Path | str, scores: Mapping[str, Mapping[str, float]]) -> None:
    """`scores` maps a group ("val", "train.<dataset>") to its scalar scores.

    Undefined scores (NaN/inf, e.g. the oscillation period of a forecast
    without one) are left out: JSON has no NaN, and DVC can't diff it.
    """
    finite = {
        group: {
            k: float(f"{v:.{SIGNIFICANT_DIGITS}g}") for k, v in values.items() if math.isfinite(v)
        }
        for group, values in scores.items()
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite, indent=2, sort_keys=True) + "\n")
