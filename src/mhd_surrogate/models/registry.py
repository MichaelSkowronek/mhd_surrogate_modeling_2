"""Build a model from its config, or load one from a checkpoint, by name."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mhd_surrogate.models.base import CHECKPOINT_META, SurrogateModel
from mhd_surrogate.models.baselines import MeanField, Persistence
from mhd_surrogate.models.dmd import DMD

MODELS: dict[str, type] = {cls.name: cls for cls in (MeanField, Persistence, DMD)}


def _model_class(name: str) -> type:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; known: {sorted(MODELS)}")
    return MODELS[name]


def build_model(config: Mapping[str, Any]) -> SurrogateModel:
    """`config` is the `model` config group: `name` picks the class, every
    other key is passed to its constructor."""
    kwargs = dict(config)
    return _model_class(kwargs.pop("name"))(**kwargs)


def load_model(directory: Path | str) -> SurrogateModel:
    meta = json.loads((Path(directory) / CHECKPOINT_META).read_text())
    return _model_class(meta["name"]).load(Path(directory))
