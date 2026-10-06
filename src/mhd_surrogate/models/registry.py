"""Build a model from its config, or load one from a checkpoint, by name.

Models are registered by import path, not imported here: a model's module is
imported only when that model is built or loaded. Importing the registry
therefore doesn't import every model and its dependencies, and the DVC
`train` stage depends only on the canonical model's code, not on every
model's (see `tests/training/test_dvc_train_stage.py`).
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mhd_surrogate.models.base import CHECKPOINT_META, SurrogateModel

# name -> "module:Class"; the name is also the class's `name` attribute.
MODELS: dict[str, str] = {
    "mean_field": "mhd_surrogate.models.baselines:MeanField",
    "persistence": "mhd_surrogate.models.baselines:Persistence",
    "dmd": "mhd_surrogate.models.dmd:DMD",
    "hankel_dmd": "mhd_surrogate.models.hankel_dmd:HankelDMD",
    "unet": "mhd_surrogate.models.unet:UNetSurrogate",
}


def model_class(name: str) -> type:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; known: {sorted(MODELS)}")
    module, _, cls = MODELS[name].partition(":")
    return getattr(importlib.import_module(module), cls)


def build_model(config: Mapping[str, Any]) -> SurrogateModel:
    """`config` is the `model` config group: `name` picks the class, every
    other key is passed to its constructor."""
    kwargs = dict(config)
    return model_class(kwargs.pop("name"))(**kwargs)


def load_model(directory: Path | str) -> SurrogateModel:
    meta = json.loads((Path(directory) / CHECKPOINT_META).read_text())
    return model_class(meta["name"]).load(Path(directory))
