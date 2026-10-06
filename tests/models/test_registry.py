import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from mhd_surrogate.models.baselines import MeanField, Persistence
from mhd_surrogate.models.registry import MODELS, build_model, load_model, model_class

ROOT = Path(__file__).resolve().parents[2]


def test_build_model_picks_the_class_by_name_and_passes_the_rest():
    model = build_model({"name": "mean_field", "chunk_t": 5})

    assert isinstance(model, MeanField)
    assert model.chunk_t == 5
    assert isinstance(build_model({"name": "persistence"}), Persistence)


def test_build_model_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        build_model({"name": "no_such_model"})


@pytest.mark.parametrize("name", ["mean_field", "persistence"])
def test_load_model_dispatches_on_the_checkpoint_name(tmp_path, name):
    model = build_model({"name": name})
    model.fit({"a": np.ones((3, 2, 4, 4), dtype=np.float32)})
    model.save(tmp_path)

    loaded = load_model(tmp_path)

    assert type(loaded) is type(model)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_each_entry_resolves_to_the_class_of_that_name(name):
    assert model_class(name).name == name


def test_every_model_config_is_registered():
    names = {
        yaml.safe_load(p.read_text())["name"] for p in (ROOT / "configs" / "model").glob("*.yaml")
    }

    assert names <= set(MODELS)


def test_importing_the_registry_imports_no_model():
    """In a fresh interpreter: this one has imported the models already."""
    modules = sorted({path.partition(":")[0] for path in MODELS.values()})
    code = (
        "import sys; import mhd_surrogate.models.registry; "
        f"print([m for m in {modules!r} if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"
