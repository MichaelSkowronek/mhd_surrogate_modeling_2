import numpy as np
import pytest

from mhd_surrogate.models.baselines import MeanField, Persistence
from mhd_surrogate.models.registry import build_model, load_model


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
