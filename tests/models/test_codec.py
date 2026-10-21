import subprocess
import sys

import numpy as np
import pytest

from mhd_surrogate.models.codec import CODECS, build_codec, codec_class, load_codec
from mhd_surrogate.models.pod_codec import PODCodec
from mhd_surrogate.models.registry import load_model


def test_build_codec_picks_the_class_by_name_and_passes_the_rest():
    codec = build_codec({"name": "pod", "rank": 7, "chunk_t": 5})

    assert isinstance(codec, PODCodec)
    assert (codec.rank, codec.chunk_t) == (7, 5)


def test_build_codec_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown codec"):
        build_codec({"name": "no_such_codec"})


@pytest.mark.parametrize("name", sorted(CODECS))
def test_each_entry_resolves_to_the_class_of_that_name(name):
    assert codec_class(name).name == name


def test_a_codec_checkpoint_loads_as_a_codec_only(tmp_path):
    codec = build_codec({"name": "pod", "rank": 2})
    codec.fit({"a": np.random.default_rng(0).normal(size=(4, 2, 3, 3)).astype(np.float32)})
    codec.save(tmp_path)

    assert isinstance(load_codec(tmp_path), PODCodec)
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path)


def test_importing_the_codec_registry_imports_no_codec():
    """In a fresh interpreter: this one has imported the codecs already."""
    modules = sorted({path.partition(":")[0] for path in CODECS.values()})
    code = (
        "import sys; import mhd_surrogate.models.codec; "
        f"print([m for m in {modules!r} if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"
