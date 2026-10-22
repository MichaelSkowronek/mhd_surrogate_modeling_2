from pathlib import Path

import pytest
from promote import KIND_OVERLAY, pin_model

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = """configMapGenerator:
  - name: serving-model
    literals:
      - MHD_MODEL_URI=models:/mhd-surrogate/1
"""


def test_pin_model_pins_the_overlays_version():
    pinned = pin_model(OVERLAY, "mhd-surrogate", "7")

    assert pinned == OVERLAY.replace("mhd-surrogate/1", "mhd-surrogate/7")
    assert pin_model(pinned, "mhd-surrogate", "7") == pinned


def test_pin_model_refuses_an_overlay_serving_another_model():
    with pytest.raises(ValueError, match="doesn't pin a version of smoke-mhd-surrogate"):
        pin_model(OVERLAY, "smoke-mhd-surrogate", "1")
    with pytest.raises(ValueError, match="doesn't pin a version"):
        pin_model(OVERLAY.replace("/1", "@champion"), "mhd-surrogate", "1")


@pytest.mark.skipif(not (ROOT / KIND_OVERLAY).exists(), reason="no deploy/ (the image has none)")
def test_the_deployed_overlay_pins_a_version_promote_can_move():
    text = (ROOT / KIND_OVERLAY).read_text()

    assert pin_model(text, "mhd-surrogate", "99") != text
