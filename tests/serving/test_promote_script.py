from pathlib import Path

import pytest
from promote import KIND_OVERLAY, checkpoint_md5, pin_model, pinned_md5

ROOT = Path(__file__).resolve().parents[2]
# The canonical model's checkpoint (promote.py's default --model-dir).
CANONICAL = "models/hankel_dmd/model"
OVERLAY = """configMapGenerator:
  - name: serving-model
    literals:
      - MHD_MODEL_URI=models:/mhd-surrogate/1
      - MHD_MODEL_CHECKPOINT_MD5=abc.dir
"""
LOCK = """schema: '2.0'
stages:
  train:
    cmd: train
    outs:
    - path: models/hankel_dmd/model
      hash: md5
      md5: f00.dir
  gate:
    cmd: gate
    outs:
    - path: models/hankel_dmd/gate.json
      hash: md5
      md5: ba5
"""


def test_pin_model_pins_the_version_and_the_checkpoint():
    pinned = pin_model(OVERLAY, "mhd-surrogate", "7", "def.dir")

    assert pinned == OVERLAY.replace("mhd-surrogate/1", "mhd-surrogate/7").replace(
        "abc.dir", "def.dir"
    )
    assert pinned_md5(pinned) == "def.dir"
    assert pin_model(pinned, "mhd-surrogate", "7", "def.dir") == pinned


def test_an_unfilled_checkpoint_pin_is_filled():
    unfilled = OVERLAY.replace("=abc.dir", "=")

    assert pinned_md5(unfilled) == ""
    assert pinned_md5(pin_model(unfilled, "mhd-surrogate", "1", "abc.dir")) == "abc.dir"


def test_pin_model_refuses_an_overlay_serving_another_model_or_pinning_no_checkpoint():
    with pytest.raises(ValueError, match="doesn't pin a version of smoke-mhd-surrogate"):
        pin_model(OVERLAY, "smoke-mhd-surrogate", "1", "x")
    with pytest.raises(ValueError, match="doesn't pin a version"):
        pin_model(OVERLAY.replace("/1", "@champion"), "mhd-surrogate", "1", "x")
    no_checkpoint = OVERLAY.replace("      - MHD_MODEL_CHECKPOINT_MD5=abc.dir\n", "")
    with pytest.raises(ValueError, match="doesn't pin a checkpoint"):
        pin_model(no_checkpoint, "mhd-surrogate", "1", "x")
    with pytest.raises(ValueError, match="doesn't pin a checkpoint"):
        pinned_md5(no_checkpoint)


def test_checkpoint_md5_reads_the_stage_output_from_dvc_lock():
    assert checkpoint_md5(LOCK, Path(CANONICAL)) == "f00.dir"
    with pytest.raises(ValueError, match="no output models/unet/model"):
        checkpoint_md5(LOCK, "models/unet/model")


in_checkout = pytest.mark.skipif(
    not (ROOT / KIND_OVERLAY).exists(), reason="no deploy/ (the image has none)"
)


@in_checkout
def test_the_deployed_overlay_pins_the_checkpoint_this_commit_gated():
    pinned = pinned_md5((ROOT / KIND_OVERLAY).read_text())
    gated = checkpoint_md5((ROOT / "dvc.lock").read_text(), CANONICAL)

    assert pinned == gated, (
        f"the kind overlay pins checkpoint {pinned or '(none)'}, but dvc.lock's canonical one "
        f"is {gated}: after `dvc repro`, run scripts/serving/promote.py in this pull request "
        "and commit the pin it writes"
    )


@in_checkout
def test_the_deployed_overlay_pins_a_version_promote_can_move():
    text = (ROOT / KIND_OVERLAY).read_text()

    assert pin_model(text, "mhd-surrogate", "99", "x") != text
