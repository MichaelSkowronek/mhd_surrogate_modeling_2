from omegaconf import OmegaConf

from mhd_surrogate.utils.hydra_resolvers import if_null


def test_if_null_falls_back_only_for_null():
    assert if_null(None, "default") == "default"
    assert if_null("outputs/x", "default") == "outputs/x"
    assert if_null(0, 5) == 0


def test_if_null_resolves_in_a_config_with_a_null_or_a_set_key():
    unset = OmegaConf.create({"resume": None, "dir": "${if_null:${resume},outputs/new}"})
    set_ = OmegaConf.create({"resume": "outputs/old", "dir": "${if_null:${resume},outputs/new}"})

    assert unset.dir == "outputs/new"
    assert set_.dir == "outputs/old"
