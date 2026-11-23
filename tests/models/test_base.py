import numpy as np

from mhd_surrogate.models.base import DeviceBlock, FitHooks, fit_model, real_size


class ClosedForm:
    def fit(self, datasets):
        self.calls = [(datasets,)]


class Iterative:
    iterative = True

    def fit(self, datasets, hooks):
        self.calls = [(datasets, hooks)]


def test_fit_model_passes_hooks_only_to_an_iterative_model():
    hooks = FitHooks()
    closed, iterative = ClosedForm(), Iterative()

    fit_model(closed, {"a": 1}, hooks)
    fit_model(iterative, {"a": 1}, hooks)

    assert closed.calls == [({"a": 1},)]
    assert iterative.calls == [({"a": 1}, hooks)]


def test_default_hooks_discard_metrics_and_neither_validate_nor_save():
    hooks = FitHooks()

    assert hooks.validate is None and hooks.state_dir is None
    assert hooks.log_metrics({"loss": 1.0}, 3) is None


def test_real_size_counts_a_complex_entry_as_two_numbers():
    assert real_size(np.zeros((3, 4), dtype=np.float32)) == 12
    assert real_size(np.zeros((3, 4), dtype=np.complex64)) == 24


def test_a_device_block_decodes_a_member_when_its_frames_are_read():
    decoded = []

    def decode(states):
        decoded.append(states.copy())
        return states * 10.0

    block = DeviceBlock(np.arange(6.0).reshape(2, 3), decode)

    assert len(block) == 2 and block.steps == 3 and decoded == []
    assert block.frames(1).tolist() == [30.0, 40.0, 50.0]
    assert [d.tolist() for d in decoded] == [[3.0, 4.0, 5.0]]
    # Frames forecast as such are their own states.
    assert DeviceBlock(np.ones((1, 2))).frames(0).tolist() == [1.0, 1.0]
