from mhd_surrogate.models.base import FitHooks, fit_model


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
