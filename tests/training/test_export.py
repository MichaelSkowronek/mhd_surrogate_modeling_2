import json
import math

from mhd_surrogate.training.export import write_metrics


def test_write_metrics_writes_grouped_finite_scores(tmp_path):
    path = tmp_path / "out" / "metrics.json"

    write_metrics(path, {"val": {"b": 2.0, "a": 1, "nan": math.nan}, "train.x": {"c": 0.123456789}})

    assert json.loads(path.read_text()) == {"val": {"a": 1.0, "b": 2.0}, "train.x": {"c": 0.123457}}


def test_write_metrics_is_byte_identical_for_the_same_scores(tmp_path):
    first, second = tmp_path / "1.json", tmp_path / "2.json"

    write_metrics(first, {"val": {"b": 2.0, "a": 1.0}})
    write_metrics(second, {"val": {"a": 1.0, "b": 2.0}})

    assert first.read_bytes() == second.read_bytes()


def test_write_metrics_hides_run_to_run_float_noise(tmp_path):
    """Two retrainings of the same model differ around the 8th digit."""
    first, second = tmp_path / "1.json", tmp_path / "2.json"

    write_metrics(first, {"val": {"rmse": 0.6185323712656381}})
    write_metrics(second, {"val": {"rmse": 0.6185323297306583}})

    assert first.read_bytes() == second.read_bytes()
