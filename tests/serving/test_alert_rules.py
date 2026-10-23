"""The alert rules (deploy/k8s/monitoring/alerts.yaml) fire when they should
and stay quiet otherwise: promtool's rule unit tests, on synthetic series
of the metrics the server exposes (`serving/metrics.py`, its buckets and
label formatting). Skipped without promtool (`deploy/install-tools.sh
promtool`) or outside a checkout (the image has no deploy/)."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from prometheus_client.utils import floatToGoString

from mhd_surrogate.serving.metrics import ENERGY_BUCKETS, SCALE_BUCKETS, SHIFT_BUCKETS

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "deploy/k8s/monitoring/alerts.yaml"

pytestmark = pytest.mark.skipif(
    shutil.which("promtool") is None or not RULES.exists(), reason="no promtool or no deploy/"
)


def rule(name: str) -> dict:
    groups = yaml.safe_load(RULES.read_text())["spec"]["groups"]
    (found,) = [r for group in groups for r in group["rules"] if r.get("alert") == name]
    return found


def fires(name: str, **labels) -> list[dict]:
    """promtool's expectation: `name` firing, with `labels`."""
    defined = rule(name)
    return [
        {
            "exp_labels": {**defined["labels"], **labels},
            "exp_annotations": defined["annotations"],
        }
    ]


def counter(series: str, per_minute: float, minutes: int = 80) -> dict:
    return {"series": series, "values": f"0+{per_minute}x{minutes}"}


def histogram(
    metric: str, value: float, bounds, labels: str = "", per_minute: int = 2, minutes: int = 80
) -> list[dict]:
    """A histogram's cumulative bucket series for observations of `value`,
    `per_minute` of them each minute."""
    series = []
    for bound in [*bounds, float("inf")]:
        rate = per_minute if value <= bound else 0
        le = f'le="{floatToGoString(bound)}"'
        series.append(counter(f"{metric}_bucket{{{', '.join(filter(None, [labels, le]))}}}", rate))
    return series


ENERGY = "mhd_forecast_energy_ratio"
FORECASTS = 'mhd_serve_requests_total{route="/v1/forecast", status="%s"}'

CASES = {
    "ForecastServiceDown": [
        ([{"series": 'up{namespace="mhd-serving", job="serve"}', "values": "0x10"}], 7, True),
        ([], 7, True),
        ([{"series": 'up{namespace="mhd-serving", job="serve"}', "values": "1x10"}], 7, False),
    ],
    "ForecastsFailing": [
        ([{"series": "mhd_forecasts_failed_total", "values": "0 0 0 1 1 1 1 1 1"}], 5, True),
        ([counter("mhd_forecasts_failed_total", 0)], 20, False),
    ],
    "ForecastServiceSaturated": [
        ([counter(FORECASTS % "200", 8), counter(FORECASTS % "503", 4)], 20, True),
        ([counter(FORECASTS % "200", 9), counter(FORECASTS % "503", 1)], 20, False),
    ],
    "ForecastBlowUp": [
        (histogram(ENERGY, 2.5, ENERGY_BUCKETS), 10, True),
        (histogram(ENERGY, 0.75, ENERGY_BUCKETS), 10, False),
    ],
    "ForecastCollapse": [
        (histogram(ENERGY, 0.4, ENERGY_BUCKETS), 65, True),
        (histogram(ENERGY, 0.75, ENERGY_BUCKETS), 65, False),
    ],
    "ForecastInputDrift": [
        (histogram("mhd_context_mean_shift", 0.2, SHIFT_BUCKETS, 'channel="u_y"'), 65, True),
        (histogram("mhd_context_mean_shift", -0.2, SHIFT_BUCKETS, 'channel="u_y"'), 65, True),
        (histogram("mhd_context_mean_shift", 0.01, SHIFT_BUCKETS, 'channel="u_y"'), 65, False),
    ],
    "ForecastInputScaleDrift": [
        (histogram("mhd_context_std_ratio", 1.8, SCALE_BUCKETS, 'channel="u_y"'), 65, True),
        (histogram("mhd_context_std_ratio", 0.6, SCALE_BUCKETS, 'channel="u_y"'), 65, True),
        (histogram("mhd_context_std_ratio", 1.0, SCALE_BUCKETS, 'channel="u_y"'), 65, False),
    ],
}
LABELLED = {"ForecastInputDrift", "ForecastInputScaleDrift"}


def promtool(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["promtool", *args], cwd=tmp_path, capture_output=True, text=True)


@pytest.fixture
def rules_file(tmp_path) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(yaml.safe_load(RULES.read_text())["spec"]))
    return path


def test_the_rules_are_valid_promql(rules_file, tmp_path):
    result = promtool(tmp_path, "check", "rules", rules_file.name)

    assert result.returncode == 0, result.stdout + result.stderr


def test_every_alert_has_cases():
    groups = yaml.safe_load(RULES.read_text())["spec"]["groups"]
    alerts = {r["alert"] for group in groups for r in group["rules"]}

    assert alerts == set(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_alert_fires_when_it_should_and_not_otherwise(name, rules_file, tmp_path):
    labels = {"channel": "u_y"} if name in LABELLED else {}
    tests = [
        {
            "interval": "1m",
            "input_series": series,
            "alert_rule_test": [
                {
                    "eval_time": f"{minute}m",
                    "alertname": name,
                    "exp_alerts": fires(name, **labels) if firing else [],
                }
            ],
        }
        for series, minute, firing in CASES[name]
    ]
    test_file = tmp_path / "test.yaml"
    test_file.write_text(yaml.safe_dump({"rule_files": [rules_file.name], "tests": tests}))

    result = promtool(tmp_path, "test", "rules", test_file.name)

    assert result.returncode == 0, result.stdout + result.stderr
