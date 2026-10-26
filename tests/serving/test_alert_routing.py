"""Where the alerts go (the monitoring Application's Alertmanager config,
deploy/argocd/apps/monitoring.yaml): every alert of the forecast service's
rules (deploy/k8s/monitoring/alerts.yaml) to email, the chart's own
nowhere, and the SMTP password from the mounted Secret, never inline.
amtool's config check and routing tests. Skipped without amtool
(`deploy/install-tools.sh amtool`) or outside a checkout (the image has no
deploy/)."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
MONITORING = ROOT / "deploy/argocd/apps/monitoring.yaml"
RULES = ROOT / "deploy/k8s/monitoring/alerts.yaml"
SECRETS_DIR = Path("/etc/alertmanager/secrets")  # where the operator mounts `secrets`

pytestmark = pytest.mark.skipif(
    shutil.which("amtool") is None or not MONITORING.exists(), reason="no amtool or no deploy/"
)


def alertmanager_values() -> dict:
    return yaml.safe_load(MONITORING.read_text())["spec"]["source"]["helm"]["valuesObject"][
        "alertmanager"
    ]


def service_alerts() -> list[dict]:
    """Each rule's alert as Alertmanager gets it: its name and labels."""
    groups = yaml.safe_load(RULES.read_text())["spec"]["groups"]
    return [
        {"alertname": rule["alert"], **rule["labels"]}
        for group in groups
        for rule in group["rules"]
        if "alert" in rule
    ]


@pytest.fixture(scope="module")
def config(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("alertmanager") / "alertmanager.yaml"
    path.write_text(yaml.safe_dump(alertmanager_values()["config"]))
    return path


def routes_to(config: Path, receiver: str, labels: dict) -> bool:
    """Whether an alert with `labels` reaches exactly `receiver`."""
    result = subprocess.run(
        [
            "amtool",
            "config",
            "routes",
            "test",
            f"--config.file={config}",
            f"--verify.receivers={receiver}",
            *(f"{key}={value}" for key, value in labels.items()),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def test_the_config_is_valid(config):
    result = subprocess.run(["amtool", "check-config", str(config)], capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr


def test_every_service_alert_is_emailed(config):
    alerts = service_alerts()

    assert len(alerts) == 7
    for alert in alerts:
        assert routes_to(config, "email", alert), alert


@pytest.mark.parametrize(
    "labels",
    [
        {"alertname": "Watchdog", "severity": "none"},
        # kind's static control-plane pods trip it.
        {"alertname": "KubePodNotReady", "namespace": "kube-system", "severity": "warning"},
        # The service's own namespace too: ForecastServiceDown covers an outage.
        {"alertname": "KubePodCrashLooping", "namespace": "mhd-serving", "severity": "warning"},
    ],
)
def test_the_charts_alerts_go_nowhere(config, labels):
    assert routes_to(config, "null", labels)
    assert not routes_to(config, "email", labels)


def test_the_smtp_password_comes_from_the_mounted_secret():
    values = alertmanager_values()
    smtp = values["config"]["global"]
    (secret,) = values["alertmanagerSpec"]["secrets"]

    assert "smtp_auth_password" not in smtp
    assert Path(smtp["smtp_auth_password_file"]).parent == SECRETS_DIR / secret
    for receiver in values["config"]["receivers"]:
        for email in receiver.get("email_configs", []):
            assert "auth_password" not in email
