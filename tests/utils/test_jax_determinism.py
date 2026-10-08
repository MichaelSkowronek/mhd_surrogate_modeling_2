import os
import subprocess
import sys

import pytest

from mhd_surrogate.utils import jax_determinism
from mhd_surrogate.utils.jax_determinism import FLAG, enable_deterministic_ops


@pytest.fixture
def fresh_backend(monkeypatch):
    """Pretend JAX hasn't initialized its backend (in this process, earlier
    tests have)."""
    monkeypatch.setattr(jax_determinism.xla_bridge, "backends_are_initialized", lambda: False)


def test_the_flag_is_added_to_xla_flags_keeping_the_others(monkeypatch, fresh_backend):
    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_to=/tmp/x")

    enable_deterministic_ops()
    enable_deterministic_ops()  # idempotent

    assert os.environ["XLA_FLAGS"].split() == ["--xla_dump_to=/tmp/x", FLAG]


def test_the_flag_is_set_when_xla_flags_is_unset(monkeypatch, fresh_backend):
    monkeypatch.delenv("XLA_FLAGS", raising=False)

    enable_deterministic_ops()

    assert os.environ["XLA_FLAGS"] == FLAG


def test_it_refuses_once_the_backend_is_initialized(monkeypatch):
    monkeypatch.setattr(jax_determinism.xla_bridge, "backends_are_initialized", lambda: True)
    monkeypatch.delenv("XLA_FLAGS", raising=False)

    with pytest.raises(RuntimeError, match="already initialized"):
        enable_deterministic_ops()
    assert "XLA_FLAGS" not in os.environ


def test_xla_accepts_the_flag_in_a_fresh_process():
    """An unknown XLA flag aborts the backend's start-up; this one runs."""
    script = (
        "import jax.numpy as jnp\n"
        "from mhd_surrogate.utils.jax_determinism import enable_deterministic_ops\n"
        "enable_deterministic_ops()\n"
        "print(float(jnp.ones(3).sum()))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "3.0"
