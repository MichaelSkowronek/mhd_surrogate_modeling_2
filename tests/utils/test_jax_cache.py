import subprocess
import sys

import jax
import pytest

from mhd_surrogate.utils.jax_cache import enable_compilation_cache

CACHE_KEYS = (
    "jax_compilation_cache_dir",
    "jax_compilation_cache_max_size",
    "jax_persistent_cache_min_compile_time_secs",
)


@pytest.fixture
def restore_jax_config():
    saved = {key: jax.config.values[key] for key in CACHE_KEYS}
    yield
    for key, value in saved.items():
        jax.config.update(key, value)


def test_enable_compilation_cache_sets_config_and_creates_dir(tmp_path, restore_jax_config):
    cache_dir = tmp_path / "nested" / "jax"

    path = enable_compilation_cache(cache_dir, max_size_gb=0.5, min_compile_time_secs=2.0)

    assert path == cache_dir.resolve()
    assert path.is_dir()
    assert jax.config.values["jax_compilation_cache_dir"] == str(path)
    assert jax.config.values["jax_compilation_cache_max_size"] == 512 * 1024**2
    assert jax.config.values["jax_persistent_cache_min_compile_time_secs"] == 2.0


def test_compiled_program_is_written_to_cache(tmp_path):
    # A fresh process: JAX initializes the cache on its first compile, which
    # in this one other tests have already triggered.
    script = (
        "import sys, jax, jax.numpy as jnp\n"
        "from mhd_surrogate.utils.jax_cache import enable_compilation_cache\n"
        "enable_compilation_cache(sys.argv[1], min_compile_time_secs=0)\n"
        "jax.jit(lambda x: jnp.sin(x) @ x.T)(jnp.ones((8, 8))).block_until_ready()\n"
    )
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True)

    assert any(p.is_file() for p in tmp_path.rglob("*"))
