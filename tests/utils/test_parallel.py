import statistics
import time

import pytest

from mhd_surrogate.utils import parallel


@pytest.fixture
def scripts_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, "SCRIPTS_DIR", tmp_path)
    return tmp_path


def _write_script(scripts_dir, name: str, body: str) -> None:
    (scripts_dir / name).write_text(body)


def test_run_subprocess_reports_success(scripts_dir):
    _write_script(scripts_dir, "ok.py", "import sys\nprint('hi')\nsys.exit(0)\n")

    result = parallel.run_subprocess("ok.py", [], "job1")

    assert result["returncode"] == 0
    assert result["script"] == "ok.py"
    assert result["label"] == "job1"
    assert result["elapsed"] >= 0


def test_run_subprocess_reports_failure_and_captures_stderr(scripts_dir):
    _write_script(
        scripts_dir, "fail.py", "import sys\nprint('boom', file=sys.stderr)\nsys.exit(1)\n"
    )

    result = parallel.run_subprocess("fail.py", [], "job2")

    assert result["returncode"] == 1
    assert "boom" in result["stderr"]


def test_run_subprocess_passes_args_through(scripts_dir):
    _write_script(
        scripts_dir,
        "echo_args.py",
        "import sys\nsys.exit(0 if sys.argv[1:] == ['--x', '1'] else 2)\n",
    )

    result = parallel.run_subprocess("echo_args.py", ["--x", "1"], "job3")

    assert result["returncode"] == 0


def test_run_parallel_runs_jobs_concurrently_not_serially(scripts_dir):
    _write_script(scripts_dir, "sleep_a_bit.py", "import time\ntime.sleep(0.3)\n")
    jobs = [("sleep_a_bit.py", [], f"job{i}") for i in range(4)]

    start = time.monotonic()
    results = parallel.run_parallel(jobs, workers=4)
    elapsed = time.monotonic() - start

    assert len(results) == 4
    assert all(r["returncode"] == 0 for r in results)
    # 4 jobs x 0.3s would be ~1.2s serially; concurrently it should be well under that.
    assert elapsed < 1.0


def test_log_failures_returns_only_failed_jobs(caplog):
    results = [
        {"script": "a.py", "label": "x", "returncode": 0, "elapsed": 0.1, "stderr": ""},
        {"script": "b.py", "label": "y", "returncode": 1, "elapsed": 0.1, "stderr": "oops"},
    ]

    with caplog.at_level("ERROR"):
        failures = parallel.log_failures(results)

    assert [f["label"] for f in failures] == ["y"]
    assert "oops" in caplog.text


# --- backends -------------------------------------------------------------
# statistics.mean is importable by Ray/process workers (a function defined in
# this test module may not be) and, unlike a builtin like `pow`, is a plain
# Python function, which Ray requires.


@pytest.mark.parametrize("backend", parallel.BACKENDS)
def test_map_tasks_returns_results_in_input_order(backend):
    tasks = [([1, 2, 3],), ([10],), ([4, 6],), ([0, 0, 9, 3],)]

    assert parallel.map_tasks(statistics.mean, tasks, backend=backend, workers=2) == [2, 10, 5, 3]


def test_map_tasks_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        parallel.map_tasks(statistics.mean, [([1],)], backend="dask")


def test_run_parallel_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        parallel.run_parallel([], workers=1, backend="dask")


@pytest.mark.parametrize("backend", ["sequential", "ray"])
def test_run_parallel_backends_run_every_job_and_report_failures(scripts_dir, backend):
    _write_script(scripts_dir, "ok.py", "print('hi')\n")
    _write_script(scripts_dir, "fail.py", "import sys\nsys.exit(3)\n")
    jobs = [("ok.py", [], "a"), ("fail.py", [], "b"), ("ok.py", [], "c")]

    results = parallel.run_parallel(jobs, workers=2, backend=backend)

    by_label = {r["label"]: r["returncode"] for r in results}
    assert by_label == {"a": 0, "b": 3, "c": 0}


def test_run_parallel_ray_accepts_per_script_memory_requests(scripts_dir):
    _write_script(scripts_dir, "heavy.py", "print('hi')\n")
    _write_script(scripts_dir, "light.py", "print('hi')\n")
    jobs = [("heavy.py", [], "a"), ("light.py", [], "b"), ("heavy.py", [], "c")]

    # light.py isn't listed: it requests no memory.
    results = parallel.run_parallel(jobs, workers=3, backend="ray", memory_gb={"heavy.py": 0.05})

    assert sorted(r["label"] for r in results) == ["a", "b", "c"]
    assert all(r["returncode"] == 0 for r in results)


def test_add_backend_args_defaults_and_choices():
    import argparse

    parser = argparse.ArgumentParser()
    parallel.add_backend_args(parser)

    args = parser.parse_args([])
    assert (args.backend, args.workers) == ("processes", None)
    args = parser.parse_args(["--backend", "ray", "--workers", "3"])
    assert (args.backend, args.workers) == ("ray", 3)
    with pytest.raises(SystemExit):
        parser.parse_args(["--backend", "dask"])
