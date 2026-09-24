import time

import pytest

from mhd_surrogate import parallel


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


def test_print_failures_returns_only_failed_jobs(capsys):
    results = [
        {"script": "a.py", "label": "x", "returncode": 0, "elapsed": 0.1, "stderr": ""},
        {"script": "b.py", "label": "y", "returncode": 1, "elapsed": 0.1, "stderr": "oops"},
    ]

    failures = parallel.print_failures(results)

    assert [f["label"] for f in failures] == ["y"]
    assert "oops" in capsys.readouterr().out
