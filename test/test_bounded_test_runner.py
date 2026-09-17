"""The development test gate must bound real processes, including their children."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


@pytest.fixture
def runner():
    return importlib.import_module("bounded_tests")


def _alive(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "stat="],
                            capture_output=True, text=True, timeout=2)
    return bool(result.stdout.strip()) and not result.stdout.lstrip().startswith("Z")


def _run(runner, tmp_path, source, **limits):
    return runner.run_guarded(
        [sys.executable, "-u", "-c", source], cwd=tmp_path,
        env=os.environ.copy(), log_path=tmp_path / "worker.log",
        memory_bytes=limits.pop("memory_bytes", 512 * 1024**2),
        timeout=limits.pop("timeout", 5), poll_seconds=.05,
        terminate_grace=.1, **limits)


def test_worker_exit_and_diagnostics_are_preserved(runner, tmp_path):
    result = _run(runner, tmp_path, "print('real failure'); raise SystemExit(7)")
    assert result["exit_code"] == 7
    assert result["reason"] == "exit"
    assert "real failure" in (tmp_path / "worker.log").read_text()


def test_timeout_kills_term_resistant_descendants(runner, tmp_path):
    child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    source = (
        "import subprocess,sys,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"child=subprocess.Popen([sys.executable,'-c',{child!r}]); "
        "Path('child.pid').write_text(str(child.pid)); time.sleep(60)")
    start = time.monotonic()
    result = _run(runner, tmp_path, source, timeout=.6)
    assert result["reason"] == "timeout"
    assert result["exit_code"] == 124
    assert time.monotonic() - start < 4
    assert not _alive(int((tmp_path / "child.pid").read_text()))


def test_aggregate_memory_includes_child_allocations(runner, tmp_path):
    child = "import time; data=bytearray(40*1024**2); time.sleep(60)"
    source = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"children=[subprocess.Popen([sys.executable,'-c',{child!r}]) for _ in range(2)]; "
        "Path('children.json').write_text(__import__('json').dumps([p.pid for p in children])); "
        "time.sleep(60)")
    result = _run(runner, tmp_path, source, memory_bytes=64 * 1024**2)
    assert result["reason"] == "memory"
    assert result["exit_code"] == 137
    assert result["peak_memory_bytes"] > 64 * 1024**2
    assert all(not _alive(pid) for pid in json.loads((tmp_path / "children.json").read_text()))


def test_monitor_failure_terminates_instead_of_running_unbounded(runner, tmp_path, monkeypatch):
    real_sample = runner.ProcessTree.sample
    calls = 0

    def broken(self):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OSError("probe cannot read process memory")
        return real_sample(self)

    monkeypatch.setattr(runner.ProcessTree, "sample", broken)
    result = _run(runner, tmp_path, "import time; time.sleep(60)")
    assert result["reason"] == "monitor_error"
    assert result["exit_code"] != 0
    assert not _alive(result["pid"])


def test_completed_worker_cannot_leave_live_children(runner, tmp_path):
    source = ("import subprocess,sys; from pathlib import Path; "
              "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
              "Path('child.pid').write_text(str(p.pid))")
    result = _run(runner, tmp_path, source)
    assert result["exit_code"] == 0
    assert not _alive(int((tmp_path / "child.pid").read_text()))


def test_suite_lock_excludes_another_process_and_releases(runner, tmp_path):
    lock = tmp_path / "suite.lock"
    code = ("from pathlib import Path; from bounded_tests import suite_lock; "
            f"\nwith suite_lock(Path({str(lock)!r})): pass")
    env = dict(os.environ, PYTHONPATH=str(Path(runner.__file__).parent))
    with runner.suite_lock(lock):
        other = subprocess.run([sys.executable, "-c", code], env=env,
                               capture_output=True, timeout=5)
        assert other.returncode != 0
        assert b"already running" in other.stderr
    with runner.suite_lock(lock):
        pass


def test_batches_keep_every_selected_case_once(runner):
    nodes = [f"test_{i // 7}.py::test_{i}" for i in range(50)]
    batches = runner.make_batches(nodes, batch_size=10, max_files=2)
    assert [node for batch in batches for node in batch] == nodes
    assert all(len(batch) <= 10 for batch in batches)
    assert all(len({n.split('::')[0] for n in batch}) <= 2 for batch in batches)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_limits_are_rejected_before_launch(runner, tmp_path, value):
    with pytest.raises(ValueError):
        _run(runner, tmp_path, "raise AssertionError('must not launch')", timeout=value)
    assert not (tmp_path / "worker.log").exists()


def test_real_pytest_coverage_fresh_workers_and_failure_receipt(runner, tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\nxfail_strict=true\n")
    (tmp_path / "test_cases.py").write_text(
        "import os,pytest,json\nfrom pathlib import Path\n"
        "def test_first():\n Path('first.pid').write_text(str(os.getpid()))\n"
        "def test_second():\n Path('second.pid').write_text(str(os.getpid()))\n"
        " assert json.loads(Path('result/result.json').read_text())['exit_code'] != 0\n"
        "@pytest.mark.skip(reason='explicit skip')\ndef test_skip(): pass\n"
        "@pytest.mark.xfail(reason='explicit expected failure')\ndef test_xfail(): assert False\n"
        "@pytest.fixture\ndef teardown_error():\n yield\n assert False,'teardown failed'\n"
        "def test_teardown(teardown_error): pass\n")
    result = runner.run_suite(
        root=tmp_path, selectors=["test_cases.py"], run_dir=tmp_path / "result",
        memory_bytes=512 * 1024**2, timeout=15, suite_timeout=60,
        batch_size=1, lock_path=tmp_path / "lock")
    assert result["exit_code"] != 0  # A passing call with failing teardown is a failure.
    assert len(result["selected"]) == 5
    assert result["completed"] == result["selected"]
    assert (tmp_path / "first.pid").read_text() != (tmp_path / "second.pid").read_text()
    assert json.loads((tmp_path / "result" / "result.json").read_text()) == result
    report = (tmp_path / "result" / "report.html").read_text()
    assert "teardown failed" in report
    assert "skipped" in report and "xfailed" in report


def test_empty_selection_is_not_a_green_suite(runner, tmp_path):
    (tmp_path / "test_empty.py").write_text("# No tests selected.\n")
    result = runner.run_suite(
        root=tmp_path, selectors=["test_empty.py"], run_dir=tmp_path / "result",
        memory_bytes=512 * 1024**2, timeout=15, suite_timeout=60,
        lock_path=tmp_path / "lock")
    assert result["exit_code"] == 5
    assert not result["selected"]


def test_source_change_cannot_receive_a_passing_receipt(runner, tmp_path):
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "test_change.py").write_text(
        "from pathlib import Path\n"
        "def test_change():\n Path(__file__).write_text('# changed during validation\\n')\n")
    result = runner.run_suite(
        root=tmp_path, selectors=["test"], run_dir=tmp_path / "result",
        memory_bytes=512 * 1024**2, timeout=15, suite_timeout=60,
        lock_path=tmp_path / "lock")
    assert result["exit_code"] != 0
    assert result["reason"] == "source_changed"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS physical-footprint API")
def test_macos_uses_compressed_footprint_instead_of_resident_size(runner):
    import ctypes
    from types import SimpleNamespace
    tree = runner.ProcessTree(os.getpid())

    def usage(pid, flavor, pointer):
        value = ctypes.cast(pointer, ctypes.POINTER(runner._DarwinUsage)).contents
        value.resident_size = 1024
        value.phys_footprint = 512 * 1024**2
        value.proc_start_abstime = 123
        return 0

    tree.libproc = SimpleNamespace(proc_pid_rusage=usage)
    assert tree.memory(os.getpid()) == (512 * 1024**2, 123)


def test_termination_persists_failure_and_cleans_child(runner, tmp_path):
    import signal
    (tmp_path / "test_wait.py").write_text(
        "import os,time\nfrom pathlib import Path\n"
        "def test_wait():\n Path('active.pid').write_text(str(os.getpid()))\n time.sleep(60)\n")
    code = (
        "from pathlib import Path; from bounded_tests import run_suite; "
        f"result=run_suite(root=Path({str(tmp_path)!r}), selectors=['test_wait.py'], "
        f"run_dir=Path({str(tmp_path / 'result')!r}), memory_bytes=512*1024**2, "
        f"timeout=15, suite_timeout=30, lock_path=Path({str(tmp_path / 'lock')!r})); "
        "raise SystemExit(result['exit_code'])")
    env = dict(os.environ, PYTHONPATH=str(Path(runner.__file__).parent))
    with (tmp_path / "outer.log").open("w") as log:
        proc = subprocess.Popen([sys.executable, "-c", code], cwd=tmp_path, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 12
            while not (tmp_path / "active.pid").exists() and time.monotonic() < deadline:
                time.sleep(.05)
            assert (tmp_path / "active.pid").exists()
            proc.send_signal(signal.SIGTERM)
            assert proc.wait(timeout=8) == 143
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=3)
    result = json.loads((tmp_path / "result" / "result.json").read_text())
    assert result["reason"] == "interrupted" and result["exit_code"] == 143
    assert not _alive(int((tmp_path / "active.pid").read_text()))
