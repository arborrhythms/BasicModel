"""Recycle between complete tests without dropping coverage or hiding failures."""
from pathlib import Path
import json

import bounded_tests as runner


def test_recycle_on_memory_pressure_keeps_every_case_once(tmp_path):
    (tmp_path / "test_pressure.py").write_text(
        "import os,time,pytest\nfrom pathlib import Path\n"
        "retained=[]\n@pytest.mark.parametrize('case',range(5))\n"
        "def test_case(case):\n"
        " retained.append(bytearray(32*1024**2))\n time.sleep(.4)\n"
        " with Path('executions').open('a') as f: f.write(f'{case}:{os.getpid()}\\n')\n")
    result = runner.run_suite(
        root=tmp_path, selectors=["test_pressure.py"], run_dir=tmp_path / "result",
        memory_bytes=128 * 1024**2, timeout=20, suite_timeout=90,
        batch_size=64, lock_path=tmp_path / "lock")
    assert result["exit_code"] == 0, result["reason"]
    assert result["selected"] == result["completed"]
    executions = [line.split(':') for line in (tmp_path / "executions").read_text().splitlines()]
    assert [int(case) for case, _ in executions] == list(range(5))
    assert len({pid for _, pid in executions}) > 1
    assert any(worker.get("recycled") for worker in result["workers"])


def test_explicit_zero_exit_does_not_erase_an_earlier_failed_test(tmp_path):
    (tmp_path / "test_failure.py").write_text(
        "import pytest\ndef test_failure(): assert False,'real failure'\n"
        "def test_exit(): pytest.exit('exit zero',returncode=0)\n")
    result = runner.run_suite(
        root=tmp_path, selectors=["test_failure.py"], run_dir=tmp_path / "result",
        memory_bytes=512 * 1024**2, timeout=20, suite_timeout=90,
        lock_path=tmp_path / "lock")
    assert result["exit_code"] != 0


def test_time_pressure_recycles_after_complete_cases_without_dropping_coverage(tmp_path):
    (tmp_path / "test_duration.py").write_text(
        "import os,time,pytest\nfrom pathlib import Path\n"
        "@pytest.mark.parametrize('case',range(6))\n"
        "def test_case(case):\n"
        " time.sleep(1.2)\n"
        " with Path('executions').open('a') as f: f.write(f'{case}:{os.getpid()}\\n')\n")
    result = runner.run_suite(
        root=tmp_path, selectors=["test_duration.py"], run_dir=tmp_path / "result",
        memory_bytes=512 * 1024**2, timeout=6, suite_timeout=45,
        batch_size=64, lock_path=tmp_path / "lock")
    assert result["exit_code"] == 0, result["reason"]
    assert result["selected"] == result["completed"]
    executions = [line.split(":") for line in (tmp_path / "executions").read_text().splitlines()]
    assert [int(case) for case, _ in executions] == list(range(6))
    assert len({pid for _, pid in executions}) > 1
    assert any(worker.get("recycled") for worker in result["workers"])
    boundaries = [json.loads(p.read_text()) for p in
                  (tmp_path / "result").glob("worker-*.recycle.json")]
    assert any(value["reason"] == "time_boundary" for value in boundaries)


def test_live_worker_receipt_keeps_completed_cases_when_a_later_case_times_out(tmp_path):
    (tmp_path / "test_progress.py").write_text(
        "import json,time\nfrom pathlib import Path\n"
        "def test_first(): pass\n"
        "def test_second():\n"
        " receipt=json.loads(Path('result/worker-000.json').read_text())\n"
        " assert receipt['exit_code'] != 0\n"
        " assert receipt['completed'] == ['test_progress.py::test_first']\n"
        " assert len(receipt['selected']) == 2\n"
        " time.sleep(30)\n")
    result = runner.run_suite(
        root=tmp_path, selectors=["test_progress.py"], run_dir=tmp_path / "result",
        memory_bytes=512 * 1024**2, timeout=4, suite_timeout=20,
        batch_size=64, lock_path=tmp_path / "lock")
    assert result["exit_code"] == 124, result["reason"]
    assert result["reason"] == "timeout"
    assert result["completed"] == ["test_progress.py::test_first"]
    assert len(result["selected"]) == 2
    snapshot = json.loads((tmp_path / "result" / "worker-000.json").read_text())
    assert snapshot["exit_code"] != 0
    assert snapshot["completed"] == result["completed"]
