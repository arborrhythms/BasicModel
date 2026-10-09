"""A stale compiler artifact gets one visible retry, never an assertion waiver."""
import importlib
import json

import pytest


@pytest.mark.parametrize("after_retry", ["pass", "cache_failure", "assertion", "teardown"])
def test_stale_header_retry_is_once_and_preserves_real_failures(tmp_path, after_retry, monkeypatch):
    runner = importlib.import_module("bounded_tests")
    monkeypatch.setenv("TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS", "1")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_cache.py").write_text('''
import os
from pathlib import Path
import pytest

class CppCompileError(RuntimeError):
    pass
CppCompileError.__module__ = "torch._inductor.exc"
MESSAGE = "fatal error: file 'header.h' has been modified since the precompiled header 'cached.h.pch' was built"

@pytest.fixture
def cleanup():
    yield
    if OUTCOME == "teardown":
        assert False, "real teardown failure"

def test_compile(cleanup):
    with Path("attempts").open("a") as f:
        f.write("attempt\\n")
    if os.environ.get("TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS") != "0" or OUTCOME == "cache_failure":
        raise CppCompileError(MESSAGE)
    if OUTCOME == "assertion":
        assert False, "real assertion after retry"

def test_other():
    with Path("other-attempts").open("a") as f:
        f.write("attempt\\n")
'''.replace('OUTCOME', repr(after_retry)))
    # Exercise the supervisor and fresh worker, with cheap compiler-fault
    # injection instead of mutating a shared live Torch cache.
    result = runner.run_suite(root=tmp_path, selectors=["test_cache.py"],
        run_dir=tmp_path / "result", memory_bytes=512 * 1024**2,
        timeout=60, suite_timeout=180, batch_size=2, lock_path=tmp_path / "lock")
    assert sorted(result["completed"]) == sorted(result["selected"])
    assert len(result["completed"]) == 2
    assert (tmp_path / "other-attempts").read_text().count("attempt") == 1
    attempts = (tmp_path / "attempts").read_text().count("attempt")
    assert attempts == (1 if after_retry == "teardown" else 2)
    assert result["exit_code"] == (0 if after_retry == "pass" else 1)
    retries = result["compile_cache_retries"]
    assert len(retries) == (0 if after_retry == "teardown" else 1)
    if retries:
        assert retries[0]["technique"] == "retry_without_precompiled_headers"
        first = json.loads((tmp_path / "result" / "worker-000.json").read_text())
        assert any(r["outcome"] == "failed" for r in first["reports"])
        assert "compile_cache_retry" in (tmp_path / "result" / "report.html").read_text()


def test_ordinary_compiler_errors_and_assertions_are_not_cache_failures():
    from pytest_worker import compile_cache_failure
    from torch._inductor.exc import CppCompileError, InductorError
    message = "file 'a.h' has been modified since the precompiled header 'a.pch' was built"
    assert compile_cache_failure(AssertionError(message)) is None
    assert compile_cache_failure(CppCompileError(["c++"], "error: invalid generated C++")) is None
    cause = CppCompileError(["c++"], message)
    wrapper = InductorError(cause, None)
    assert compile_cache_failure(wrapper)["kind"] == "stale_precompiled_header"
    for failure in (AssertionError("real assertion"), RuntimeError("unrelated failure")):
        failure.__context__ = cause
        assert compile_cache_failure(failure) is None
