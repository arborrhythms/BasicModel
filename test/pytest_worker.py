"""Private disposable pytest worker. Invoke through test_report.py for limits."""
import json
import os
from pathlib import Path
import sys
import pytest


def compile_cache_failure(exception):
    """Recognize a stale Inductor PCH, not arbitrary compiler/test failures.

    Inspect typed exceptions, including backend wrappers, instead of matching
    assertion text in a rendered traceback. The fresh-worker retry disables
    only PCH reuse; it neither edits the shared cache nor disables compilation.
    """
    pending, seen = [exception], set()
    while pending:
        error = pending.pop()
        if not isinstance(error, BaseException) or id(error) in seen:
            continue
        seen.add(id(error))
        types = {(cls.__module__, cls.__name__) for cls in type(error).__mro__}
        if ("torch._inductor.exc", "CppCompileError") in types:
            message = str(error)
            if ("has been modified since the precompiled header" in message
                    and "was built" in message):
                return dict(kind="stale_precompiled_header",
                            technique="retry_without_precompiled_headers", message=message)
        # Only Torch compiler wrappers can forward a cache fault. An
        # assertion or unrelated exception raised while handling it is a
        # separate failure, even if Python retained that fault as context.
        if ("torch._dynamo.exc", "BackendCompilerFailed") in types:
            pending.extend((error.__cause__, error.__context__,
                            getattr(error, "inner_exception", None)))
    return None


class ResultCollector:
    def __init__(self, recycle_file=None, result_path=None):
        self.selected, self.completed, self.reports = [], [], []
        self.slow_selected = []
        self.recycle_file = Path(recycle_file) if recycle_file else None
        self.recycled = False
        self.failed = False
        self.result_path = Path(result_path) if result_path else None
        self.active = None
        self.device = None

    def publish(self, exit_code=125):
        """Persist completed cases even while a later test is still running."""
        if self.result_path is None:
            return
        temporary = self.result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(dict(
            exit_code=exit_code, selected=self.selected, completed=self.completed,
            slow_selected=self.slow_selected,
            reports=self.reports, recycled=self.recycled, active=self.active,
            device=self.device), indent=2) + "\n")
        temporary.replace(self.result_path)

    def pytest_collection_finish(self, session):
        self.selected = [item.nodeid for item in session.items]
        self.slow_selected = [item.nodeid for item in session.items
                              if item.get_closest_marker("slow") is not None]
        self.publish()

    def pytest_runtest_logstart(self, nodeid, location):
        self.active = nodeid
        self.publish()

    def pytest_runtest_logfinish(self, nodeid, location):
        self.completed.append(nodeid)
        self.active = None
        self.publish()
        if (self.recycle_file is not None and self.recycle_file.exists()
                and len(self.completed) < len(self.selected)):
            self.recycled = True
            pytest.exit("Resource boundary: recycle worker after this completed test",
                        returncode=1 if self.failed else 0)

    def pytest_runtest_logreport(self, report):
        self.failed = self.failed or report.failed
        if report.when != "call" and not (report.failed or report.skipped):
            return
        outcome = report.outcome
        if hasattr(report, "wasxfail"):
            outcome = "xfailed" if report.skipped else "xpassed"
        record = dict(
            nodeid=report.nodeid, phase=report.when, outcome=outcome,
            duration=report.duration, message=str(report.longrepr or "")[-4000:],
            stdout="\n".join(content for name, content in report.sections if "stdout" in name.lower())[-4000:])
        if getattr(report, "compile_cache_failure", None) is not None:
            record["compile_cache_failure"] = report.compile_cache_failure
        self.reports.append(record)
        self.publish()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        result = yield
        report = result.get_result()
        if report.failed and call.excinfo is not None:
            report.compile_cache_failure = compile_cache_failure(call.excinfo.value)


def configure_device():
    """Resolve a requested accelerator before model imports, without CPU fallback."""
    device = os.environ.get("BASICMODEL_DEVICE", "cpu").strip().lower()
    if device == "cpu":
        return device
    import torch
    if device == "gpu":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            raise RuntimeError("GPU training was requested but no accelerator is available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS training was requested but MPS is unavailable")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    if device != "mps" and not device.startswith("cuda"):
        raise ValueError(f"Unsupported test device: {device}")
    os.environ["BASICMODEL_DEVICE"] = device
    return str(torch.device(device))


def main():
    request_path, result_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    collector = ResultCollector(request.get("recycle_file"), result_path)
    args = [*request["selectors"], "-q", "--tb=short", "-p", "no:cacheprovider", "-p", "no:xdist"]
    if request.get("collect"):
        args.append("--collect-only")
    for key, option in (("keyword", "-k"), ("marker", "-m")):
        if request.get(key):
            args.extend([option, request[key]])
    code = 125
    try:
        collector.device = configure_device()
        code = int(pytest.main(args, plugins=[collector]))
    finally:
        collector.publish(code)
    # A recycled worker has published its final receipt.  Exit directly so
    # interpreter shutdown handlers cannot keep the process boundary alive.
    if collector.recycled:
        os._exit(code)
    return code


if __name__ == "__main__":
    sys.exit(main())
