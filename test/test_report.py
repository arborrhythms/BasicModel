#!/usr/bin/env python
"""Run the test suite and generate an HTML report using the Report class.

Usage:
    make test          # runs pytest and generates report
    python test/test_report.py   # same, standalone
"""

import os
import sys
import time

# Ensure bin/ is importable
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
from visualize import Report


class ResultCollector:
    """Pytest plugin that collects test outcomes."""

    def __init__(self):
        self.results = []  # list of (nodeid, outcome, duration, message, stdout)
        self.start_time = None

    def pytest_sessionstart(self, session):
        self.start_time = time.time()

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or (report.when == "setup" and report.outcome == "error"):
            duration = report.duration
            if report.passed:
                outcome = "passed"
                msg = ""
            elif report.failed:
                outcome = "failed"
                msg = str(report.longrepr)[:200] if report.longrepr else ""
            elif report.skipped:
                if hasattr(report, 'wasxfail'):
                    outcome = "xfailed"
                    msg = report.wasxfail or ""
                else:
                    outcome = "skipped"
                    msg = str(report.longrepr)[:200] if report.longrepr else ""
            else:
                outcome = report.outcome
                msg = ""
            # Capture stdout from test (printed probe/query output)
            stdout = ""
            for section_name, content in report.sections:
                if "stdout" in section_name.lower():
                    stdout += content
            self.results.append((report.nodeid, outcome, duration, msg, stdout))

    def pytest_runtest_makereport(self, item, call):
        # Capture xfail
        pass

    @property
    def elapsed(self):
        return time.time() - self.start_time if self.start_time else 0


def generate_report(test_dir=None):
    """Run pytest on the test directory and produce an HTML report."""
    if test_dir is None:
        test_dir = os.path.dirname(os.path.abspath(__file__))

    collector = ResultCollector()

    # Run pytest programmatically, excluding this file to avoid recursion
    exit_code = pytest.main(
        [test_dir, "-v", "--tb=short", f"--ignore={os.path.abspath(__file__)}"],
        plugins=[collector],
    )

    # Build report
    report = Report()

    # Summary counts
    passed = sum(1 for r in collector.results if r[1] == "passed")
    failed = sum(1 for r in collector.results if r[1] == "failed")
    skipped = sum(1 for r in collector.results if r[1] == "skipped")
    xfailed = sum(1 for r in collector.results if r[1] == "xfailed")
    total = len(collector.results)

    # Summary table
    rows = [
        ["Total tests", str(total)],
        ["Passed", f'<span class="match">{passed}</span>'],
        ["Failed", f'<span class="mismatch">{failed}</span>' if failed else "0"],
    ]
    if xfailed:
        rows.append(["Expected failures", f'<span style="color:#cc0">{xfailed}</span>'])
    if skipped:
        rows.append(["Skipped", str(skipped)])
    rows.extend([
        ["Duration", f"{collector.elapsed:.1f}s"],
        ["Exit code", str(exit_code)],
    ])
    report.add_table("Test Summary", ["Metric", "Value"], rows)

    # Group results by test file
    by_file = {}
    for nodeid, outcome, duration, msg, stdout in collector.results:
        parts = nodeid.split("::", 1)
        filename = parts[0]
        testname = parts[1] if len(parts) > 1 else nodeid
        by_file.setdefault(filename, []).append((testname, outcome, duration, msg, stdout))

    # Per-file tables
    testpoint_output = []
    for filename, tests in sorted(by_file.items()):
        rows = []
        for testname, outcome, duration, msg, stdout in tests:
            if outcome == "passed":
                status = '<span class="match">PASS</span>'
            elif outcome == "failed":
                status = '<span class="mismatch">FAIL</span>'
            elif outcome == "xfailed":
                status = '<span style="color:#cc0">XFAIL</span>'
            elif outcome == "skipped":
                status = "SKIP"
            else:
                status = outcome.upper()
            row = [testname, status, f"{duration:.3f}s"]
            if msg:
                row.append(f"<pre>{msg[:200]}</pre>")
            else:
                row.append("")
            rows.append(row)
            # Collect testpoint stdout for the diagnostic section
            if "test_testpoint" in filename and stdout.strip():
                testpoint_output.append((testname, stdout.strip()))
        report.add_table(filename, ["Test", "Status", "Duration", "Details"], rows)

    # Testpoint diagnostic section -- embedding probes and server queries
    if testpoint_output:
        tp_rows = []
        for testname, output in testpoint_output:
            short_name = testname.split("::")[-1] if "::" in testname else testname
            tp_rows.append([short_name, f"<pre>{output}</pre>"])
        report.add_table("Testpoint: Embedding Probes &amp; Server Queries",
                         ["Probe", "Output"], tp_rows)

    path = report.write_html()
    return exit_code, path


if __name__ == "__main__":
    exit_code, path = generate_report()
    sys.exit(exit_code)
