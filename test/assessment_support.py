"""Saved-age prerequisites and independent reports on the proposed operator rungs."""
from contextvars import ContextVar
import json
import os
from pathlib import Path

import pytest

_current = ContextVar('assessment_test', default=None)


def record(model, vocabulary, *, scalar=None):
    current = _current.get()
    if current is None:
        return
    from AssessmentProbe import known_probe
    request, reports = current
    reports.append(known_probe(model, vocabulary, scalar=scalar, source=request.node.nodeid,
                               trace=bool(os.environ.get('ASSESSMENT_REPORT'))))


@pytest.fixture(autouse=True)
def _assessment_report(request):
    mark = request.node.get_closest_marker('assessment_rung')
    if mark is None:
        yield
        return
    reports = []
    token = _current.set((request, reports))
    try:
        yield
    finally:
        _current.reset(token)
        from AssessmentProbe import unavailable, format_table, readiness
        if not reports:
            reports.append(unavailable(request.node.nodeid,
                'fixture exercises structure or numeric tensors; no taught text vocabulary'))
        for report in reports:
            report['prerequisite'] = int(mark.args[0])
            report['readiness'] = readiness(report, report['prerequisite'])
            request.node.user_properties.append(('assessment', json.dumps(report)))
            request.node.add_report_section('teardown', 'assessment',
                report['readiness']+'\n'+format_table(report))
            path = os.environ.get('ASSESSMENT_REPORT')
            if path:
                with Path(path).open('a') as stream:
                    stream.write(json.dumps(report)+'\n')


def pytest_configure(config):
    config.addinivalue_line('markers', 'requires_mental_age(task): check the saved model against this age-chart row before the test')


@pytest.fixture(autouse=True)
def _check_saved_age(request):
    mark = request.node.get_closest_marker('requires_mental_age')
    if mark is None:
        return
    from Assessment import checkpoint_qualification
    artifact = request.getfixturevalue('phase_artifact')
    path = request.getfixturevalue('phase_registry').resolve(artifact.path)
    result = checkpoint_qualification(path, *mark.args)
    request.node.user_properties.append(('assessed_age', result['age']))
    if result['status'] != 'qualified':
        pytest.fail(f"{result['status']} for {mark.args[0]!r}: checkpoint age {result['age']:.2f}, "
                    f"requires {result['required']:.2f}", pytrace=False)


@pytest.fixture
def at_mental_age(request):
    """Record an unmet prerequisite as its own pytest outcome, also under xdist."""
    from Assessment import assess_age, at_age
    def run(model, years, measure):
        # This fixture receives a live subject. Its current training may have
        # changed its capability since construction or restoration.
        assess_age(model, all_rows=False)
        result = at_age(model, years, measure)
        if result['status'] == 'not qualified at that age':
            request.node.user_properties.append(('mental_age_qualification', result['status']))
            request.node.add_report_section('call', 'mental age',
                f"{result['status']} (has {result['age']:.2f}, requires {years:.2f})")
        return result
    return run


def pytest_report_teststatus(report):
    if report.when == 'call' and report.passed:
        for name, status in report.user_properties:
            if name == 'mental_age_qualification':
                return status, 'N', status
