"""Proposed rung reports accompany operator assertions; they do not gate them."""
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
