"""Optional pytest timing observer; enable with PYTEST_PLUGINS=perf_observer.

PERF_OBSERVER_DIR selects local JSON evidence. Collection, fixtures, random
streams and test outcomes are unchanged. Separate workers write separate files.
"""
import gc
import json
import os
from pathlib import Path
import time
from collections import defaultdict

import pytest

_started = time.perf_counter()
_phases = defaultdict(float)
_fixtures = defaultdict(lambda: [0, 0.])
_gc_started = 0.
_gc_seconds = 0.
_gc_calls = 0


def _gc_observer(phase, info):
    global _gc_started, _gc_seconds, _gc_calls
    if phase == 'start':
        _gc_started = time.perf_counter()
    else:
        _gc_seconds += time.perf_counter()-_gc_started
        _gc_calls += 1


def pytest_sessionstart(session):
    gc.callbacks.append(_gc_observer)


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    start = time.perf_counter()
    yield
    row = _fixtures[fixturedef.argname]
    row[0] += 1
    row[1] += time.perf_counter()-start


def pytest_runtest_logreport(report):
    _phases[report.when] += report.duration


def pytest_sessionfinish(session, exitstatus):
    gc.callbacks.remove(_gc_observer)
    output = Path(os.environ['PERF_OBSERVER_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    (output/(str(os.getpid())+'.json')).write_text(json.dumps(dict(
        seconds=time.perf_counter()-_started, phases=dict(_phases), fixtures=dict(_fixtures),
        gc_seconds=_gc_seconds, gc_calls=_gc_calls, selected=len(session.items),
        exitstatus=int(exitstatus)), indent=2)+'\n')
