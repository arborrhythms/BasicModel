"""Match fresh initializations without modifying any standing assertion."""
import os
import pytest
import gate_runtime


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    marker = item.get_closest_marker('shared_training')
    group = 'shared:' + marker.args[0] if marker else item.nodeid
    gate_runtime.begin(group)


def pytest_runtest_logreport(report):
    gate_runtime.flush()


def pytest_sessionfinish(session, exitstatus):
    gate_runtime.flush()
