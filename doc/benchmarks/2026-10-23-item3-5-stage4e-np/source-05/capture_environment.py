"""Loaded observers stay in this pytest worker, not its harness children."""
import os


def pytest_configure():
    observers = {'capture_environment', 'sweep_capture', 'arrival_capture', 'identity_keep_probe'}
    remaining = [name for name in os.environ.get('PYTEST_PLUGINS', '').split(',')
                 if name and name not in observers]
    if remaining:
        os.environ['PYTEST_PLUGINS'] = ','.join(remaining)
    else:
        os.environ.pop('PYTEST_PLUGINS', None)
