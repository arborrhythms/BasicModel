"""Central RUN_SLOW gate for long training, quality and compilation checks."""
import os
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long training/quality/compilation check; opt in with RUN_SLOW=1")


def pytest_collection_modifyitems(items):
    if os.environ.get("RUN_SLOW") == "1":
        return
    skipped = pytest.mark.skip(reason="slow -- set RUN_SLOW=1")
    for item in items:
        if item.get_closest_marker("slow") is not None:
            item.add_marker(skipped)
