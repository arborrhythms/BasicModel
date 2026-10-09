#!/usr/bin/env python
"""Bounded test entry point: make test, or python test/test_report.py [test paths]."""
import sys
from bounded_tests import main


def generate_report(test_dir=None):
    """Compatibility entry point; the HTML report accompanies result.json."""
    return main([str(test_dir)] if test_dir is not None else [])


if __name__ == "__main__":
    exit_code, path = main()
    sys.exit(exit_code)
