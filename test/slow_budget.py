"""Explicit repeated-work budgets for optional slow tests.

RUN_SLOW=1 defaults to half the historical repetitions (Alec, 2026-10-26).
RUN_SLOW_FRACTION=1 restores the historical work. This helper never selects
tests, changes an assertion, draws a seed, or changes a production setting.
"""
import math
import os


def work_fraction():
    if os.environ.get('RUN_SLOW') != '1':
        return 1.0
    try:
        fraction = float(os.environ.get('RUN_SLOW_FRACTION', '0.5'))
    except ValueError as exc:
        raise ValueError('RUN_SLOW_FRACTION must be greater than 0 and at most 1') from exc
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('RUN_SLOW_FRACTION must be greater than 0 and at most 1')
    return fraction


def slow_count(full, *, minimum=1):
    """Scale repeated work, retaining the caller's structural minimum."""
    if type(full) is not int or type(minimum) is not int or not 1 <= minimum <= full:
        raise ValueError('slow counts require integers with 1 <= minimum <= full')
    return max(minimum, math.ceil(full * work_fraction()))
