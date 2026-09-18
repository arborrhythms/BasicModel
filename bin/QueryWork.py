"""Ephemeral shared query cost, charged before selected reads or execution.

A boundary controller supplies the remaining episode allowance.  This object
has no semantic state, reset, child allowance, or checkpoint owner; committed
thought history records any durable account of the work actually spent.
"""
from collections import Counter
from types import MappingProxyType


class QueryWorkExhausted(RuntimeError):
    """A selected query has no remaining shared work allowance."""


class QueryWorkBudget:
    """One mutable, non-renewable allowance shared by a selected query tree."""

    def __init__(self, limit):
        if type(limit) is not int or limit < 0:
            raise ValueError("query work limit must be a non-negative integer")
        self._limit = limit
        self._spent = 0
        self._counts = Counter()

    @property
    def remaining(self):
        return self._limit - self._spent

    @property
    def spent(self):
        return self._spent

    @property
    def counts(self):
        return MappingProxyType(dict(self._counts))

    def consume(self, kind, units=1):
        """Atomically spend a named amount, returning false on overdraw."""
        if not isinstance(kind, str) or not kind:
            raise ValueError("query work requires a named cost")
        if type(units) is not int or units < 0:
            raise ValueError("query cost must be a non-negative integer")
        if units > self.remaining:
            return False
        self._spent += units
        if units:
            self._counts[kind] += units
        return True

    def require(self, kind, units=1):
        """Spend or raise without partially changing the allowance."""
        if not self.consume(kind, units):
            raise QueryWorkExhausted("shared query work_budget exhausted")


def capture_limits(work, max_nodes, max_records):
    """Tighten capture limits to leave shared work for a structural proof.

    These are local caps only: capture, traversal, and nested calls still
    charge the same meter.  Standalone readers retain their explicit limits.
    """
    if work is None:
        return max_nodes, max_records
    share = max(1, work.remaining // 3)
    return min(max_nodes, share), min(max_records, share)
