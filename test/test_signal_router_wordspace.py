"""Dispatch test: routerKind selects between chart and signal paths."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import pytest
import torch
import Language
from Language import Chart


def test_chart_default_routerkind_is_chart():
    # Force a clean XML state so a previously-loaded fixture (e.g.
    # XOR_grammar.xml setting routerKind=signal) doesn't leak through
    # the singleton. Pass router_kind explicitly to bypass the XML read.
    chart = Chart(nInput=4, nOutput=4, max_depth=3, hidden_dim=16,
                  feature_dim=4, router_kind="chart")
    assert chart.router_kind == "chart"


def test_chart_routerkind_signal_dispatches_to_signal_router():
    chart = Chart(nInput=4, nOutput=4, max_depth=3, hidden_dim=16, feature_dim=4,
                  router_kind="signal")
    assert chart.router_kind == "signal"
    # Without attach_layer_ops, compose raises a clear runtime error.
    with pytest.raises(RuntimeError, match="attach_layer_ops"):
        chart.compose(torch.randn(1, 4, 4), word_space=None)


import torch.nn as nn


class _Stub(nn.Module):
    def forward(self, left, right):
        return left + right


class _StubWordSpace:
    def __init__(self):
        self.current_rules = {}
        self.generate_rules = {}
        self._compose_generation = 0

    def host_layer(self, tier, rule_name):
        return None


def test_signal_generate_emits_rules_dict_after_compose():
    chart = Chart(nInput=4, nOutput=4, max_depth=3, hidden_dim=16,
                  feature_dim=4, router_kind="signal")
    router = chart._ensure_signal_router()
    router.attach_layer_ops(ops=[_Stub()], rule_ids=[3], tier="S")
    ws = _StubWordSpace()
    target = torch.randn(2, 4, 4)
    chart.compose(target, word_space=ws)
    g = chart.generate(target, word_space=ws)
    assert isinstance(g, dict)
    assert "S" in g
    rows = g["S"]
    assert len(rows) == 2
    for row in rows:
        assert isinstance(row, list)
        for rid in row:
            assert rid == 3


def test_signal_compose_populates_current_rules():
    chart = Chart(nInput=4, nOutput=4, max_depth=3, hidden_dim=16,
                  feature_dim=4, router_kind="signal")
    router = chart._ensure_signal_router()
    router.attach_layer_ops(ops=[_Stub()], rule_ids=[7], tier="S")
    ws = _StubWordSpace()
    rules = chart.compose(torch.randn(2, 4, 4), word_space=ws)
    assert isinstance(rules, dict)
    assert "S" in rules
    rows = rules["S"]
    assert len(rows) == 2
    for row in rows:
        assert isinstance(row, list)
        for rid in row:
            assert isinstance(rid, int)
            # Emitted ids are global rule_ids (we passed [7] above).
            assert rid == 7
