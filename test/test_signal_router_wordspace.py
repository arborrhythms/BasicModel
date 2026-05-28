"""Signal-router-on-WordSubSpace tests.

Stage 3 cleanup (2026-05-27): the chart was retired; the signal router
(``LanguageLayer``) is the canonical parser, owned directly by
``WordSubSpace.languageLayer``. These tests exercise the LanguageLayer
in isolation, the way the chart tests used to drive ``Chart`` directly.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import pytest
import torch
import torch.nn as nn

import Language
from Language import LanguageLayer


def _make_signal_router():
    return LanguageLayer(
        n_input=4, n_output=4,
        hidden_dim=16, feature_dim=4,
        max_depth=3, temperature=1.0,
    )


def test_language_layer_constructs_without_ops():
    """LanguageLayer can be constructed before any ops are attached."""
    router = _make_signal_router()
    assert isinstance(router, LanguageLayer)
    assert not router._unary_layers and not router._binary_layers


def test_language_layer_compose_without_ops_raises():
    """Calling compose before any ops are attached fails loud."""
    router = _make_signal_router()
    with pytest.raises(RuntimeError, match="attach_layer_ops"):
        router.compose(torch.randn(1, 4, 4), word_space=None)


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


def test_language_layer_generate_emits_rules_dict_after_compose():
    router = _make_signal_router()
    router.attach_layer_ops(ops=[_Stub()], rule_ids=[3], tier="S")
    ws = _StubWordSpace()
    target = torch.randn(2, 4, 4)
    router.compose(target, word_space=ws)
    g = router.generate(target, word_space=ws)
    assert isinstance(g, dict)
    assert "S" in g
    rows = g["S"]
    assert len(rows) == 2
    for row in rows:
        assert isinstance(row, list)
        for rid in row:
            assert rid == 3


def test_language_layer_compose_populates_current_rules():
    router = _make_signal_router()
    router.attach_layer_ops(ops=[_Stub()], rule_ids=[7], tier="S")
    ws = _StubWordSpace()
    rules = router.compose(torch.randn(2, 4, 4), word_space=ws)
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
