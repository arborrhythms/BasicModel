"""End-to-end acceptance: stack NOT (unary) in front of AND/OR (binary)
inside one SignalRouter and confirm the dispatch produces sensible
per-tier rule selections plus full-graph gradient flow.

The ops here are minimal float-tensor proxies for AND / OR / NOT;
plugging in real GRAMMAR_LAYER_CLASSES instances is a follow-up plan
(see Task 13 open questions).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch
import torch.nn as nn

import Language
from Language import Chart


class _StubWordSpace:
    def __init__(self):
        self.current_rules = {}
        self.generate_rules = {}
        self._compose_generation = 0
    def host_layer(self, tier, rule_name):
        return None


class _AndOp(nn.Module):
    """Multiplicative AND: matches the existing pi-style conjunction."""
    def forward(self, left, right):
        return left * right


class _OrOp(nn.Module):
    """Additive OR clipped to [-1,1]: matches the existing sigma-style
    disjunction shape."""
    def forward(self, left, right):
        return (left + right).clamp(min=-1.0, max=1.0)


class _NotOp(nn.Module):
    """Sign flip; XOR-fixture truths live in {-1, 1}."""
    def forward(self, x):
        return -x


def test_xor_router_emits_per_tier_rule_dict():
    chart = Chart(nInput=4, nOutput=4, max_depth=3, hidden_dim=16,
                  feature_dim=4, router_kind="signal")
    router = chart._ensure_signal_router()
    # Single tier "S": unary NOT (rule_id 0) and binary AND/OR (rule_ids 1, 2).
    router.attach_unary_ops(ops=[_NotOp()], rule_ids=[0], tier="S")
    router.attach_layer_ops(ops=[_AndOp(), _OrOp()], rule_ids=[1, 2], tier="S")
    ws = _StubWordSpace()
    rules = chart.compose(torch.randn(2, 4, 4), word_space=ws)
    # One key per tier; unary + binary rule_ids merged in route order.
    assert list(rules.keys()) == ["S"]
    for row in rules["S"]:
        for rid in row:
            assert rid in (0, 1, 2), f"unexpected rule_id {rid}"


def test_xor_router_gradients_reach_all_three_ops():
    chart = Chart(nInput=4, nOutput=4, max_depth=3, hidden_dim=16,
                  feature_dim=4, router_kind="signal")
    router = chart._ensure_signal_router()

    class _ParamApply(nn.Module):
        def __init__(self, D, op, arity):
            super().__init__()
            self.proj = nn.Linear(D, D, bias=False)
            self.op = op
            self.arity = arity
        def forward(self, *args):
            if self.arity == 1:
                return self.op(self.proj(args[0]))
            return self.op(self.proj(args[0]), self.proj(args[1]))

    D = 4
    pnot = _ParamApply(D, _NotOp(), 1)
    pand = _ParamApply(D, _AndOp(), 2)
    por = _ParamApply(D, _OrOp(), 2)
    router.attach_unary_ops(ops=[pnot], rule_ids=[0], tier="S")
    router.attach_layer_ops(ops=[pand, por], rule_ids=[1, 2], tier="S")

    ws = _StubWordSpace()
    x = torch.randn(2, 4, D, requires_grad=True)
    chart.compose(x, word_space=ws)
    # The unary op is exercised on the soft slab (mixture). Binary ops
    # show up in the marginal_slab path (which sums per-op reductions
    # via the soft DP marginals). Combine all three slabs into the loss.
    loss = (router._last_soft_slab.sum()
            + router._last_hard_slab.sum()
            + router._last_routing["marginal_slab"].sum())
    loss.backward()
    for name, p in [("not", pnot.proj), ("and", pand.proj), ("or", por.proj)]:
        assert p.weight.grad is not None and p.weight.grad.abs().sum() > 0, \
            f"no gradient reached the {name} op"
