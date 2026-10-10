"""End-to-end acceptance: stack NOT (unary) in front of AND/OR (binary)
inside one LanguageLayer and confirm the dispatch produces sensible
per-space_role rule selections plus full-graph gradient flow.

The ops here are minimal float-tensor proxies for AND / OR / NOT;
plugging in real GRAMMAR_LAYER_CLASSES instances is a follow-up plan
(see Task 13 open questions).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch
import torch.nn as nn

import Language
from Language import LanguageLayer


def _make_router():
    return LanguageLayer(
        n_input=4, n_output=4, hidden_dim=16, feature_dim=4,
        max_depth=3, temperature=1.0,
    )


class _StubSymbolSpace:
    def __init__(self):
        self.current_rules = {}
        self.generate_rules = {}
        self._compose_generation = 0
    def host_layer(self, space_role, rule_name):
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


def test_xor_router_emits_per_space_role_rule_dict():
    router = _make_router()
    # Single space_role "SS": unary NOT (rule_id 0) and binary AND/OR (rule_ids 1, 2).
    router.attach_unary_ops(ops=[_NotOp()], rule_ids=[0], space_role="SS")
    router.attach_layer_ops(ops=[_AndOp(), _OrOp()], rule_ids=[1, 2], space_role="SS")
    ss = _StubSymbolSpace()
    rules = router.compose(torch.randn(2, 4, 4), word_space=ss)
    # One key per space_role; unary + binary rule_ids merged in route order.
    assert list(rules.keys()) == ["SS"]
    for row in rules["SS"]:
        for rid in row:
            assert rid in (0, 1, 2), f"unexpected rule_id {rid}"


def test_xor_router_gradients_reach_all_three_ops(monkeypatch):
    router = _make_router()

    class _ParamApply(nn.Module):
        def __init__(self, D, op, arity):
            super().__init__()
            self.proj = nn.Linear(D, D, bias=False)
            nn.init.eye_(self.proj.weight)
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
    router.attach_unary_ops(ops=[pnot], rule_ids=[0], space_role="SS")
    router.attach_layer_ops(ops=[pand, por], rule_ids=[1, 2], space_role="SS")

    ss = _StubSymbolSpace()
    # Exercise NOT -> AND -> OR -> AND -> STOP through the real dispatcher.
    # Fixed, small operands keep OR away from its clamp and every product
    # nonzero. This is an operator-gradient fixture, not a learned routing
    # or convergence assertion; no random initialization decides coverage.
    select = router.operation_layer.select_logits
    actions = iter((6, 0, 1, 0))  # N=4: six binary then four unary actions.
    def select_each_op(logits, **kwargs):
        action = next(actions, logits.shape[-1] - 1)
        kwargs['replay_action'] = torch.full((logits.shape[0],), action,
                                           device=logits.device, dtype=torch.long)
        return select(logits, **kwargs)
    monkeypatch.setattr(router.operation_layer, 'select_logits', select_each_op)
    x = torch.linspace(.05, .2, 16).reshape(1, 4, D).requires_grad_()
    rules = router.compose(x, word_space=ss)
    assert rules['SS'] == [[0, 1, 2, 1]]
    loss = router._last_output.square().sum()
    loss.backward()
    for name, p in [("not", pnot.proj), ("and", pand.proj), ("or", por.proj)]:
        assert p.weight.grad is not None and p.weight.grad.abs().sum() > 0, \
            f"no gradient reached the {name} op"
