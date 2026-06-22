"""NeuralToolUser live cutover: the executor wired into LanguageLayer.compose
behind the <neuralToolUser> flag (doc/plans/NeuralToolUser.md).

Covers the adapter against a REAL BinaryStructuredReductionLayer (reuses the
layer's _stacked_reduced / chooser / _selected_reduced / compact_hard), and
the compose branch: flag on uses the hard-parse executor and populates the
route store; flag off (default) is the unchanged soft-DP fold loop.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch
import torch.nn as nn

from Language import (
    BinaryStructuredReductionLayer, NeuralToolUser, LanguageLayer,
    _BinaryStepperAdapter,
)


class _AddOp(nn.Module):
    def forward(self, left, right):
        return left + right


class _MulOp(nn.Module):
    def forward(self, left, right):
        return left * right


def test_adapter_bridges_real_binary_layer():
    # The adapter must drive the executor using the REAL layer internals.
    B, N, D = 1, 6, 4
    layer = BinaryStructuredReductionLayer(
        d_model=D, ops=[_AddOp(), _MulOp()], r_copy=1)
    adapter = _BinaryStepperAdapter(layer)
    x = torch.randn(B, N, D)
    cs, rs = adapter.score(x)
    assert cs.shape == (B, N, 1)
    assert rs.shape == (B, N - 1, 2)
    nxt = adapter.apply(x, *(_viterbi_masks(cs, rs)))
    assert nxt.shape[0] == B and 1 <= nxt.shape[1] <= N      # folded, not grown


def _viterbi_masks(cs, rs):
    from Language import binary_tiling_viterbi
    r = binary_tiling_viterbi(cs, rs)
    return r["copy_mask"], r["reduce_mask"]


def test_executor_folds_a_real_layer():
    B, N, D = 1, 6, 4
    layer = BinaryStructuredReductionLayer(
        d_model=D, ops=[_AddOp(), _MulOp()], r_copy=1)
    ntu = NeuralToolUser(max_levels=16)
    x = torch.randn(B, N, D)
    final_x, stats = ntu.parse_greedy(x, NeuralToolUser.binary_layer_stepper(layer))
    assert 1 <= final_x.shape[1] <= N
    assert stats.step_count >= 1
    assert len(stats.route) == stats.step_count


def _router_with_ops(neural_tool_user):
    router = LanguageLayer(
        n_input=4, n_output=4, hidden_dim=16, feature_dim=4,
        max_depth=3, temperature=1.0)
    router.attach_layer_ops(ops=[_AddOp(), _MulOp()], rule_ids=[1, 2], space_role="SS")
    router.neural_tool_user = neural_tool_user
    return router


def test_compose_flag_on_populates_route_store():
    router = _router_with_ops(neural_tool_user=True)
    x = torch.randn(2, 5, 4)
    router.compose(x, word_space=None)
    # The route store has an entry for the reduction space_role.
    assert len(router._ntu_route) >= 1
    space_role, route = next(iter(router._ntu_route.items()))
    assert len(route) >= 1
    assert "reduce_mask" in route[0] and "dist_probs" in route[0]
    # The matching space_role routing is flagged neural_tool_user.
    assert router._last_space_role_routings[space_role].get("neural_tool_user") is True


def test_real_layer_chooser_receives_policy_gradient():
    # The learning signal: a route's cross-product log-prob (what the
    # two-pass advantage multiplies) must backprop to the REAL layer's
    # chooser params -- here the anchor-dot anchors. This is what makes the
    # chooser trainable end-to-end (otherwise the policy never learns).
    from Language import (
        cross_product_action_dist, cross_product_route_logprob,
        binary_tiling_viterbi,
    )
    B, N, D = 1, 6, 4
    layer = BinaryStructuredReductionLayer(
        d_model=D, ops=[_AddOp(), _MulOp()], r_copy=1)
    adapter = _BinaryStepperAdapter(layer)
    x = torch.randn(B, N, D)
    cs, rs = adapter.score(x)                    # live scores from the chooser
    route = binary_tiling_viterbi(cs, rs)
    dist = cross_product_action_dist(cs, rs)
    logp = cross_product_route_logprob(
        dist, route["copy_mask"], route["reduce_mask"])
    logp.sum().backward()
    assert layer.reduce_anchor.grad is not None
    assert layer.reduce_anchor.grad.abs().sum() > 0


def test_real_layer_mlp_chooser_receives_policy_gradient():
    # Same, with the MLP chooser: the gradient reaches its owned params.
    from Language import (
        cross_product_action_dist, cross_product_route_logprob,
        binary_tiling_viterbi,
    )
    B, N, D = 1, 6, 4
    layer = BinaryStructuredReductionLayer(
        d_model=D, ops=[_AddOp(), _MulOp()], r_copy=1, chooser="mlp")
    adapter = _BinaryStepperAdapter(layer)
    x = torch.randn(B, N, D)
    cs, rs = adapter.score(x)
    route = binary_tiling_viterbi(cs, rs)
    dist = cross_product_action_dist(cs, rs)
    logp = cross_product_route_logprob(
        dist, route["copy_mask"], route["reduce_mask"])
    logp.sum().backward()
    assert layer.chooser.tool_embedding.grad is not None
    assert any(p.grad is not None for p in layer.chooser.mlp.parameters())


def test_compose_flag_off_uses_soft_loop_and_empty_store():
    router = _router_with_ops(neural_tool_user=False)
    x = torch.randn(2, 5, 4)
    router.compose(x, word_space=None)
    assert router._ntu_route == {}                          # executor not run
    # The soft-DP path records binary_rounds, not the NTU flag.
    tr = next(iter(router._last_space_role_routings.values()))
    assert "binary_rounds" in tr
    assert not tr.get("neural_tool_user", False)
