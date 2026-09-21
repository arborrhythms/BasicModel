import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch
import pytest

from Language import compact_hard, compact_soft, binary_tiling_viterbi


def _trivial_op(left, right):
    return left + right


def test_compact_hard_lengths_match_viterbi_route():
    B, N, D = 2, 6, 4

    x = torch.randn(B, N, D)
    reduced = _trivial_op(x[:, :-1], x[:, 1:])
    copy_score = torch.randn(B, N, 1)
    reduce_score = torch.randn(B, N - 1, 1)
    route = binary_tiling_viterbi(copy_score, reduce_score)
    y, meta = compact_hard(
        x=x, reduced=reduced,
        copy_mask=route["copy_mask"], reduce_mask=route["reduce_mask"],
    )
    assert y.shape == (B, N, D)  # padded to input length
    for b in range(B):
        n_reduce = int(route["reduce_mask"][b].sum().item())
        n_copy = int(route["copy_mask"][b].sum().item())
        assert int(meta["lengths"][b].item()) == n_copy + n_reduce
        assert n_copy + 2 * n_reduce == N


def test_compact_hard_provenance_pointers():
    B, N, D = 1, 5, 3
    x = torch.arange(B * N * D).view(B, N, D).float()
    reduced = x[:, :-1] + x[:, 1:]
    # Force tiling: REDUCE(0,1), COPY(2), REDUCE(3,4) by hand-built masks.
    cm = torch.zeros(B, N, 1)
    cm[0, 2, 0] = 1.0
    rm = torch.zeros(B, N - 1, 1)
    rm[0, 0, 0] = 1.0
    rm[0, 3, 0] = 1.0
    y, meta = compact_hard(x=x, reduced=reduced,
                           copy_mask=cm, reduce_mask=rm)
    L = int(meta["lengths"][0].item())
    assert L == 3
    # Slot 0: REDUCE(0,1) -> src_left=0 src_right=1 action_kind=1
    assert int(meta["src_left"][0, 0].item()) == 0
    assert int(meta["src_right"][0, 0].item()) == 1
    assert int(meta["action_kind"][0, 0].item()) == 1
    # Slot 1: COPY(2) -> src_left=2 src_right=-1 action_kind=0
    assert int(meta["src_left"][0, 1].item()) == 2
    assert int(meta["src_right"][0, 1].item()) == -1
    assert int(meta["action_kind"][0, 1].item()) == 0
    # Slot 2: REDUCE(3,4) -> src_left=3 src_right=4 action_kind=1
    assert int(meta["src_left"][0, 2].item()) == 3
    assert int(meta["src_right"][0, 2].item()) == 4
    assert int(meta["action_kind"][0, 2].item()) == 1


def test_compact_hard_span_start_end_propagation():
    B, N, D = 1, 4, 2
    x = torch.randn(B, N, D)
    reduced = x[:, :-1] + x[:, 1:]
    cm = torch.zeros(B, N, 1)
    cm[0, 0, 0] = 1.0
    cm[0, 3, 0] = 1.0
    rm = torch.zeros(B, N - 1, 1)
    rm[0, 1, 0] = 1.0  # REDUCE(1,2)
    span_start = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    span_end   = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    _, meta = compact_hard(
        x=x, reduced=reduced, copy_mask=cm, reduce_mask=rm,
        span_start=span_start, span_end=span_end,
    )
    L = int(meta["lengths"][0].item())
    assert L == 3
    # Slot 0: COPY(0) -> [0,0]
    # Slot 1: REDUCE(1,2) -> [1,2]
    # Slot 2: COPY(3) -> [3,3]
    assert meta["span_start"][0, :L].tolist() == [0, 1, 3]
    assert meta["span_end"][0, :L].tolist() == [0, 2, 3]


def test_compact_soft_returns_length_N_slab():
    B, N, D = 2, 5, 4
    x = torch.randn(B, N, D)
    reduced = x[:, :-1] + x[:, 1:]
    p_copy = torch.full((B, N), 1.0)               # all-copy marginals
    p_reduce = torch.zeros(B, N - 1)
    y_soft = compact_soft(
        x=x, reduced=reduced,
        copy_marginal=p_copy, reduce_marginal=p_reduce,
    )
    assert y_soft.shape == (B, N, D)
    # All-copy means y_soft == x.
    assert torch.allclose(y_soft, x, atol=1e-5)


def test_compact_soft_single_reduction_blends_neighbours():
    B, N, D = 1, 4, 3
    x = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]],
                     dtype=torch.float32)
    reduced = x[:, :-1] + x[:, 1:]
    # Force REDUCE@1 with mass 1; copies elsewhere.
    p_copy = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    p_reduce = torch.tensor([[0.0, 1.0, 0.0]])
    y_soft = compact_soft(
        x=x, reduced=reduced,
        copy_marginal=p_copy, reduce_marginal=p_reduce,
    )
    # Position 0: copy x_0
    assert torch.allclose(y_soft[0, 0], x[0, 0])
    # Position 1: r_1 = x_1 + x_2
    assert torch.allclose(y_soft[0, 1], x[0, 1] + x[0, 2])
    # Position 2: shifted x_3 (post-reduction)
    assert torch.allclose(y_soft[0, 2], x[0, 3])
    # Position 3: pad (zero) since the sequence shrunk
    assert torch.allclose(y_soft[0, 3], torch.zeros(D))


def test_compact_soft_gradient_flows_through_marginals():
    B, N, D = 1, 4, 3
    x = torch.arange(1, B * N * D + 1, dtype=torch.float32).reshape(B, N, D)
    reduced = x[:, :-1] + x[:, 1:]
    p_copy = torch.full((B, N), 0.5, requires_grad=True)
    p_reduce = torch.full((B, N - 1), 0.25, requires_grad=True)
    y_soft = compact_soft(
        x=x, reduced=reduced,
        copy_marginal=p_copy, reduce_marginal=p_reduce,
    )
    # Every legal tiling preserves the sum under addition; its derivative
    # with respect to routing is zero. Squared packed values distinguish
    # the tilings and therefore provide actual routing credit.
    y_soft.square().sum().backward()
    assert p_copy.grad is not None and p_copy.grad.abs().sum() > 0
    assert p_reduce.grad is not None and p_reduce.grad.abs().sum() > 0


@pytest.mark.parametrize("n", range(1, 7))
def test_soft_compaction_agrees_with_every_committed_tiling(n):
    from test_signal_router_brute_force import enumerate_tilings

    x = torch.arange(1, n + 1, dtype=torch.float64).reshape(1, n, 1)
    reduced = x[:, :-1] * x[:, 1:]
    for tiling in enumerate_tilings(n, 1, 1):
        copy = torch.zeros(1, n, 1, dtype=x.dtype)
        reduce = torch.zeros(1, n - 1, 1, dtype=x.dtype)
        position = 0
        for kind, _ in tiling:
            if kind == "copy":
                copy[:, position] = 1
                position += 1
            else:
                reduce[:, position] = 1
                position += 2
        hard, _ = compact_hard(x=x, reduced=reduced, copy_mask=copy, reduce_mask=reduce)
        soft = compact_soft(x=x, reduced=reduced, copy_marginal=copy.squeeze(-1),
                            reduce_marginal=reduce.squeeze(-1))
        torch.testing.assert_close(soft, hard, rtol=0, atol=0, msg=str(tiling))


@pytest.mark.parametrize("compiled", [False, True])
def test_soft_compaction_matches_enumerated_outputs_and_derivatives(compiled):
    from Language import binary_tiling_soft_dp
    from test_signal_router_brute_force import enumerate_tilings

    n = 5
    x = torch.arange(1, n + 1, dtype=torch.float64).reshape(1, n, 1).requires_grad_()
    reduced = x[:, :-1] * x[:, 1:]
    copy_score = torch.linspace(-.7, .9, n, dtype=x.dtype).reshape(1, n, 1).requires_grad_()
    reduce_score = torch.linspace(.6, -.4, n - 1, dtype=x.dtype).reshape(1, n - 1, 1).requires_grad_()
    scores, outputs = [], []
    for tiling in enumerate_tilings(n, 1, 1):
        score = x.new_zeros(())
        output, position = [], 0
        for kind, _ in tiling:
            if kind == "copy":
                score = score + copy_score[0, position, 0]
                output.append(x[0, position])
                position += 1
            else:
                score = score + reduce_score[0, position, 0]
                output.append(reduced[0, position])
                position += 2
        outputs.append(torch.cat((torch.stack(output), x.new_zeros(n - len(output), 1))))
        scores.append(score)
    expected = (torch.stack(scores).softmax(0)[:, None, None] * torch.stack(outputs)).sum(0)[None]
    marginals = binary_tiling_soft_dp(copy_score, reduce_score)
    compact = torch.compile(compact_soft, backend="eager", fullgraph=True) if compiled else compact_soft
    actual = compact(x=x, reduced=reduced, copy_marginal=marginals["copy_marginal"],
                     reduce_marginal=marginals["reduce_marginal"])
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    operands = (x, copy_score, reduce_score)
    want = torch.autograd.grad(expected.square().sum(), operands, retain_graph=True)
    got = torch.autograd.grad(actual.square().sum(), operands)
    for measured, reference in zip(got, want):
        torch.testing.assert_close(measured, reference, rtol=1e-9, atol=1e-9)


def test_recursive_product_consumes_each_operand_once():
    from types import SimpleNamespace
    from Language import LanguageLayer, ProductLayer

    router = LanguageLayer(n_input=4, n_output=4, hidden_dim=8,
                           feature_dim=1, max_depth=3)
    router.attach_layer_ops(ops=[ProductLayer(nInput=1, nOutput=1)], rule_ids=[0])
    layer = next(iter(router._binary_layers.values()))
    with torch.no_grad():
        layer.copy_anchor.zero_()
        layer.reduce_anchor.fill_(1)
    operands = torch.tensor([2., 3., 5., 7.]).reshape(1, 4, 1).requires_grad_()
    router.compose(operands, SimpleNamespace(wholeSpace=None))
    torch.testing.assert_close(router._last_root_state.reshape(()), operands.prod())
    router._last_root_state.sum().backward()
    assert torch.isfinite(operands.grad).all()
    assert (operands.grad != 0).all()


def test_binary_reduction_excludes_padding_from_the_next_round():
    from Language import BinaryStructuredReductionLayer, ProductLayer

    layer = BinaryStructuredReductionLayer(d_model=1, ops=[ProductLayer()])
    with torch.no_grad():
        layer.copy_anchor.zero_()
        layer.reduce_anchor.fill_(1)
    x = torch.tensor([[[2.], [3.], [5.], [7.]], [[11.], [13.], [1e30], [1e30]]])
    hard, soft, route = layer(x, lengths=torch.tensor([4, 2]))
    torch.testing.assert_close(hard, torch.tensor([[[6.], [35.], [0.], [0.]],
                                                 [[143.], [0.], [0.], [0.]]]))
    torch.testing.assert_close(soft, hard)
    assert route["lengths"].tolist() == [2, 1]
