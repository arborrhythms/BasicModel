import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch

from SignalRouter import compact_hard, compact_soft, binary_tiling_viterbi


def _trivial_op(left, right):
    return left + right


def test_compact_hard_lengths_match_viterbi_route():
    B, N, D = 2, 6, 4
    torch.manual_seed(0)
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
    x = torch.randn(B, N, D)
    reduced = x[:, :-1] + x[:, 1:]
    p_copy = torch.full((B, N), 0.5, requires_grad=True)
    p_reduce = torch.full((B, N - 1), 0.25, requires_grad=True)
    y_soft = compact_soft(
        x=x, reduced=reduced,
        copy_marginal=p_copy, reduce_marginal=p_reduce,
    )
    y_soft.sum().backward()
    assert p_copy.grad is not None and p_copy.grad.abs().sum() > 0
    assert p_reduce.grad is not None and p_reduce.grad.abs().sum() > 0
