"""Concept parts compose in presence charts; one inventory, two COO passes."""
import pytest
import torch
import Spaces
from test_cs_sparse_weights import _cs, _mint_row


def test_union_holds_one_present_part_across_orders():
    cs = _cs(nS=128, order=4)
    a0 = torch.full((cs._order_caps()[0], 1), -1.)
    a0[0] = 1.
    source = 0
    rows = []
    for order in range(1, 5):
        row = _mint_row(cs, order, 100 + order)
        cs.add_concept_edge(row, source, weight=1.)
        for missing in range(1, 8):
            cs.add_concept_edge(row, missing, weight=1.)
        source = row
        rows.append(row)
    _, a = cs.cs_forward_content(a0, torch.randn(128, 8))
    torch.testing.assert_close(a[rows], torch.ones(4, 1), atol=2e-6, rtol=0)


def test_sigma_reads_same_order_conjunction_and_lower_order_parts():
    cs = _cs()
    cs.conceptual_pi = True
    p = _mint_row(cs, 1, 101)
    u = _mint_row(cs, 1, 102)
    cs.add_concept_edge(p, 0, weight=1., conjunctive=True)
    cs.add_concept_edge(p, 1, weight=-1., conjunctive=True)
    cs.add_concept_edge(u, p, weight=1.)
    cs.add_concept_edge(u, 2, weight=1.)
    a0 = torch.full((32, 4), -1.)
    a0[:3] = torch.tensor([[1., 1., -1., -1.], [-1., 1., -1., 1.], [-1., -1., 1., -1.]])
    _, a = cs.cs_forward_content(a0, torch.randn(64, 8))
    torch.testing.assert_close(a[p], torch.tensor([1., -1., -1., -1.]), atol=3e-6, rtol=0)
    torch.testing.assert_close(a[u], torch.tensor([1., -1., 1., -1.]), atol=3e-6, rtol=0)


def test_unspecified_exponents_start_at_zero_and_can_learn():
    cs = _cs()
    r = _mint_row(cs, 1, 101)
    cs.add_concept_edge(r, 0)
    ly = Spaces._concept_alloc_of(cs).layer()
    assert ly.values.item() == 0.
    ly.conjunctive.add_edge(r, 1)
    assert ly.conjunctive.values.item() == 0.
    a0 = torch.full((32, 1), -.5)
    _, a = cs.cs_forward_content(a0, torch.randn(64, 8))
    a[r].sum().backward()
    assert torch.isfinite(ly.values.grad).all()
    assert ly.values.grad.abs().sum() > 0


def test_transpose_splits_union_and_implies_conjunctive_parts():
    cs = _cs()
    cs.conceptual_pi = True
    p = _mint_row(cs, 1, 101)
    u = _mint_row(cs, 1, 102)
    for col in (0, 1):
        cs.add_concept_edge(p, col, weight=1., conjunctive=True)
        cs.add_concept_edge(u, col + 2, weight=1.)
    y = torch.zeros(64, 1)
    y[p] = .81
    y[u] = .84
    restored = cs.cs_reverse_presence(y)
    torch.testing.assert_close(restored[:4, 0], torch.tensor([.9, .9, .6, .6]), atol=1e-6, rtol=0)


def test_weak_disjuncts_have_declared_accumulation_bound():
    cs = _cs(nS=128, order=2)
    r = _mint_row(cs, 1, 101)
    for part in range(32):
        cs.add_concept_edge(r, part, weight=1.)
    a0 = torch.full((cs._order_caps()[0], 1), -.998)
    _, a = cs.cs_forward_content(a0, torch.randn(128, 8))
    presence = float((a[r, 0] + 1) / 2)
    assert presence == pytest.approx(1 - .999 ** 32, abs=2e-6)
    assert presence - .001 < .031  # K=32; union bound (K-1)*epsilon


def test_negated_disjunct_and_mixed_parts_on_one_concept():
    cs = _cs()
    cs.conceptual_pi = True
    row = _mint_row(cs, 1, 101)
    cs.add_concept_edge(row, 0, weight=-1., conjunctive=True)
    cs.add_concept_edge(row, 1, weight=1., conjunctive=True)
    cs.add_concept_edge(row, 2, weight=-1.)
    a0 = torch.full((32, 1), -.5)
    _, a = cs.cs_forward_content(a0, torch.randn(64, 8))
    expected = 1 - (1 - .75 * .25) * (1 - .75)
    assert float(a[row]) == pytest.approx(2 * expected - 1, abs=1e-6)


def test_sparse_growth_preserves_adam_and_pending_credit():
    from Layers import SparseLayer
    layer = SparseLayer(4, 3, nonlinear=False)
    layer.add_edge(2, 0, .4)
    optimizer = torch.optim.Adam([layer.values], lr=.01)
    layer._optimizer = optimizer
    layer(torch.ones(4, 1)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    before = optimizer.state[layer.values]['exp_avg'].clone()
    pending = layer.fold_presence(torch.full((4, 1), .25)).sum()
    layer.add_edge(2, 1, .2)
    torch.testing.assert_close(optimizer.state[layer.values]['exp_avg'][:1], before)
    assert optimizer.state[layer.values]['exp_avg'][1] == 0
    pending.backward()
    assert layer.values.grad[0] != 0
    assert layer.values.grad[1] == 0
    optimizer.step()
    layer.remove_edges([(2, 0)])
    assert optimizer.param_groups[0]['params'][0] is layer.values
    assert optimizer.state[layer.values]['exp_avg'].shape == (1,)


def test_recycled_edge_cannot_receive_pending_gradient():
    from Layers import SparseLayer
    layer = SparseLayer(3, 2, nonlinear=False)
    layer.add_edge(1, 0, .5)
    loss = layer.fold_presence(torch.full((3, 1), .5)).sum()
    layer.remove_edges([(1, 0)])
    layer.add_edge(1, 0, .5)
    loss.backward()
    assert layer.values.grad is None


@pytest.mark.slow
@pytest.mark.parametrize('width', [4, 8])
def test_unseeded_xor_learns_through_concept_pyramid(width):
    """900 updates, one unseeded run; a null remains a failing learning gate."""
    cs = _cs(nS=128, order=1)
    cs.conceptual_pi = True
    conjunctions = [_mint_row(cs, 1, 100 + i) for i in range(width)]
    union = _mint_row(cs, 1, 200)
    for row in conjunctions:
        cs.add_concept_edge(row, 0, conjunctive=True)
        cs.add_concept_edge(row, 1, conjunctive=True)
        cs.add_concept_edge(union, row)
    ly = Spaces._concept_alloc_of(cs).layer()
    assert torch.count_nonzero(ly.values) == 0
    assert torch.count_nonzero(ly.conjunctive.values) == 0
    optimizer = torch.optim.Adam([ly.values, ly.conjunctive.values], lr=.03)
    a0 = torch.full((cs._order_caps()[0], 4), -1.)
    a0[:2] = torch.tensor([[-.9, -.9, .9, .9], [-.9, .9, -.9, .9]])
    target = torch.tensor([0., 1., 1., 0.])
    dictionary = torch.randn(128, 8)
    for _ in range(900):
        optimizer.zero_grad()
        _, a = cs.cs_forward_content(a0, dictionary)
        loss = torch.nn.functional.mse_loss((a[union] + 1) / 2, target)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        _, a = cs.cs_forward_content(a0, dictionary)
        output = (a[union] + 1) / 2
        mse = float(torch.nn.functional.mse_loss(output, target))
    print({'conjunctions': width, 'updates': 900, 'mse': mse, 'outputs': output.tolist()})
    assert mse < .1, f'XOR learning null with {width} conjunctions: MSE={mse:.9f}'


def test_zero_conjunctive_exponent_receives_finite_directional_credit():
    cs = _cs()
    cs.conceptual_pi = True
    row = _mint_row(cs, 1, 101)
    cs.add_concept_edge(row, 0, conjunctive=True)
    _, a = cs.cs_forward_content(torch.full((32, 1), .5), torch.zeros(64, 8))
    a[row].sum().backward()
    grad = Spaces._concept_alloc_of(cs).layer().conjunctive.values.grad
    assert torch.isfinite(grad).all() and float(grad.abs().sum()) > 0


def test_frozen_conjunctive_edge_stays_frozen_across_pending_growth():
    cs = _cs()
    cs.conceptual_pi = True
    row = _mint_row(cs, 1, 101)
    cs.add_concept_edge(row, 0, .5, conjunctive=True)
    cs.freeze_concept(101)
    _, a = cs.cs_forward_content(torch.full((32, 1), .5), torch.ones(64, 8))
    other = _mint_row(cs, 1, 102)
    cs.add_concept_edge(other, 1, .5, conjunctive=True)
    a[row].sum().backward()
    ly = Spaces._concept_alloc_of(cs).layer().conjunctive
    assert ly.values.grad is None or not ly.values.grad.any()


def test_four_conjunctions_represent_xor_with_declared_signed_parts():
    """Representation check only: the learning gates start every exponent at zero."""
    cs = _cs()
    cs.conceptual_pi = True
    rows = [_mint_row(cs, 1, 100 + i) for i in range(4)]
    union = _mint_row(cs, 1, 200)
    for i, row in enumerate(rows):
        cs.add_concept_edge(row, 0, weight=1. if i % 2 else -1., conjunctive=True)
        cs.add_concept_edge(row, 1, weight=-1. if i % 2 else 1., conjunctive=True)
        cs.add_concept_edge(union, row, weight=1.)
    a0 = torch.full((32, 4), -1.)
    a0[:2] = torch.tensor([[-1., -1., 1., 1.], [-1., 1., -1., 1.]])
    _, a = cs.cs_forward_content(a0, torch.zeros(64, 8))
    torch.testing.assert_close(a[union], torch.tensor([-1., 1., 1., -1.]), atol=3e-6, rtol=0)
