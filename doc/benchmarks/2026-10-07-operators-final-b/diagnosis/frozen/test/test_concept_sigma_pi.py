"""Concept parts compose in presence charts; one inventory, two COO passes."""
import pytest
import torch
import Spaces
from test_cs_sparse_weights import _cs, _mint_row, _evidence, _mint_field, _field_forward


def test_union_holds_one_present_part_across_orders():
    cs = _cs(nS=128, order=4)
    a0 = torch.zeros(cs._order_caps()[0], 1, 1, 2)
    a0[0, :, :, 0] = 1.
    source = 0
    rows = []
    for order in range(1, 5):
        row = _mint_row(cs, order, 100 + order)
        cs.add_concept_edge(row, source, weight=1.)
        start, end = cs.order_slice(order - 1)
        for missing in range(start, end):
            if missing != source:
                cs.add_concept_edge(row, missing, weight=1.)
        source = row
        rows.append(row)
    _, a = cs.cs_forward_content(a0, torch.randn(128, 8))
    torch.testing.assert_close(a[rows, :, 0, 0], torch.ones(4, 1), atol=2e-6, rtol=0)


def test_sigma_reads_same_order_conjunction_and_lower_order_parts():
    cs = _cs()
    cs.conceptual_pi = True
    p = _mint_field(cs, 101)
    u = _mint_row(cs, 1, 102)
    cs.add_concept_edge(p, 0, weight=1., conjunctive=True)
    cs.add_concept_edge(p, 1, weight=1., conjunctive=True, negated=True)
    cs.add_concept_edge(u, p, weight=1.)
    cs.add_concept_edge(u, 2, weight=1.)
    positive = torch.zeros(32, 4)
    positive[:3] = torch.tensor([[1., 1., 0., 0.], [0., 1., 0., 1.], [0., 0., 1., 0.]])
    a0 = _evidence(positive, 1 - positive)
    _, a = _field_forward(cs, a0, torch.randn(64, 8))
    torch.testing.assert_close(a[p, :, 0, 0], torch.tensor([1., 1., 1., 0.]), atol=3e-6, rtol=0)
    torch.testing.assert_close(a[u, :, 0, 0], torch.tensor([1., 1., 1., 0.]), atol=3e-6, rtol=0)


def test_unwritten_alternative_receives_credit_from_present_witness():
    cs = _cs()
    r = _mint_row(cs, 1, 101)
    cs.add_concept_edge(r, 0)
    ly = Spaces._concept_alloc_of(cs).layer()
    assert ly.values.item() == 0.
    a0 = _evidence(torch.full((32, 1), .25), torch.full((32, 1), .75))
    _, a = cs.cs_forward_content(a0, torch.randn(64, 8))
    a[r].sum().backward()
    assert torch.isfinite(ly.values.grad).all()
    assert ly.values.grad.abs().sum() > 0
    assert a[r].count_nonzero() == 0


def test_attribution_selects_a_case_and_demands_all_conjunctive_parts():
    cs = _cs()
    cs.conceptual_pi = True
    p = _mint_field(cs, 101)
    u = _mint_row(cs, 1, 102)
    for col in (0, 1):
        cs.add_concept_edge(p, col, weight=1., conjunctive=True)
        cs.add_concept_edge(u, col + 2, weight=1.)
    y = torch.zeros(sum(cs._order_caps()), 1, 1, 2)
    y[p, :, :, 0] = .81
    y[u, :, :, 0] = .84
    restored = cs.cs_reverse_presence(y)
    torch.testing.assert_close(restored[:4, 0, 0, 0], torch.tensor([.81, .81, .84, 0.]), atol=1e-6, rtol=0)


def test_weak_disjuncts_do_not_accumulate():
    cs = _cs(nS=128, order=2)
    r = _mint_row(cs, 1, 101)
    for part in range(32):
        cs.add_concept_edge(r, part, weight=1.)
    a0 = _evidence(torch.full((cs._order_caps()[0], 1), .001))
    _, a = cs.cs_forward_content(a0, torch.randn(128, 8))
    presence = float(a[r, 0, 0, 0])
    assert presence == pytest.approx(.001, abs=2e-7)


def test_negated_disjunct_and_mixed_parts_on_one_concept():
    cs = _cs()
    cs.conceptual_pi = True
    row = _mint_field(cs, 101)
    cs.add_concept_edge(row, 0, weight=1., conjunctive=True, negated=True)
    cs.add_concept_edge(row, 1, weight=1., conjunctive=True)
    cs.add_concept_edge(row, 2, weight=1., negated=True)
    a0 = _evidence(torch.full((32, 1), .25), torch.full((32, 1), .75))
    _, a = _field_forward(cs, a0, torch.randn(64, 8))
    expected = max(min(.75, .25), .75)
    assert float(a[row, 0, 0, 0]) == pytest.approx(expected, abs=1e-6)


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


def test_unwritten_conjunction_has_no_invented_directional_credit():
    cs = _cs()
    cs.conceptual_pi = True
    row = _mint_field(cs, 101)
    cs.add_concept_edge(row, 0, conjunctive=True)
    _, a = _field_forward(cs, _evidence(torch.full((32, 1), .5)), torch.zeros(64, 8))
    if a[row].requires_grad:
        a[row].sum().backward()
    grad = Spaces._concept_alloc_of(cs).layer().conjunctive.values.grad
    assert grad is None or (torch.isfinite(grad).all() and float(grad.abs().sum()) == 0)


def test_frozen_conjunctive_edge_stays_frozen_across_pending_growth():
    cs = _cs()
    cs.conceptual_pi = True
    row = _mint_field(cs, 101)
    cs.add_concept_edge(row, 0, .5, conjunctive=True)
    cs.freeze_concept(101)
    _, a = _field_forward(cs, _evidence(torch.full((32, 1), .5)), torch.ones(64, 8))
    other = _mint_field(cs, 102)
    cs.add_concept_edge(other, 1, .5, conjunctive=True)
    a[row].sum().backward()
    ly = Spaces._concept_alloc_of(cs).layer().conjunctive
    assert ly.values.grad is None or not ly.values.grad.any()



def test_pi_edges_are_rejected_above_the_field():
    cs = _cs()
    row = _mint_row(cs, 1, 201)
    with pytest.raises(ValueError, match='order-0'):
        cs.add_concept_edge(row, 0, 1., conjunctive=True)
    assert Spaces._concept_alloc_of(cs).layer().conjunctive.nnz == 0
