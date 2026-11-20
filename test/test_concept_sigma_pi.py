"""Concept parts compose in presence charts; one inventory, two COO passes."""
import pytest
import torch
import Spaces
from symbol_fixtures import readable
from test_cs_sparse_weights import _cs, _mint_row, _evidence, _mint_field, _field_forward


def test_union_holds_one_present_part_across_orders():
    cs = _cs(nS=128, order=4)
    a0 = torch.zeros(cs._order_caps()[0], 1, 1, 2)
    a0[0, :, :, 0] = 1.
    source = 0
    rows = []
    for order in range(1, 5):
        row = _mint_row(cs, order, 100 + order)
        start, end = cs.order_slice(order - 1)
        readable(cs, *range(start, end))
        cs.add_concept_edge(row, source, weight=1.)
        for missing in range(start, end):
            if missing != source:
                cs.add_concept_edge(row, missing, weight=1.)
        source = row
        rows.append(row)
    _, a = cs.cs_forward_content(a0, torch.randn(128, 8))
    torch.testing.assert_close(a[rows, :, 0, 0], torch.ones(4, 1), atol=2e-6, rtol=0)




def test_unwritten_alternative_receives_credit_from_present_witness():
    cs = _cs()
    r = _mint_row(cs, 1, 101)
    readable(cs, 0)
    cs.add_concept_edge(r, 0)
    ly = Spaces._concept_alloc_of(cs).layer()
    assert ly.values.item() == 0.
    a0 = _evidence(torch.full((32, 1), .25), torch.full((32, 1), .75))
    _, a = cs.cs_forward_content(a0, torch.randn(64, 8))
    a[r].sum().backward()
    assert torch.isfinite(ly.values.grad).all()
    assert ly.values.grad.abs().sum() > 0
    assert a[r].count_nonzero() == 0




def test_weak_disjuncts_do_not_accumulate():
    cs = _cs(nS=128, order=2)
    r = _mint_row(cs, 1, 101)
    readable(cs, *range(32))
    for part in range(32):
        cs.add_concept_edge(r, part, weight=1.)
    a0 = _evidence(torch.full((cs._order_caps()[0], 1), .001))
    _, a = cs.cs_forward_content(a0, torch.randn(128, 8))
    presence = float(a[r, 0, 0, 0])
    assert presence == pytest.approx(.001, abs=2e-7)




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







