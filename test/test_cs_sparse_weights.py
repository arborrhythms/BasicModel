"""Concept rows compose paired evidence over the bounded taper span S.

Each part matrix reads 2(S+1) sources: S positive poles and EVERYTHING,
then S negative poles and its absent counterpart. Codes are stored once.
"""
from basic_fixtures import feature as basic_feature
from order_fixtures import force_order
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import math
import torch
from definition_fixtures import with_definitions
import pytest

import Spaces
from symbol_fixtures import readable
from test_basicmodel import _populate_test_config

_D = 8


def _cs(nS=64, order=3, dim=_D, *, batch_size=1, order_limit=None, order_reservations=()):
    nP = 4
    _populate_test_config(
        inputDim=dim, perceptDim=dim, conceptDim=dim, symbolDim=dim,
        wordDim=dim, outputDim=dim,
        nInput=nP, nPercepts=nP, nConcepts=nS, nSymbols=nS,
        nWords=nS, nOutput=nS, nWhere=0, nWhen=0,
    )
    from util import TheXMLConfig
    TheXMLConfig.data['architecture']['training']['batchSize'] = batch_size
    cs = Spaces.ConceptualSpace([nP, dim], [nS, dim], [nS, dim],
        order_limit=max(1, order) if order_limit is None else order_limit,
        order_reservations=order_reservations)
    object.__setattr__(cs, "_symbolic_order", order)
    object.__setattr__(cs, "_serial", False)
    return with_definitions(cs)


def _evidence(positive, negative=None):
    """Fixture with explicit evidence; an omitted pole means unknown."""
    if negative is None:
        negative = torch.zeros_like(positive)
    return torch.stack((positive, negative), -1).unsqueeze(2)


def _mint_row(cs, order, cid):
    """First-seen allocated row of cid in its order block (dual-towers rev 2:
    only ALLOCATED rows enter a rung)."""
    force_order(cs, cid, order)
    return cs._csw_concept_row(order, cid)



def _mint_field(cs, cid):
    """Eight existing native predicates before a composed order-0 row."""
    for i in range(8):
        row = _mint_row(cs, 0, 10000 + i)
        basic_feature(cs, row, 'ps', 1000 + i, 1.)
    return _mint_row(cs, 0, cid)


def _field_forward(cs, evidence, dictionary=None):
    return cs.cs_forward_content(cs._compose_order0(evidence))

def test_order_slice_taper_contract():
    """The taper partitions the whole inventory; there is no overflow tail."""
    cs = _cs(nS=64, order=3)
    assert cs._order_caps() == (35, 17, 8, 4)
    assert cs.order_slice(0) == (0, 35)
    assert cs.order_slice(1) == (35, 52)
    assert cs.order_slice(2) == (52, 60)
    assert cs.order_slice(3) == (60, 64)
    with pytest.raises(Spaces.ConceptRowCapacityError, match='order 9'):
        cs.order_slice(9)
    cs2 = _cs(nS=16, order=1)
    assert cs2._order_caps() == (11, 5)
    assert cs2.order_slice(0) == (0, 11)
    assert cs2.order_slice(1) == (11, 16)


# -- the shared untyped square store (v3) --------------------------------------

def test_add_concept_edge_dedup_and_query():
    cs = _cs()                                       # order 0 is rows 0 to 34
    readable(cs, 5, 6)
    r1 = cs.add_concept_edge(35, 5, weight=1.5)
    r2 = cs.add_concept_edge(35, 5, weight=9.0)      # repeat -> no-op
    assert r1 == r2
    cs.add_concept_edge(35, 6, weight=0.5)
    got = cs.concept_weights(35)
    assert (5, 1.5) in got and (6, 0.5) in got
    assert cs.concept_weights(36) == []              # different row




def test_any_edge_on_snap_row_fails_loud():
    cs = _cs(nS=16, order=1)                         # n_snap = 8
    for row in (0, 7):
        try:
            cs.add_concept_edge(row, cs.order_slice(1)[0])
            assert False, "snap-row edges must raise (codebook rows, not edges)"
        except ValueError:
            pass
    s0 = cs._sparse_families(0)
    assert s0.nnz == 0


def test_family_values_grow_tail_preserving():
    cs = _cs()                                       # order 0 is rows 0 to 34
    readable(cs, 0, 1, 2)
    cs.add_concept_edge(35, 0)
    cs.add_concept_edge(35, 1)
    s = cs._sparse_families(0)
    with torch.no_grad():
        s.values.copy_(torch.tensor([3.0, 4.0]))
    cs.add_concept_edge(36, 2, weight=9.0)           # grow
    assert torch.allclose(s.values.detach(), torch.tensor([3.0, 4.0, 9.0]))


# -- source activation (still the snap's readout) + dictionary decoder --------

def test_source_code_activation_matches_dotproduct():
    cs = _cs()
    B, N, V = 2, 3, 5
    event = torch.randn(B, N, _D)
    W = torch.randn(V, _D)
    act = cs.source_code_activation(event, W, nonneg=False)
    assert act.shape == (V, B)
    ref = torch.einsum('bnd,vd->vb', event, W)        # sum over slots of <e,W>
    assert torch.allclose(act, ref, atol=1e-5)


def test_source_code_activation_is_nonneg_presence_by_default():
    cs = _cs()
    event = torch.randn(2, 3, _D)
    W = torch.randn(5, _D)
    act = cs.source_code_activation(event, W)          # default nonneg=True
    assert (act >= 0).all()                            # features are presences
    ref = torch.einsum('bnd,vd->vb', event, W).clamp(min=0.0)
    assert torch.allclose(act, ref, atol=1e-5)


def test_membership_read_is_input_dependent_and_differentiable():
    from test_concept_memberships import binary_features
    # The basic percepts have a property half, so a property and its Not
    # have different vectors and the concept's column does not cancel.
    from basic_fixtures import with_own_codes
    cs = with_own_codes(_cs(nS=16, order=1), (48, 49))
    # One byte: over [48, 49] each property is seen both ways and every row
    # reads the same on its two lanes, so the concept reads nothing.
    native, extents = binary_features(cs, torch.tensor([[48]]))
    primitive = native[2]
    with torch.no_grad():
        primitive.members[:, 48:50] = torch.tensor([[.2, .8], [.7, .3]])
    read = cs.cs_read_memberships(native, extents)
    assert read.shape == (cs._order_caps()[0], 1, 1, 2)
    # Fractional properties keep both lanes inside (0, 1) at the basic
    # percepts' rows that carry them: the part present, its property .2 for
    # and .8 against, the other .7 and .3.
    field, basic = cs._percept_field, cs.basic_percepts
    rows = field.values[[field.keys.index(('bp', basic.index[((48,), (p,))])) for p in (0, 1)], 0, 0]
    torch.testing.assert_close(rows.detach(), torch.tensor([[.2, .8], [.7, .3]]))
    # The concept reads its entries in the order-zero Sigma, and the basic
    # percepts' read of the property memberships takes its error: a
    # concept's error reaches which bytes belong to a property.
    read.sum().backward()
    sigma = cs.order_zero
    assert sigma.value.grad is not None and sigma.value.grad.abs().sum() > 0
    assert field.values.requires_grad and primitive.members.grad is not None
    # The read depends on its input, at the basic percepts and at the concept.
    values = field.values.detach().clone()
    with torch.no_grad():
        primitive.members[:, 48:50] *= .5
    changed = cs.cs_read_memberships(native, extents)
    assert not torch.equal(cs._percept_field.values.detach(), values)
    assert not torch.equal(changed.detach(), read.detach())


def test_membership_read_never_updates_the_dictionary():
    from test_concept_memberships import binary_features
    cs = _cs(nS=16, order=1)
    native, extents = binary_features(cs, torch.tensor([[48, 49]]))
    W = cs.similarity_codebook.getW()
    before = W.detach().clone()
    for training in (False, True):
        cs.train(training)
        cs.cs_read_memberships(native, extents)
        assert torch.equal(W, before)


def test_forward_content_shape_and_stacking():
    cs = _cs(nS=16, order=1)
    capacity = cs._order_caps()[0]
    a0 = _evidence(torch.rand(capacity, 2))
    row = _mint_row(cs, 1, 101)
    readable(cs, 0)
    cs.add_concept_edge(row, 0, 2.)
    a = cs.cs_forward_content(a0)
    assert a.shape == (cs.nVectors, 2, 1, 2)
    torch.testing.assert_close(a[:capacity], a0)
    # The store's Sigma: tanh of the weight times the symbol's value.
    torch.testing.assert_close(a[row, :, 0, 0], torch.tanh(2 * a0[0, :, 0, 0]))
    assert a[row + 1:].count_nonzero() == 0


def test_forward_content_literal_poles_and_standing_bias():
    cs = _cs(nS=16, order=1)
    a0 = _evidence(torch.zeros(8, 1))
    a0[3, 0, 0, 0] = .5
    a0[5, 0, 0, 1] = .25
    row = _mint_row(cs, 1, 201)
    readable(cs, 3, 5)
    cs.add_concept_edge(row, 3, 3.)
    cs.add_concept_edge(row, 5, .5, negated=True)
    a = cs.cs_forward_content(a0)
    # A sum of the entries, the Not entry reading row 5's not-value.
    assert float(a[row, 0, 0, 0]) == pytest.approx(math.tanh(3 * .5 + .5 * .25))
    cs.add_concept_edge(row, 16, .25)
    a = cs.cs_forward_content(a0)
    torch.testing.assert_close(a[row, 0, 0], torch.tensor([math.tanh(3 * .5 + .5 * .25 + .25), 0.]))


def test_forward_content_activation_is_bounded():
    cs = _cs(nS=16, order=2)
    r1, r2 = _mint_row(cs, 1, 101), _mint_row(cs, 2, 102)
    readable(cs, 0)
    cs.add_concept_edge(r1, 0, 3.)
    cs.add_concept_edge(r2, r1, 4.)
    a = cs.cs_forward_content(_evidence(torch.full((8, 2), .99)))
    assert a.min() >= 0 and a.max() <= 1
    assert a[r1, 0, 0, 0] > .9 and a[r2, 0, 0, 0] > .9


def test_forward_content_empty_store_passes_a0():
    cs = _cs(nS=16, order=1)
    a0 = _evidence(torch.rand(8, 2), torch.rand(8, 2))
    a = cs.cs_forward_content(a0)
    assert torch.equal(a[:8], a0)
    assert a[8:].count_nonzero() == 0


def test_level_stats_are_report_only():
    cs = _cs(nS=16, order=2)
    row = _mint_row(cs, 1, 101)
    readable(cs, 0)
    cs.add_concept_edge(row, 0, 1.)
    cs.cs_forward_content(_evidence(torch.rand(8, 2, requires_grad=True)))
    assert cs._cs_wave_qe is None
    assert len(cs._cs_level_acts) == 3
    assert all(type(v) is float for v in cs._cs_level_acts)
    rows = cs._cs_level_rows
    assert len(rows) == 2
    assert all(r.dtype == torch.long and not r.requires_grad for r in rows)
    assert rows[0].shape == (8, 2)
    assert rows[1].shape == (1, 2)
    assert int(rows[1][0, 0]) == row


def test_forward_content_kernels_agree():
    cs = _cs(nS=16, order=1)
    row = _mint_row(cs, 1, 101)
    readable(cs, 0, 1)
    cs.add_concept_edge(row, 0, .4, negated=True)
    cs.add_concept_edge(row, 1, .7)
    a0 = _evidence(torch.rand(8, 2), torch.rand(8, 2))
    a1 = cs.cs_forward_content(a0)
    Spaces._concept_alloc_of(cs).layer().kernel = 'spmm'
    a2 = cs.cs_forward_content(a0)
    torch.testing.assert_close(a1, a2)


def test_forward_content_reads_a_not_entry_and_holds_no_code():
    cs = _cs(nS=16, order=1)
    a0 = _evidence(torch.rand(8, 1), torch.rand(8, 1))
    row = _mint_row(cs, 1, 101)
    readable(cs, 0)
    cs.add_concept_edge(row, 0, 1., negated=True)
    a = cs.cs_forward_content(a0)
    assert a.shape == (cs.nVectors, 1, 1, 2)
    from Layers import signed_lanes
    # A Not entry reads the symbol's not-value: the concept's sign is the
    # symbol's exchanged.
    torch.testing.assert_close(a[row, 0, 0], signed_lanes(torch.tanh(a0[0, 0, 0, 1] - a0[0, 0, 0, 0])))


def test_forward_content_differentiable():
    cs = _cs(nS=16, order=2)
    r1, r2 = _mint_row(cs, 1, 101), _mint_row(cs, 2, 102)
    readable(cs, 0)
    cs.add_concept_edge(r1, 0, .5)
    cs.add_concept_edge(r2, r1, .5)
    positive = torch.rand(8, 2, requires_grad=True)
    field = cs.cs_forward_content(_evidence(positive))
    field.sum().backward()
    assert positive.grad is not None and positive.grad.abs().sum() > 0
    ly = Spaces._concept_alloc_of(cs).layer()
    assert ly.values.grad is not None and ly.values.grad.abs().sum() > 0
