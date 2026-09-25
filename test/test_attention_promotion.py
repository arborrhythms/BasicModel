"""Provisional concept rows retain witnessed context and observed use.

Matching contexts assign disjunctive parts; conceptualPi admits complete
co-present sets as conjunctive parts. Detached use EWMA promotes the same
row in place and replenishes the per-order provisional pool.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import pytest
import torch

import Spaces
from test_cs_sparse_weights import _evidence
from test_basicmodel import _populate_test_config

_D = 8


def _cs(nS=64, order=3, promote=True):
    nP = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D,
        wordDim=_D, outputDim=_D,
        nInput=nP, nPercepts=nP, nConcepts=nS, nSymbols=nS,
        nWords=nS, nOutput=nS, nWhere=0, nWhen=0,
    )
    cs = Spaces.ConceptualSpace([nP, _D], [nS, _D], [nS, _D])
    object.__setattr__(cs, "_symbolic_order", order)
    object.__setattr__(cs, "_serial", False)
    if promote:
        object.__setattr__(cs, "_promotion_enabled", True)
    return cs


def _mint_order0(cs, n):
    """Mint ``n`` order-0 concepts with reserved snap rows; returns
    ``[(cid, row)]`` in allocation order (rows 0, 1, 2, ...)."""
    alloc = Spaces._concept_alloc_of(cs)
    out = []
    for _ in range(n):
        cid = alloc.new_concept()
        row = cs._csw_concept_row(0, cid)
        assert row is not None
        out.append((cid, int(row)))
    return out


def _observe(cs, active, B_rows=None):
    """Stage a synthetic admitted field (one batch row) on the promotion
    stashes and run the collector. ``active`` maps global row -> signed
    activation; the level-rows stash is the full snap block (order-0 rows
    are always staged by the pyramid)."""
    N = int(cs.nVectors)
    a = torch.zeros(sum(cs._order_caps()), 1, 1, 2)
    for r, v in active.items():
        a[int(r), 0, 0, 0] = float(v)
    snap = torch.arange(cs._order_caps()[0]).unsqueeze(-1)
    object.__setattr__(cs, "_promo_last_acts", a)
    object.__setattr__(cs, "_cs_level_rows", [snap])
    if cs.conceptual_pi:
        n0 = cs._order_caps()[0]
        cs._cs_position_evidence = a[:n0].unsqueeze(-2)
        cs._cs_position_spans = torch.tensor([[[0, 1]]])
        cs._cs_extents = torch.tensor([[[0, 1]]])
    cs.promotion_observe()



def _pool_rows(cs, *, assigned=True):
    ly = Spaces._concept_alloc_of(cs).layer()
    mask = ly.provisional & ly.assigned if assigned else ly.provisional
    return mask.nonzero().flatten().tolist()


def _fixture(pi=False):
    cs = _cs(nS=128, order=2)
    cs.conceptual_pi = pi
    cs.concept_pool_size = 2
    rows = _mint_order0(cs, 6)
    return cs, rows


def test_first_matching_context_assigns_pair_before_discovery():
    cs, rows = _fixture()
    (x, a), (y, b), (_, c), *_ = rows
    _observe(cs, {a: 1., c: 1.})
    assert not _pool_rows(cs)
    _observe(cs, {b: 1., c: 1.})
    ly = Spaces._concept_alloc_of(cs).layer()
    assigned = _pool_rows(cs)
    assert len(assigned) == 1
    r = assigned[0]
    assert dict(cs.concept_weights(r)) == {a: 1., b: 1.}
    assert ly.participation[r].item() == pytest.approx(.1)
    assert cs.concept_id_at_row(r) is None
    assert cs.promotion_pass() == []
    assert ly.where[r][c] > 0
    assert not hasattr(cs, '_promotion_cache_state')


def test_use_discovers_identity_in_place_and_replenishes_pool():
    cs, rows = _fixture()
    a, b, c = [r for _, r in rows[:3]]
    _observe(cs, {a: 1., c: 1.})
    _observe(cs, {b: 1., c: 1.})
    r = _pool_rows(cs)[0]
    cs.truth_criterion = 1.  # truth significance no longer controls discovery
    for i in range(20):
        _observe(cs, {a if i % 2 else b: 1., c: 1.})
    discovered = cs.promotion_pass()
    assert len(discovered) == 1
    assert cs.concept_id_at_row(r) == discovered[0]
    assert set(cs.concept_parts(discovered[0])) == {('sym', rows[0][0]), ('sym', rows[1][0])}
    start, end = cs.order_slice(1)
    ly = Spaces._concept_alloc_of(cs).layer()
    assert int(ly.provisional[start:end].sum()) == cs.concept_pool_size
    assert not ly.participation.requires_grad


def test_matching_alternative_joins_existing_kind_without_context_as_part():
    cs, rows = _fixture()
    a, b, c, d = [r for _, r in rows[:4]]
    _observe(cs, {a: 1., c: 1.})
    _observe(cs, {b: 1., c: 1.})
    r = _pool_rows(cs)[0]
    _observe(cs, {d: 1., c: 1.})
    assert set(dict(cs.concept_weights(r))) == {a, b, d}
    assert c not in dict(cs.concept_weights(r))


def test_nonrecurring_row_decays_and_is_recycled():
    cs, rows = _fixture()
    cs.concept_pool_size = 1
    a, b, c, d, e, f = [r for _, r in rows]
    _observe(cs, {a: 1., c: 1.})
    _observe(cs, {b: 1., c: 1.})
    old = _pool_rows(cs)[0]
    for _ in range(5):
        _observe(cs, {d: 1., f: 1.})
    _observe(cs, {e: 1., f: 1.})
    assert _pool_rows(cs) == [old]
    assert set(dict(cs.concept_weights(old))) == {d, e}
    assert cs.concept_id_at_row(old) is None


def test_objects_never_acquire_witnessed_kinds():
    cs, _ = _fixture()
    word, obj, meta = cs.create_word_object_meta([1], 2, key='cat')
    ly = Spaces._concept_alloc_of(cs).layer()
    assert cs._csw_row_of(word) in cs._witnessed_rows()
    assert cs._csw_row_of(obj) not in cs._witnessed_rows()
    assert cs._csw_row_of(meta) not in cs._witnessed_rows()
    a, b = cs._csw_row_of(obj), cs._csw_row_of(word)
    _observe(cs, {a: 1., b: 1.})
    assert not bool(ly.witnessed[a])


def test_pool_parameter_controls_reservation_and_gate_off_is_inert():
    cs = _cs(promote=False)
    cs._ensure_concept_pool()
    assert not _pool_rows(cs, assigned=False)
    cs._promotion_enabled = True
    cs.concept_pool_size = 2
    cs._ensure_concept_pool()
    ly = Spaces._concept_alloc_of(cs).layer()
    for order in range(1, len(cs._order_caps())):
        start, end = cs.order_slice(order)
        assert int(ly.provisional[start:end].sum()) == 2


def test_nonfinite_acts_fail_loud():
    cs, _ = _fixture()
    with pytest.raises(RuntimeError, match='NaN/Inf'):
        _observe(cs, {0: float('nan')})


def test_provisional_parts_receive_gradient_through_use_gate():
    cs, rows = _fixture()
    a, b, c = [r for _, r in rows[:3]]
    _observe(cs, {a: .8, c: .8})
    _observe(cs, {b: .8, c: .8})
    r = _pool_rows(cs)[0]
    ly = Spaces._concept_alloc_of(cs).layer()
    a0 = torch.full((cs._order_caps()[0], 1), .5, requires_grad=True)
    _, acts = cs.cs_forward_content(_evidence(a0), torch.randn(128, 8))
    acts[r].sum().backward()
    assert ly.values.grad.abs().sum() > 0
    assert ly.participation.grad is None


def test_pool_checkpoint_preserves_both_parts_where_use_and_next_assignment():
    cs, rows = _fixture(pi=True)
    _observe(cs, {r: 1. for _, r in rows[:3]})
    from test_structural_checkpoint import _model_with
    from types import SimpleNamespace
    model = _model_with(cs, SimpleNamespace())
    saved = model._collect_structural_extras()
    restored, _ = _fixture(pi=True)
    target = _model_with(restored, SimpleNamespace())
    target._restore_structural_extras(saved)
    before = Spaces._concept_alloc_of(cs).layer()
    after = Spaces._concept_alloc_of(restored).layer()
    assert before._tensor_rows == after._tensor_rows
    torch.testing.assert_close(before.participation, after.participation)
    torch.testing.assert_close(before.conjunctive.values, after.conjunctive.values)
    torch.testing.assert_close(before.where, after.where)
    for space in (cs, restored):
        _observe(space, {r: 1. for _, r in rows[:3]})
    torch.testing.assert_close(before.participation, after.participation)


def test_registered_pool_checkpoint_has_one_sidecar_restore(tmp_path):
    from test_structural_checkpoint import _model_with
    from types import SimpleNamespace
    cs, rows = _fixture(pi=True)
    _observe(cs, {r: 1. for _, r in rows[:3]})
    model = _model_with(cs, SimpleNamespace())
    model.conceptualSpaces = torch.nn.ModuleList([cs])
    model.conceptualSpace = cs
    path = tmp_path / 'pool.pt'
    model.save_weights(path)
    restored, _ = _fixture(pi=True)
    target = _model_with(restored, SimpleNamespace())
    target.conceptualSpaces = torch.nn.ModuleList([restored])
    target.conceptualSpace = restored
    assert target.load_weights(path, require_match=True)
    before = Spaces._concept_alloc_of(cs).layer()
    after = Spaces._concept_alloc_of(restored).layer()
    torch.testing.assert_close(before.participation, after.participation)
    torch.testing.assert_close(before.conjunctive.values, after.conjunctive.values)


def test_row_pool_prunes_weak_edges_when_use_discovers_it():
    cs, rows = _fixture()
    a, b, c = [r for _, r in rows[:3]]
    _observe(cs, {a: 1., c: 1.})
    _observe(cs, {b: 1., c: 1.})
    r = _pool_rows(cs)[0]
    ly = Spaces._concept_alloc_of(cs).layer()
    cs.add_concept_edge(r, rows[3][1], weight=.0001)
    frozen = Spaces._concept_alloc_of(cs).new_concept()
    frozen_row = cs._csw_concept_row(1, frozen)
    cs.add_concept_edge(frozen_row, a, weight=.4)
    cs.freeze_concept(frozen)
    ly.participation[r] = .9
    assert cs.promotion_pass()
    assert rows[3][1] not in dict(cs.concept_weights(r))
    _, activation = cs.cs_forward_content(
        _evidence(torch.full((cs._order_caps()[0], 1), .25)), torch.zeros(128, _D))
    activation[frozen_row].sum().backward()
    assert ly.values.grad[ly._index[(frozen_row, a)]] == 0
