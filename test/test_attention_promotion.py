"""Familiar candidates share MTM's lifecycle; only kept closes mint them."""
from dataclasses import replace
import pytest
import torch
from definition_fixtures import with_definitions
from order_fixtures import force_order
import Spaces
from test_basicmodel import _populate_test_config
from FamiliarTier import centroid, observe_candidate
from MediumTermMemory import MediumTermMemory

_D = 8


def _cs(nS=64, order=3, promote=True):
    _populate_test_config(inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D,
        wordDim=_D, outputDim=_D, nInput=4, nPercepts=4, nConcepts=nS,
        nSymbols=nS, nWords=nS, nOutput=nS, nWhere=0, nWhen=0)
    cs = Spaces.ConceptualSpace([4, _D], [nS, _D], [nS, _D])
    cs._symbolic_order, cs._serial, cs._promotion_enabled = order, False, promote
    return with_definitions(cs)


def _mint_order0(cs, n):
    out = []
    for index in range(n):
        cid = Spaces._concept_alloc_of(cs).new_concept()
        force_order(cs, cid, 0)
        row = cs._csw_concept_row(0, cid)
        with torch.no_grad():
            cs.similarity_codebook.W[row].copy_(torch.eye(_D)[index % _D])
        out.append((cid, row))
    return out


def _fixture(pi=False):
    cs = _cs(nS=128, order=2)
    cs.conceptual_pi = pi
    return cs, _mint_order0(cs, 6)


def _close(cs, batch=0):
    cs.mtm.close(None, stream=batch,
        references=getattr(cs, '_field_activated_references', {}).get(batch, ()))
    cs._field_activated_references = {}


def _observe(cs, active, *, close=True):
    a = torch.zeros(sum(cs._order_caps()), 1, 1, 2)
    for row, pair in active.items():
        a[row, 0, 0] = torch.tensor(pair if isinstance(pair, tuple) else (pair, 0.))
    result = cs.observe_open_field(a)
    if close:
        _close(cs)
    return result


def _candidates(cs, *, order=None):
    return {key: entry for key, entry in cs.mtm.entries.items()
            if entry.order is not None and (order is None or entry.order == order)}


def test_open_matches_and_updates_centroid_with_bounded_support():
    pool = MediumTermMemory(4, n_vectors=8, field_width=2)
    key = pool.open_candidate(order=1, when=0, value=torch.tensor([1., 0.]), support={0: 2.})
    assert key == ('candidate', 0) and pool.match_candidate(1, torch.tensor([1., .1])) == key
    assert pool.match_candidate(2, torch.tensor([1., 0.])) is None
    pool.update_candidate(key, when=1, value=torch.tensor([0., 1.]), support={1: 2., 2: 1.})
    entry = pool.entries[key]
    assert entry.count == 2 and len(entry.support) == 2 and entry.when == 1
    torch.testing.assert_close(entry.value, torch.tensor([.4, .6]))
    assert entry.support == {0: 2., 1: 2.}


def test_capacity_and_access_horizon_are_shared_with_identities():
    pool = MediumTermMemory(2, horizon=1, n_vectors=8)
    origin = ('ltm', 'fixture', 1)
    pool.open(origin, kind=None, when=0, value=torch.ones(2))
    key = pool.open_candidate(order=0, when=1, value=torch.ones(2), support={0: 1.})
    pool.close(None, stream=0, references=(origin, key))
    pool.close(None, stream=0, references=(origin,))
    fresh = pool.open_candidate(order=0, when=3, value=torch.ones(2), support={1: 1.})
    assert key not in pool.entries and origin in pool.entries
    pool.close(None, stream=0, references=(fresh,))
    pool.close(None, stream=0)
    pool.close(None, stream=0)
    assert not pool.entries and not pool.transient_used.any()


def test_first_open_field_is_familiar_without_allocating_an_inventory_identity():
    cs, rows = _fixture()
    a, b, c = [r for _, r in rows[:3]]
    before = cs._concept_allocator.next_id
    _observe(cs, {a: 1., b: 1., c: 1.}, close=False)
    assert len(_candidates(cs)) == 3
    assert cs._concept_allocator.next_id == before
    assert all(entry.symbol is None and entry.count == 1 for entry in _candidates(cs).values())


def test_matching_witnessed_context_opens_next_order_with_context_in_support():
    cs, rows = _fixture()
    a, b, c, d = [r for _, r in rows[:4]]
    _observe(cs, {a: 1., c: 1., d: 1.})
    _observe(cs, {b: 1., c: 1., d: 1.})
    kinds = _candidates(cs, order=1)
    assert len(kinds) == 1
    assert set(next(iter(kinds.values())).support) == {a, b, c, d}
    assert cs.mtm.promote_candidates(cs) == []


def test_count_mints_once_with_weighted_definition_form_and_released_transient():
    cs, rows = _fixture()
    pool = cs.mtm
    pointers = pool.transient_ids.data_ptr(), pool.transient_used.data_ptr(), cs.similarity_codebook.W.data_ptr()
    support = {rows[0][1]: 2., rows[1][1]: 1.}
    for count in range(cs.concept_mint_count):
        key = observe_candidate(cs, 1, support)
        transient = pool.entries[key].transient
        _close(cs)
        if count < cs.concept_mint_count-1:
            assert pool.promote_candidates(cs) == []
    expected = centroid(cs, support)
    minted = pool.promote_candidates(cs)
    assert len(minted) == 1 and pool.entries[key].symbol == minted[0]
    assert pool.entries[key].transient is None
    assert not pool.transient_used[transient-int(pool.transient_ids[0])]
    row = cs._csw_row_of(minted[0])
    assert dict(cs.concept_weights(row)) == {r: w*3 for r, w in support.items()}
    torch.testing.assert_close(cs.similarity_codebook.W[row], expected)
    assert pool.promote_candidates(cs) == []
    assert pointers == (pool.transient_ids.data_ptr(), pool.transient_used.data_ptr(), cs.similarity_codebook.W.data_ptr())


def test_both_corner_refines_and_cannot_mint_until_it_leaves_the_horizon():
    cs, rows = _fixture()
    cs.mtm.horizon = 2
    for _ in range(3):
        key = observe_candidate(cs, 1, {rows[0][1]: 1., rows[1][1]: 1.}, pair=(1., 1.))
        _close(cs)
    assert cs.mtm.entries[key].refinement['stale'] == 2
    assert cs.mtm.promote_candidates(cs) == []
    for _ in range(3):
        key = observe_candidate(cs, 1, {rows[0][1]: 1., rows[1][1]: 1.})
        _close(cs)
    assert cs.mtm.promote_candidates(cs)


def test_gate_off_is_inert_and_nonfinite_field_fails_loud():
    cs, rows = _fixture()
    cs._promotion_enabled = False
    _observe(cs, {0: 1., 1: 1.})
    assert not cs.mtm.entries
    cs._promotion_enabled = True
    with pytest.raises(RuntimeError, match='NaN/Inf'):
        _observe(cs, {0: float('nan')})


def test_candidate_read_has_gradient_on_members_and_no_inventory_pool():
    cs, rows = _fixture()
    key = observe_candidate(cs, 1, {rows[0][1]: 2., rows[1][1]: 1.})
    value = cs.mtm.read_candidates(cs)[key]
    value.sum().backward()
    grad = cs.similarity_codebook.W.grad
    torch.testing.assert_close(grad[rows[0][1]], torch.full((_D,), 2/3))
    torch.testing.assert_close(grad[rows[1][1]], torch.full((_D,), 1/3))
    assert not hasattr(cs._concept_allocator.layer(), 'participation')


def test_candidate_checkpoint_restores_counter_support_lanes_and_next_match():
    cs, rows = _fixture()
    key = observe_candidate(cs, 1, {rows[0][1]: .7, rows[1][1]: .3}, pair=(.7,.2))
    _close(cs)
    state = cs.mtm.state()
    restored = MediumTermMemory(cs.mtm.capacity, n_vectors=cs.nVectors, field_width=cs.mtm.field_width)
    pointers = restored.transient_ids.data_ptr(), restored.transient_used.data_ptr()
    restored.restore(state)
    assert restored.candidate_counter == 1 and restored.entries[key].support == cs.mtm.entries[key].support
    assert restored.entries[key].readings == cs.mtm.entries[key].readings
    assert restored.match_candidate(1, cs.mtm.entries[key].value) == key
    assert restored.open_candidate(order=0, when=2, value=torch.ones(_D), support={0:1.}) == ('candidate',1)
    assert pointers == (restored.transient_ids.data_ptr(), restored.transient_used.data_ptr())
    state['entries'][0]['support'][0] = 99.
    assert restored.entries[key].support[0] == .7


def test_cover_merges_by_maximum_and_does_not_reinitialize_learned_edges():
    cs, rows = _fixture()
    for _ in range(3):
        key = observe_candidate(cs, 1, {rows[0][1]: .3, rows[1][1]: .2})
        _close(cs)
    cid = cs.mtm.promote_candidates(cs)[0]
    row = cs._csw_row_of(cid)
    store = cs._concept_allocator.layer()
    with torch.no_grad():
        store.values[store._index[row, rows[0][1]]] = 4.
    cs.cover_members(cid, {rows[0][1]: .5, rows[1][1]: .1})
    assert dict(cs.concept_weights(row)) == pytest.approx({rows[0][1]:4., rows[1][1]:.6})
