"""Redundancy certificates free seats without aliasing retired identities."""
import copy
import warnings
from types import SimpleNamespace

import pytest
import torch

from ConceptForgetting import reclaim_concepts, warn_capacity
from Layers import TernaryTruthStore
from test_concept_capacity_policy import _cs


@torch.no_grad()
def concept(cs, *, part=10, magnitude=.5, edges=True):
    cid = cs.new_concept()
    cs.add_part(cid, part)
    cs.add_whole(cid, 20)
    row = cs._csw_concept_row(0, cid)
    if edges:
        cs._concept_allocator.layer().features.add_edge(row, 0, .7)
    cs.similarity_codebook.W[row].fill_(magnitude)
    return cid, row


def test_smaller_exact_redundancy_frees_seat_but_never_reuses_handle():
    cs = _cs(4)
    weak, row = concept(cs, magnitude=.1)
    strong, _ = concept(cs, magnitude=.9)
    with pytest.warns(RuntimeWarning, match='maximum preallocation'):
        unique, _ = concept(cs, part=11, magnitude=0.)
    assert warn_capacity(cs)
    report = reclaim_concepts(cs)
    assert report['retired'] == (weak,) and report['reclaimed'] == 1
    assert cs._csw_row_of(strong) is not None and cs._csw_row_of(unique) is not None
    assert cs._csw_row_of(weak) is None and weak in cs._concept_allocator.retired
    with pytest.warns(RuntimeWarning, match='maximum preallocation'):
        new, new_row = concept(cs, part=12)
    assert new > unique and new != weak and new_row == row
    assert cs.concept_id_at_row(row) == new
    assert cs._csw_row_of(weak) is None
    assert cs._concept_allocator.reclaimed_seats == 1
    with pytest.raises(ValueError, match='retired concept'):
        cs._csw_concept_row(0, weak)


def test_live_reference_preserves_weak_candidate_and_keeps_a_representative():
    cs = _cs(8)
    weak, _ = concept(cs, magnitude=.1)
    strong, _ = concept(cs, magnitude=.9)
    store = TernaryTruthStore(8, capacity=4)
    store.append_idea(torch.ones(8), sentence_index=0)
    store.refs[0, 0] = weak
    report = reclaim_concepts(cs, store)
    assert report['retired'] == (strong,)
    assert cs._csw_row_of(weak) is not None


def test_two_empty_concepts_are_not_redundant():
    cs = _cs(8)
    first, _ = concept(cs, magnitude=0., edges=False)
    second, _ = concept(cs, magnitude=0., edges=False)
    before = copy.deepcopy(cs._concept_allocator.layer()._tensor_rows)
    with pytest.warns(RuntimeWarning, match='no safely redundant'):
        report = reclaim_concepts(cs)
    assert report['reclaimed'] == 0
    assert cs._concept_allocator.layer()._tensor_rows == before
    assert not {first, second} & cs._concept_allocator.retired


@torch.no_grad()
def test_unused_mints_are_reclaimed_by_magnitude_without_erasing_defined_or_referenced_concepts():
    cs = _cs(16)
    ids = [cs.new_concept() for _ in range(9)]
    rows = [cs._csw_concept_row(0, cid) for cid in ids]
    large, small, referenced, defined, posted, edge_source, edge_target, feature, witnessed = ids
    layer = cs._concept_allocator.layer()
    for row in rows:
        cs.similarity_codebook.W[row].zero_()
    cs.similarity_codebook.W[rows[0], 0] = .9
    cs.similarity_codebook.W[rows[1], 1] = .05  # A different direction, not an exact duplicate.
    cs.add_part(defined, 42)
    store = cs._definition_store
    stored = store.append_idea(torch.ones(8), sentence_index=0)
    store.refs[stored, 0] = referenced
    store._leaf_postings[rows[4], 0] = {stored}
    layer.add_edge(rows[6], rows[5], 0.)  # Zero coefficients still have sparse placement.
    layer.features.add_edge(rows[7], 0, 0.)
    layer.witnessed[rows[8]] = True
    retired = []
    drop = cs._concept_allocator.drop
    def record_drop(cid):
        retired.append(cid)
        return drop(cid)
    cs._concept_allocator.drop = record_drop
    report = reclaim_concepts(cs, store)
    assert report['reclaimed'] == 2 and report['retired'] == (small, large)
    assert retired == [small, large]
    assert all(cs._csw_row_of(cid) is not None for cid in ids[2:])
    assert {small, large} <= cs._concept_allocator.retired


def test_incoming_sparse_edge_or_record_prevents_recycling():
    cs = _cs(8)
    first, row = concept(cs, magnitude=.1)
    second, _ = concept(cs, magnitude=.9)
    parent, parent_row = concept(cs, part=42)
    cs.add_part(parent, ('sym', second))
    layer = cs._concept_allocator.layer()
    layer.add_edge(parent_row, row, 1.)
    with pytest.warns(RuntimeWarning, match='no safely redundant'):
        report = reclaim_concepts(cs)
    assert report['reclaimed'] == 0
    assert cs._csw_row_of(first) == row


def test_different_order_or_definition_never_merges():
    cs = _cs(8)
    first, _ = concept(cs, magnitude=.1)
    second, _ = concept(cs, magnitude=.9)
    cs._concept_allocator.reference_orders[second] = 1
    with pytest.warns(RuntimeWarning, match='no safely redundant'):
        assert reclaim_concepts(cs)['reclaimed'] == 0
    cs._concept_allocator.reference_orders[second] = 0
    cs.add_part(second, 11)
    with pytest.warns(RuntimeWarning, match='no safely redundant'):
        assert reclaim_concepts(cs)['reclaimed'] == 0


def test_exact_physical_preallocation_warns_even_for_external_handles():
    cs = _cs(4)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        for index in range(4):
            assert cs._csw_concept_row(0, 1000 + index) == index
        warn_capacity(cs)
    assert sum('maximum preallocation' in str(w.message) for w in seen) == 1


def test_equal_descriptions_do_not_merge_distinct_individuals():
    cs = _cs(8)
    first, _ = concept(cs, magnitude=.1)
    second, _ = concept(cs, magnitude=.9)
    cs._concept_allocator.reference_orders.update({first: 1, second: 1})
    with pytest.warns(RuntimeWarning, match='no safely redundant'):
        assert reclaim_concepts(cs)['reclaimed'] == 0


def test_optimizer_moments_follow_surviving_edges_and_reused_code_seat_is_clean():
    cs = _cs(8)
    weak, weak_row = concept(cs, magnitude=.1)
    strong, strong_row = concept(cs, magnitude=.9)
    matrix, code = cs._concept_allocator.layer().features, cs.similarity_codebook.W
    optimizer = torch.optim.Adam([matrix.values, code])
    matrix._optimizer = optimizer
    (matrix.values.sum() + code.sum()).backward()
    optimizer.step()
    surviving_moment = optimizer.state[matrix.values]['exp_avg'][matrix._index[strong_row, 0]].clone()
    survivor_code_moment = optimizer.state[code]['exp_avg'][strong_row].clone()
    model = SimpleNamespace(_optimizer=optimizer, _optimizer_leaves=lambda opt: [opt])
    assert reclaim_concepts(cs, model=model)['retired'] == (weak,)
    torch.testing.assert_close(optimizer.state[matrix.values]['exp_avg'][0], surviving_moment)
    torch.testing.assert_close(optimizer.state[code]['exp_avg'][strong_row], survivor_code_moment)
    assert not optimizer.state[code]['exp_avg'][weak_row].any()
    assert not code.grad[weak_row].any()


def test_reclaimed_capacity_and_taxonomy_provenance_roundtrip():
    from test_structural_checkpoint import _model_with
    cs = _cs(8)
    weak, row = concept(cs, magnitude=.1)
    strong, _ = concept(cs, magnitude=.9)
    reclaim_concepts(cs)
    cs._concept_allocator.ltm_edges[strong, 101] = {('ltm', 'fixture', 992)}
    source = _model_with(cs, SimpleNamespace())
    saved = source._collect_structural_extras()['conceptual_spaces'][0]['allocator']
    target_cs = _cs(8)
    _model_with(target_cs, SimpleNamespace())._restore_allocator_extras(target_cs, saved)
    assert target_cs._concept_allocator.reclaimed_seats == 1
    assert target_cs._concept_allocator.ltm_edges == cs._concept_allocator.ltm_edges
    new, seat = concept(target_cs, part=12)
    assert new > strong and seat == row and target_cs._csw_row_of(weak) is None


def test_only_unreferenced_zero_relevance_column_is_pruned():
    from IndependentComponents import ColumnDictionary
    cs = _cs(8)
    cid, _ = concept(cs)
    verbs = ColumnDictionary(24)
    verbs.directions[str(cid)] = torch.nn.Parameter(torch.zeros(24))
    verbs.relevance[str(cid)] = torch.nn.Parameter(torch.tensor(0.))
    cs.components.verbs = verbs
    with pytest.warns(RuntimeWarning, match='no safely redundant'):
        assert reclaim_concepts(cs, retained=[cid])['reclaimed'] == 0
    assert reclaim_concepts(cs)['retired'] == (cid,)
    assert str(cid) not in verbs.directions and str(cid) not in verbs.relevance


def test_component_coordinates_protect_referenced_seats():
    from IndependentComponents import ColumnDictionary
    cs = _cs(8)
    first, row = concept(cs)
    second, other = concept(cs)
    verbs = ColumnDictionary(24)
    direction = torch.zeros(24)
    direction[[row, other]] = 1
    verbs.directions['99'] = torch.nn.Parameter(direction)
    verbs.relevance['99'] = torch.nn.Parameter(torch.tensor(1.))
    cs.components.verbs = verbs
    with pytest.warns(RuntimeWarning, match='no safely redundant'):
        assert reclaim_concepts(cs)['reclaimed'] == 0


def test_concept_meaning_starts_zero_and_grows_without_changing_direction():
    from MeaningCodes import defined_code
    direction = torch.tensor([1., 2., 3., 4.])
    values = [defined_code(direction, count) for count in (0, 1, 4, 20)]
    assert not values[0].any()
    norms = [float(value.norm()) for value in values]
    assert norms == pytest.approx([0., .2, .5, 20/24])
    for value in values[1:]:
        torch.testing.assert_close(value / value.norm(), direction / direction.norm())
