"""Refinement precedes a raise; geometry belongs to the current field."""
from types import SimpleNamespace

import pytest
import torch

import Spaces
from test_cs_sparse_weights import _cs, _mint_row
from test_concept_memberships import binary_features


def field_case(raw, patience=2):
    cs = _cs(nS=128, order=2)
    cs._mereology_raise = True
    cs.mereology_refine_patience = patience
    object.__setattr__(cs, '_model', SimpleNamespace(_training_step_count=0))
    _mint_row(cs, 0, 100)
    individual = _mint_row(cs, 1, 101)
    cs.add_concept_edge(individual, 0, 1.)
    native, extents = binary_features(cs, torch.tensor([raw]))
    read = cs.cs_read_memberships(native, extents)
    _, field = cs.cs_forward_content(read, torch.rand(cs.nVectors, cs.nDim))
    return cs, individual, field


def observe(cs, field, step):
    cs._model._training_step_count = step
    return cs.refine_over_collected(field=field)


def test_contiguous_both_stays_in_the_field_after_stalled_learning():
    cs, row, field = field_case([49, 48])
    for step in range(5):
        requests = observe(cs, field, step)
    assert any(r['sym'] == 101 and r['op'] == 'refine' for r in requests)
    assert not cs._refinement_raise_ready[row].any()
    assert not Spaces._concept_alloc_of(cs).retired


def test_scattered_both_requires_completed_updates_before_a_raise():
    # The unrelated middle position cannot bridge the two supports.
    cs, row, field = field_case([49, 65, 48])
    for _ in range(5):
        observe(cs, field, 0)
    assert not cs._refinement_raise_ready[row].any()
    observe(cs, field, 1)
    assert not cs._refinement_raise_ready[row].any()
    requests = observe(cs, field, 2)
    assert cs._refinement_raise_ready[row, 0]
    assert any(r['sym'] == 101 and r['op'] == 'raise' and r['runs'] == 2
               for r in requests)


def test_improvement_restarts_refinement_and_pure_reading_clears_it():
    cs, row, field = field_case([49, 65, 48])
    observe(cs, field, 0)
    observe(cs, field, 1)
    improved = field.clone()
    improved[row, ..., 1] *= .5
    observe(cs, improved, 2)
    assert not cs._refinement_raise_ready[row].any()
    observe(cs, improved, 3)
    assert not cs._refinement_raise_ready[row].any()
    observe(cs, improved, 4)
    assert cs._refinement_raise_ready[row].any()
    improved[row, ..., 1] = 0
    observe(cs, improved, 5)
    assert not cs._refinement_raise_ready[row].any()


def test_convergence_state_roundtrips_with_definition_inventory():
    cs, row, field = field_case([49, 65, 48])
    observe(cs, field, 0)
    observe(cs, field, 1)
    store = Spaces._concept_alloc_of(cs).layer()
    saved = store.parts_extras()
    other, other_row, other_field = field_case([49, 65, 48])
    Spaces._concept_alloc_of(other).layer().load_parts_extras(saved)
    observe(other, other_field, 2)
    assert other._refinement_raise_ready[other_row].any()
    # A new attended slot does not own this history.
    assert float(store.refinement_best[row]) == 1.


def test_pure_or_unknown_scattered_reading_never_requests_a_raise():
    for raw in ([49, 65, 49], [65, 65, 65]):
        cs, row, field = field_case(raw)
        for step in range(5):
            observe(cs, field, step)
        assert not cs._refinement_raise_ready[row].any()


def test_context_observer_cannot_bypass_refinement_before_assigning_a_kind():
    cs, focal, field = field_case([49, 65, 48])
    alternative = _mint_row(cs, 1, 102)
    cs._promotion_enabled = True
    store = Spaces._concept_alloc_of(cs).layer()
    store.witnessed[[focal, alternative]] = True
    from FamiliarTier import observe_candidate
    key = observe_candidate(cs, 1, {0: 1.}, focal=alternative)
    for step in range(3):
        cs._model._training_step_count = step
        cs.observe_open_field(field.detach())
        assigned = [entry for entry in cs.mtm.entries.values() if entry.order == 2]
        assert len(assigned) == (1 if step == 2 else 0)
    assert {focal, alternative, 0} <= set(assigned[0].support)


def test_releasing_one_batch_row_preserves_the_other_fields():
    cs, row, field = field_case([49, 65, 48])
    cs._refinement_field = field.expand(-1, 2, -1, -1).detach().clone()
    cs._refinement_raise_ready = torch.ones(cs.nVectors, 2, dtype=torch.bool)
    cs._clear_percept_field(batch=0)
    assert not cs._refinement_field[:, 0].any()
    torch.testing.assert_close(cs._refinement_field[:, 1], field[:, 0])
    assert not cs._refinement_raise_ready[:, 0].any()
    assert cs._refinement_raise_ready[:, 1].all()


@pytest.mark.parametrize('grouped', [False, True])
def test_extent_containment_does_not_fill_gaps_between_part_witnesses(grouped):
    cs = _cs(nS=128, order=2)
    cs._mereology_raise = True
    cs.mereology_refine_patience = 2
    object.__setattr__(cs, '_model', SimpleNamespace(_training_step_count=0))
    _mint_row(cs, 0, 100)
    individual = _mint_row(cs, 1, 101)
    cs.add_concept_edge(individual, 0, 1.)
    cs.add_concept_feature(0, 'ps', (1, 2) if grouped else 1, 1.)
    cs.add_concept_feature(0, 'ps', (3, 4) if grouped else 2, -1.)
    ids = torch.tensor([[1, 2, 99, 3, 4] if grouped else [1, 99, 2]])
    starts = torch.arange(ids.shape[1])
    spans = torch.stack((starts, starts + 1), -1)[None]
    extents = torch.tensor([[[0, ids.shape[1]]]])
    read = cs.cs_read_memberships((ids, spans, None, ids, spans), extents)
    # Both literals are contained by this subject, as before the routing fix.
    torch.testing.assert_close(read[0, 0, 0], torch.tensor([1., 1.]))
    _, field = cs.cs_forward_content(read, torch.rand(cs.nVectors, cs.nDim))
    for step in range(3):
        requests = observe(cs, field, step)
    assert cs._refinement_raise_ready[individual, 0]
    assert all(r['runs'] == 2 for r in requests)


@pytest.mark.parametrize('run', range(3))
def test_learned_particular_stays_at_one_scattered_kind_raises_to_two(run):
    cs, felix, _ = field_case([49, 49])
    garfield = _mint_row(cs, 1, 102)
    cs.add_concept_edge(garfield, 0, 1., negated=True)
    cs._promotion_enabled = True
    store = Spaces._concept_alloc_of(cs).layer()
    # Both primitive poles are now directly observed. Learn the particular's
    # link to that predicate instead of relearning its already-known absence.
    with torch.no_grad():
        store.values[store._index[(felix, 0)]] = 0.
    optimizer = torch.optim.Adam([store.values], lr=.03)
    native, extents = binary_features(cs, torch.tensor([[49], [48], [65]]))
    dictionary = torch.rand(cs.nVectors, cs.nDim)  # no selected initialization
    target = torch.tensor([[1., 0.], [0., 1.], [0., 0.]])
    initial = None
    for step in range(32):
        optimizer.zero_grad()
        read = cs.cs_read_memberships(native, extents)
        _, field = cs.cs_forward_content(read, dictionary)
        loss = (field[felix, :, 0] - target).square().mean()
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        optimizer.step()
        store.project_parts()
        cs._model._training_step_count += 1
    assert initial > .1
    native, extents = binary_features(cs, torch.tensor([[49, 49]]))
    read = cs.cs_read_memberships(native, extents)
    _, field = cs.cs_forward_content(read, dictionary)
    torch.testing.assert_close(field[felix, 0, 0], torch.tensor([1., 0.]))
    observe(cs, field, 32)
    context = torch.zeros(store.nOutput)
    context[0] = 1.
    evidence = torch.ones(2 * (store.nOutput + 1))
    assert cs.maybe_raise_order(felix, garfield, context, batch=0, evidence=evidence) is None
    # The same learned property over separated particulars cannot become
    # pure by further fitting this definition: both supports remain witnessed.
    native, extents = binary_features(cs, torch.tensor([[49, 65, 48]]))
    for step in range(33, 37):
        optimizer.zero_grad()
        read = cs.cs_read_memberships(native, extents)
        _, field = cs.cs_forward_content(read, dictionary)
        loss = (field[felix, 0, 0] - torch.tensor([1., 0.])).square().mean()
        loss.backward()
        optimizer.step()
        store.project_parts()
        observe(cs, field, step)
    cat = cs.maybe_raise_order(felix, garfield, context, batch=0, evidence=evidence)
    assert cat is not None and cs.mtm.entries[cat].order == 2
    assert set(cs.mtm.entries[cat].support) == {0, felix, garfield}
    assert cs.mtm.entries[cat].symbol is None
    # The geometry and permission are per turn, never the row's identity.
    cs._clear_percept_field()
    assert cs.maybe_raise_order(felix, garfield, context, batch=0, evidence=evidence) is None
