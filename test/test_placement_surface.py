"""Placement mechanism certificates on the layout gate-4 native-row fixture."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from MeaningCodes import DEFINEDNESS_RECURRENCE, identity_code, defined_code
from MereologicalCodes import placement_step
from test_review13_subspaces import fixture, word


def lesson(monkeypatch):
    # The layout gate-4 form fixture, with eight pole pairs so different
    # directions can be measured (its original one pair cannot separate codes).
    cs, cb, ps, alloc = fixture(meaning_pairs=8)
    rows = [word(cs, [(4 if i % 2 == 0 else 7, 1.)]) for i in range(7)]
    names = dict(zip(rows, (b'p', b'q', b'r', b's', b't', b'u', b'v')))
    monkeypatch.setattr(cs, 'word_surface_for_row', names.get)
    reading = SimpleNamespace(word_rows=torch.tensor([rows[0], rows[1], rows[0]]), concept_ids=None)
    derived = cb.mereology
    # Five already placed, inactive symbols supply the specified negatives.
    derived.place_rows(rows[2:], torch.zeros(len(cb.W)), b'background')
    cs._sentence_seen_rows[0, :2] = torch.tensor(rows[:2])
    return cs, derived, rows, reading


def test_layout_lesson_context_rises_without_random_collapse(monkeypatch, tmp_path):
    cs, derived, rows, reading = lesson(monkeypatch)
    rng = torch.get_rng_state().clone()
    series = []
    # Two definedness recurrences: eight closes, fixed before observing results.
    for close in range(2 * DEFINEDNESS_RECURRENCE):
        cs.prime_seen(torch.tensor([rows[:2]]))
        derived.observe_close(reading)
        series.append(derived.collapse_audit(rows[:2]))
        assert derived.placement_witnesses[rows[:2]].tolist() == [close+1, close+1]
    (tmp_path/'certificate.json').write_text(json.dumps(dict(series=series), indent=2))
    assert series[-1]['context'] > series[0]['context']
    assert series[-1]['random'] <= series[0]['random'] + 1e-6
    assert torch.equal(rng, torch.get_rng_state())
    from MereologicalCodes import MereologicalCodes
    assert not hasattr(MereologicalCodes, 'occurrence_terms')


def test_meet_step_uses_candidate_mass_and_negative_subgradient():
    q = F.normalize(torch.tensor([1., 2., 0.]), dim=0)
    context = torch.tensor([[.8, .6, 0.]])
    negatives = torch.tensor([[0., .6, .8]])
    weights = torch.tensor([2.])
    gradient = (q < context[0]).float() * 2/context.sum() - (q < negatives[0]).float()/negatives.sum()
    expected = F.normalize((q + .1*gradient).clamp_min(0), dim=0)
    torch.testing.assert_close(placement_step(q, context, weights, negatives), expected)


def test_read_point_geometry_conserves_energy_and_never_writes(monkeypatch):
    cs, derived, rows, reading = lesson(monkeypatch)
    a,b = rows[:2]
    derived.placement[[a,b]] = F.normalize(torch.ones(2,derived.meaning_pairs),dim=-1)
    derived.placement_witnesses[[a,b]] = 1
    cs.prime_seen(torch.tensor([[a]]))
    stored, seen = cs._priming_boosts.clone(), cs._priming_seen.clone()
    context = cs.placement_context(0)
    torch.testing.assert_close(context,(seen[0]-1).clamp_min(0),rtol=0,atol=0)
    assert stored[0,b] > 1 and seen[0,b] == 1
    first, second = cs.relevance_weights(batch=1), cs.relevance_weights(batch=1)
    assert torch.equal(first,second) and torch.equal(stored,cs._priming_boosts)
    assert torch.equal(seen,cs._priming_seen)
    audit = cs._last_priming_diffusion
    torch.testing.assert_close(audit['before'],audit['after'],rtol=0,atol=1e-9*cs.nVectors)
    calls=[]
    original=cs.placement_context
    monkeypatch.setattr(cs,'placement_context',lambda batch=0: calls.append(batch) or original(batch))
    derived.observe_close(reading)
    assert calls==[0]


def test_unchanged_forward_reuses_edges_bridge_forms_and_witnesses(monkeypatch):
    cs, derived, rows, reading = lesson(monkeypatch)
    from Spaces import _concept_alloc_of
    parts = _concept_alloc_of(cs).parts
    a, b = (cs.concept_id_at_row(row) for row in rows[:2])
    parts.add(a, b)
    from Priming import part_edges
    part_edges(cs)
    cs.project_priming_to_towers(None, None, batch=1)
    derived.begin_forward()
    cs.components.begin_forward()
    counts = cs._priming_edge_rebuilds, cs._priming_bridge_rebuilds, derived.form_rows_derived, cs.components._inventory_rebuilds
    from Priming import part_edges
    part_edges(cs)
    cs.project_priming_to_towers(None, None, batch=1)
    derived.begin_forward()
    cs.components.begin_forward()
    assert counts == (cs._priming_edge_rebuilds, cs._priming_bridge_rebuilds, derived.form_rows_derived, cs.components._inventory_rebuilds)
    assert derived.last_rederived_rows == 0
    # Appending a chain, even one mentioning the same symbol, is not taxonomy.
    revision = parts.symbol_revision
    parts.add((parts.OCCURRENCE, -7), (parts.SYMBOL, a))
    from Priming import part_edges
    part_edges(cs)
    assert parts.symbol_revision == revision and cs._priming_edge_rebuilds == counts[0]
    with pytest.raises(ModuleNotFoundError):
        __import__('PartRead')


def test_promotion_folds_form_and_takes_one_context_step(monkeypatch):
    import weakref
    from ClauseRow import attach_clause_index
    from Layers import LTM0
    from MemoryConsolidation import consolidate
    from Meaning import ConceptualMeaning
    cs, derived, rows, reading = lesson(monkeypatch)
    cs._symbolic_order = 3
    cb = cs.similarity_codebook
    store = LTM0(cb.W.shape[-1], capacity=32, percept_width=3)
    object.__setattr__(cs, '_clause_store_ref', weakref.ref(store))
    attach_clause_index(SimpleNamespace(), store, cs)
    values = []
    addresses = []
    for i, weight in enumerate((1., .5, .25)):
        value = torch.zeros(cb.W.shape[-1]); value[:3] = torch.tensor([.2,.4,.6]) + i*.1
        value[3:] = .2*(i+1)
        row = store.append_meaning(ConceptualMeaning.from_description(value), order=0,
            evidence=(weight,0.), sentence_index=i, document_key='promoting')
        values.append(value)
        addresses.append(store.occurrence_of(row))
        if i == 0:
            cs.mtm.open(addresses[0], kind=None, value=value, when=i)
        else:
            cs.mtm.bind(addresses[0], addresses[-1], kind=None, value=value, when=i)
    cs.prime_seen(torch.tensor([rows[:2]]))
    context = derived.observe_close(reading)
    before_codes = derived.placement.clone()
    placed = derived.placement_witnesses > 0
    key = derived.reading_key(reading)
    cs._mtm_placement_contexts = {0: (context,key)}
    actions = cs.mtm.promote_candidates(cs)
    assert actions
    symbol = cs.mtm.entries[addresses[0]].symbol
    row = cs._csw_row_of(symbol)
    cb.lookup_rows(row)
    expected_form = (torch.stack(values)[:,:3] * torch.tensor([1.,.5,.25])[:,None]).sum(0)/1.75
    torch.testing.assert_close(cb.W[row,:3], expected_form)
    q = F.normalize(identity_code(derived.content_key(row), derived.meaning_pairs,
                                 derived.meaning_ones), dim=0)
    positives = (context > 0) & placed
    negatives = derived._negatives(row, context, placed, key)
    expected = derived.placement_step(q, before_codes[positives], context[positives], before_codes[negatives])
    torch.testing.assert_close(derived.placement[row], expected)
    assert int(derived.placement_witnesses[row]) == 1
    torch.testing.assert_close(cb.W[row,3:11], defined_code(expected,1))
    assert cb.W[row,11:].eq(0).all()


def test_placement_checkpoint_is_versioned_and_fixed_capacity(monkeypatch):
    cs, derived, rows, reading = lesson(monkeypatch)
    cs.prime_seen(torch.tensor([rows[:2]])); derived.observe_close(reading)
    state = copy.deepcopy(derived.state_dict())
    saved = derived.placement_extras()
    pointers = derived.placement.data_ptr(), derived.placement_witnesses.data_ptr()
    derived.load_placement_extras(saved)
    derived.load_state_dict(state)
    assert pointers == (derived.placement.data_ptr(), derived.placement_witnesses.data_ptr())
    with pytest.raises(ValueError, match='placement checkpoint version'):
        derived.load_placement_extras(None)
    bad = dict(state); bad.pop('placement_version')
    with pytest.raises(ValueError, match='placement checkpoint version'):
        derived.load_state_dict(bad)
    bad = dict(saved, placement=torch.zeros(1, derived.meaning_pairs))
    with pytest.raises(ValueError, match='constructed column'):
        derived.load_placement_extras(bad)
    with pytest.raises(RuntimeError, match='resizing is forbidden'):
        cs.prime_seen(torch.zeros(cs.nStreams+1,1,dtype=torch.long))
