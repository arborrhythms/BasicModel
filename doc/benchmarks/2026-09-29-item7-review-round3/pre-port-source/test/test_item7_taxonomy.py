"""Language testimony and META use the native concept hierarchy."""
import copy
from dataclasses import replace

import pytest
import torch

from Spaces import _concept_alloc_of
from test_item9b_interpret import _operator
from ClauseRow import ClauseConcept
from reading_fixtures import finish_reading
from test_item7_acceptance import SentenceFixture


def test_part_testimony_symbols_the_parent_one_order_above_the_object():
    cs, interpret = _operator()
    word_cat = interpret.lookup_word([7], [1], form='cat')
    cat = interpret.forward(word_cat, order=1)
    word_animal = interpret.lookup_word([8], [1], form='animal')
    animal = interpret.forward(word_animal, order=1)
    parent = cs.index_part_row(cat, animal)
    assert parent != animal
    assert cs._concept_source_order(parent) == cs._concept_source_order(cat) + 1
    assert cat in cs.taxonomy_children(parent)
    assert parent in cs.taxonomy_parents(cat)
    assert parent in cs.word_concepts('animal')
    assert cs.index_part_row(cat, animal) == parent
    assert not cs.is_meta(parent)
    assert word_cat not in cs.taxonomy_children(parent)


def test_one_meta_generalizes_over_several_words_and_objects():
    cs, interpret = _operator()
    first = interpret.lookup_word([7], [1], form='bank')
    second = interpret.lookup_word([8], [1], form='shore')
    river = interpret.forward(first)
    money = cs.new_concept()
    _concept_alloc_of(cs).reference_orders[money] = 1
    meta = cs.bind_meta((first, second), (river, money))
    assert cs.meta_members(meta) == ((first, second), (river, money))
    assert cs.is_meta(meta)
    assert cs.word_references.candidates(first) == (river, money)
    assert cs.word_references.deref(first, selected=river) == river
    assert cs.word_references.deref(second, selected=money) == money
    with pytest.raises(ValueError, match='selection'):
        cs.word_references.deref(first)


def test_meta_binding_does_not_discriminate_at_binding_time():
    cs, interpret = _operator()
    word = interpret.lookup_word([7], [1], form='bank')
    first = interpret.forward(word)
    second = cs.new_concept()
    _concept_alloc_of(cs).reference_orders[second] = 1
    meta = cs.bind_meta((word,), (first, second))
    assert interpret.forward(word, selected=second) == second
    assert interpret.forward(word, selected=first) == first
    assert cs.meta_members(meta)[1] == (first, second)


def test_wholespace_taxonomy_payload_is_dropped_instead_of_quarantined():
    cs, _ = _operator()
    with pytest.warns(UserWarning, match='WholeSpace.*taxonomy'):
        cs.load_vocab_extras({'version': 1, 'taxonomy': {'1': [2, 3]},
                             'taxonomy_parent': {'2': 1}, 'meta_trust': {'1': [1, 0, 0, 0]}})
    assert 'legacy_whole_structure' not in cs.vocab_extras()
    assert not hasattr(cs, '_legacy_whole_structure')


def test_wholespace_has_no_taxonomy_or_truth_writer():
    from Spaces import WholeSpace
    for name in ('insert_meta', 'insert_relation', 'taxonomy_children', 'taxonomy_parent',
                 '_migrate_signed_int_taxonomy', '_record_truth_activations'):
        assert not hasattr(WholeSpace, name), name


def test_closing_uses_the_ordered_object_parent_and_deduplicates_it():
    from types import SimpleNamespace
    import torch
    from ClauseRow import Clause, attach_clause_index
    from Layers import TernaryTruthStore
    from Meaning import ConceptualMeaning
    cs, interpret = _operator()
    first = interpret.forward(interpret.lookup_word([7], [1], form='cat'))
    second = interpret.forward(interpret.lookup_word([8], [1], form='animal'))
    predicate = interpret.forward(interpret.lookup_word([9], [1], form='part'))
    store = TernaryTruthStore(8, capacity=16)
    model = SimpleNamespace()
    attach_clause_index(model, store, cs)
    store._predicate_kind = lambda ref: 'part' if ref == predicate else 'operator'
    source = Clause(ConceptualMeaning(torch.ones(3, 8), torch.ones(3, dtype=torch.bool)),
                    relation='part', refs=(first, predicate, second))
    row = store.write_clause(source)
    assert int(store.refs[row, 2]) == second
    parent = _concept_alloc_of(cs).relate_idx['part-kind', second, cs._concept_source_order(first) + 1]
    assert cs._concept_source_order(parent) == cs._concept_source_order(first) + 1
    assert first in cs.taxonomy_children(parent)
    assert store.write_clause(source) == row
    assert len(store) == 1


def test_failed_closing_does_not_leave_a_taxonomy_prefix():
    from dataclasses import replace
    from types import SimpleNamespace
    import torch
    from ClauseRow import Clause, attach_clause_index
    from Layers import TernaryTruthStore
    from Meaning import ConceptualMeaning
    cs, interpret = _operator()
    left = interpret.forward(interpret.lookup_word([7], [1], form='cat'))
    right = interpret.forward(interpret.lookup_word([8], [1], form='animal'))
    store = TernaryTruthStore(8, capacity=1)
    attach_clause_index(SimpleNamespace(), store, cs)
    source = Clause(ConceptualMeaning(torch.ones(3, 8), torch.ones(3, dtype=torch.bool)),
                    relation='part', refs=(left, left, right))
    bad = replace(source, children=(source,), refs=(('clause', 0), left, 99999))
    alloc = _concept_alloc_of(cs)
    before = alloc.next_id, dict(alloc.placement), alloc.records(right)
    with pytest.raises(ValueError):
        store.write_clause(bad)
    assert before == (alloc.next_id, dict(alloc.placement), alloc.records(right))
    assert not len(store)


def test_retiring_one_testimony_object_preserves_its_shared_meta():
    cs, interpret = _operator()
    word = interpret.lookup_word([7], [], form='bank')
    first = interpret.forward(word, occurrence=(0, 0))
    alloc = _concept_alloc_of(cs)
    second = cs.new_concept()
    alloc.reference_orders[second] = 1
    meta = cs.bind_meta((word,), (first, second))
    interpret.boundary()
    interpret.boundary()
    assert first in alloc.retired
    assert meta not in alloc.retired
    assert cs.meta_members(meta) == ((word,), (second,))
    assert cs.word_references.deref(word) == second


def test_restore_rebuilds_native_meta_membership_and_invalidates_direct_index():
    from Models import BaseModel
    from test_structural_checkpoint import _model_with
    from types import SimpleNamespace
    cs, interpret = _operator()
    model = _model_with(cs, SimpleNamespace())
    word = interpret.lookup_word([7], [], form='wug')
    obj = interpret.forward(word)
    alloc = _concept_alloc_of(cs)
    meta = alloc.interpretations[word, 1][1]
    snapshot = model._collect_structural_extras()['conceptual_spaces'][0]['allocator']
    # A pre-item-7 native META has its binary sigma definition and the owned
    # word/object binding, but no n-ary membership tags yet.
    for blob in snapshot['layers'].values():
        for key, records in blob['constituents'].items():
            blob['constituents'][key] = [record for record in records
                                        if record[0] not in ('word', 'object')]
    _ = cs.word_references
    BaseModel._restore_allocator_extras(model, cs, snapshot)
    assert cs.meta_members(meta) == ((word,), (obj,))
    assert cs.word_references.deref(word) == obj


def test_closing_binds_subject_form_to_particular_without_widening_compose_access(monkeypatch):
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('lift', ('lower', 'the', 'cat'), 'runs'))
    row = f.store.write_clause(clause)
    cid = int(f.store.row_ids[row])
    assert cid in f.cs.word_concepts('cat')
    assert cid not in f.cs.word_concepts('the')
    assert cid not in f.cs.word_concepts('runs')
    assert f.cs._concept_source_order(cid) == 1
    assert f.cs.resolve_word_concept('cat', order=1, previous=cid) == cid
    # The lexical tensor face has no situation: the initial object remains
    # the source, and only bounded reference resolution may select the row.
    assert f.cs.interpret.forward(f.words['cat'][0], order=1) == f.noun('cat')


@pytest.mark.parametrize('invalid', ['width', 'capacity'])
def test_native_phrase_admission_is_atomic(monkeypatch, invalid):
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('part', 'cat', ('verb', 'chases', 'mouse')))
    assert isinstance(clause.refs[2], ClauseConcept)
    if invalid == 'width':
        clause = replace(clause, refs=(*clause.refs[:2],
            replace(clause.refs[2], point=torch.zeros(9))))
    else:
        caps = f.cs._order_caps()
        order = max(f.cs._concept_source_order(cid) for cid in clause.refs[2].members) + 1
        base = sum(caps[:order])
        _concept_alloc_of(f.cs).layer(0)._row_next[base] = caps[order]
    alloc = _concept_alloc_of(f.cs)
    before = alloc.next_id, copy.deepcopy(alloc.placement), copy.deepcopy(alloc.relate_idx)
    with pytest.raises((ValueError, RuntimeError)):
        f.store.write_clause(clause)
    assert before == (alloc.next_id, alloc.placement, alloc.relate_idx)
    assert len(f.store) == 0


def test_selected_anaphor_form_binds_to_its_own_end_particular(monkeypatch):
    f = SentenceFixture(monkeypatch)
    prior = f.store.write_clause(f.clause(('lift', 'cat', 'runs')))
    entry = f.program(('lift', 'he', 'rests'))
    refs = entry.concept_ids.clone()
    refs[0] = f.store.row_ids[prior]
    values = entry.leaves.clone()
    values[0] = f.store.point_of_row(int(refs[0]))
    entry = replace(entry, reference_ids=refs, reference_orders=torch.tensor([1, -1]),
                    reference_values=values)
    clause = finish_reading(f.language, entry, registry=f.registry)
    row = f.store.write_clause(clause)
    assert int(f.store.row_ids[row]) in f.cs.word_concepts('he')
    assert int(f.store.row_ids[row]) not in f.cs.word_concepts('rests')
    assert f.cs.interpret.forward(f.words['he'][0], order=1) == f.noun('he')
