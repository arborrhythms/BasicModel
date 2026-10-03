"""Language testimony and META use the native concept hierarchy."""
import copy
from dataclasses import replace

import pytest
import torch

from Spaces import _concept_alloc_of
from test_word_interpretation import _operator
from reading_fixtures import finish_reading
from test_clause_acceptance import SentenceFixture


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
    assert parent not in cs.definitions.word_ids
    assert word_cat not in cs.taxonomy_children(parent)


def test_definition_rows_generalize_over_several_words_and_objects():
    cs, interpret = _operator()
    first = interpret.lookup_word([7], [1], form='bank')
    river = interpret.forward(first)
    second = interpret.lookup_word([8], [1], form='shore')
    money = interpret.forward(second)
    interpret.define(first, money)
    interpret.define(second, river)
    assert cs.definitions.objects(first) == (river, money)
    assert cs.definitions.deref(first, selected=river) == river
    assert cs.definitions.deref(second, selected=money) == money
    with pytest.raises(ValueError, match='selection'):
        cs.definitions.deref(first)


def test_definition_binding_preserves_ambiguous_objects_until_selection():
    cs, interpret = _operator()
    word = interpret.lookup_word([7], [1], form='bank')
    first = interpret.forward(word)
    second = cs.new_concept()
    _concept_alloc_of(cs).reference_orders[second] = 1
    interpret.define(word, second)
    assert interpret.forward(word, selected=second) == second
    assert interpret.forward(word, selected=first) == first
    assert cs.definitions.objects(word) == (first, second)


def test_wholespace_taxonomy_payload_is_dropped_instead_of_quarantined():
    cs, _ = _operator()
    with pytest.warns(UserWarning, match='WholeSpace.*taxonomy'):
        cs.load_vocab_extras({'version': 1, 'taxonomy': {'1': [2, 3]},
                             'taxonomy_parent': {'2': 1}, 'meta_trust': {'1': [1, 0, 0, 0]}})
    assert 'legacy_whole_structure' not in cs.vocab_extras()
    assert not hasattr(cs, '_legacy_whole_structure')




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
def test_referenced_phrase_admission_is_atomic(monkeypatch, invalid):
    from ClauseRow import attach_clause_index
    from Layers import TernaryTruthStore
    from Meaning import ConceptualMeaning
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('part', 'cat', ('verb', 'chases', 'mouse')))
    assert clause.refs[2] == ('clause', 0)
    if invalid == 'width':
        child = clause.children[0]
        child = replace(child, point=torch.zeros(9),
                        meaning=ConceptualMeaning.from_description(torch.zeros(9)))
        clause = replace(clause, children=(child,))
    else:
        # A phrase needs its own LTM occurrence, not an inventory seat.
        # With only one store row the whole two-row transaction is refused.
        f.store = TernaryTruthStore(f.store.nDim, capacity=1)
        attach_clause_index(f.model, f.store, f.cs)
    alloc = _concept_alloc_of(f.cs)
    before = alloc.next_id, copy.deepcopy(alloc.placement), copy.deepcopy(alloc.relate_idx)
    if invalid == 'width':
        with pytest.raises(ValueError):
            f.store.write_clause(clause)
    else:
        assert f.store.write_clause(clause) == -1
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
