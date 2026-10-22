"""Membership is a read of the retained derivation, including after load."""
import pytest
import torch

from test_clause_acceptance import SentenceFixture


def test_postings_rebuild_from_part_without_inverting_the_narrowed_point(monkeypatch):
    from MemoryIndex import configure_model_index
    f = SentenceFixture(monkeypatch)
    row = f.store.write_clause(f.clause(('lift', 'a', ('verb', 'v', 'b'))))
    expected = {f.cs._csw_row_of(f.noun(name)) for name in ('a', 'v', 'b')}
    def forbidden(*args, **kwargs):
        raise AssertionError('membership must not generate or invert a concept')
    f.language.generate_policy_logits = forbidden
    configure_model_index(f.model, f.store)
    assert set(f.store.leaf_terms(row, 0)) == expected
    original = f.store.slots[row, 0].clone()
    f.store.slots[row, 0].fill_(.5)
    f.store.reindex_meanings()
    assert set(f.store.leaf_terms(row, 0)) == expected
    f.store.slots[row, 0].copy_(original)
    state = f.store.state_dict()
    extras = f.store.semantic_extras()
    # A stale serialized cache is rebuilt from the retained source.
    state['posting_codes'] = torch.full_like(state['posting_codes'], 999)
    # Keep its sorted/unique wire contract while changing every term.
    state['posting_codes'] += torch.arange(len(state['posting_codes']))
    f.store.load_state_dict(state)
    f.store.load_semantic_extras(extras)
    configure_model_index(f.model, f.store)
    assert set(f.store.leaf_terms(row, 0)) == expected
    assert 999 not in f.store.leaf_terms(row, 0)


def test_part_membership_terminates_on_cycles_and_retains_reachable_leaves():
    from PartRead import leaves
    from LTM1 import LTM1
    parts = LTM1(32)
    parts.add(11, 2, symbol=10)
    parts.add(10, 3, symbol=11)
    assert set(leaves(parts, 10)) == {2, 3}


def test_closing_excludes_its_own_witness_without_restricting_copular_reads():
    from dataclasses import replace
    from Meaning import ConceptualMeaning
    from ThoughtReferences import bindings, open_slots, question, with_slots
    from ThoughtClosing import closing_question
    from ClauseRow import Clause
    from part_fixtures import native_store
    store, (name, predicate, answer) = native_store(points=torch.eye(4)[:3])
    value = ConceptualMeaning(torch.eye(4)[:3], torch.ones(3, dtype=torch.bool),
        role_refs=(None, ('sym', predicate), ('sym', answer)), sentence_kind='relation',
        bindings={'_forward_references': ((0, name),)})
    earlier = store.append_meaning(question(value, (('referent', 0),)),
                                   kind='question', evidence=(0., 0.), sentence_index=0)
    assert open_slots(store.meaning_of(earlier))
    source = store.append_meaning(with_slots(
        replace(value, bindings=dict(bindings(value), _equality=True)),
        (('referent', 0),), pair=(1., 0.)), evidence=(1., 0.), sentence_index=1)
    raw = store.raw_meaning_of(source)
    clause = Clause(raw, relation='part', refs=(-1, predicate, answer), evidence=(1., 0.))
    closing = closing_question(raw, clause, store=store, written_rows={id(clause): source})
    assert closing is not None and open_slots(closing) == (('referent', 0),)
    assert closing.role_refs == raw.role_refs
    # Ordinary interpretation still reads the witnessed copular complement.
    assert store.meaning_of(source).role_refs[0] == ('sym', answer)
    # The same observed complement can resolve a different occurrence.
    resolved = store.meaning_of(earlier)
    assert not open_slots(resolved)
    assert resolved.role_refs[0] == ('sym', answer)
    assert bindings(resolved)['_thought_witnesses'] == (store.occurrence_of(source),)
    assert store.raw_meaning_of(source).role_refs == raw.role_refs


@pytest.mark.parametrize('relative', (False, True))
def test_forward_name_accepts_a_witnessed_occurrence_complement(relative):
    from dataclasses import replace
    from Meaning import ConceptualMeaning
    from ThoughtReferences import bindings, open_slots, question, with_slots
    from part_fixtures import native_store
    store, (name, predicate, answer) = native_store(points=torch.eye(4)[:3])
    atom = replace(ConceptualMeaning.from_description(torch.eye(4)[2]),
                   role_refs=(('sym', answer), None, None))
    if relative:
        complement = store.parts.add(predicate, answer)
        assert store._concept_point(complement) is None
        reference = ('sym', complement)
    else:
        complement = store.append_meaning(atom, evidence=(1., 0.), sentence_index=0)
        reference = store.occurrence_of(complement)
    value = ConceptualMeaning(torch.eye(4)[:3], torch.ones(3, dtype=torch.bool),
        role_refs=(None, ('sym', predicate), reference), sentence_kind='relation',
        bindings={'_forward_references': ((0, name),), '_equality': True})
    source = store.append_meaning(with_slots(value, (('referent', 0),), pair=(1., 0.)),
                                 evidence=(1., 0.), sentence_index=1)
    pending = store.append_meaning(question(value, (('referent', 0),)),
                                  kind='question', evidence=(0., 0.), sentence_index=2)
    raw = store.raw_meaning_of(pending)
    resolved = store.meaning_of(pending)
    assert not open_slots(resolved)
    assert not bindings(resolved).get('_forward_references')
    assert resolved.role_refs[0] == reference
    assert bindings(resolved)['_thought_witnesses'] == (store.occurrence_of(source),)
    assert store.raw_meaning_of(pending).role_refs == raw.role_refs


def test_typed_part_read_follows_attachments_and_keeps_the_clause_subject(monkeypatch):
    from PartRead import expression_components
    from QueryWork import QueryWorkBudget
    f = SentenceFixture(monkeypatch)
    tree = ('verb', ('lift', 'a', ('verb', 'v', 'b')), 'ending')
    row = f.store.write_clause(f.clause(tree))
    root = f.store.raw_meaning_of(row).role_refs[0]
    terms, _, incomplete = expression_components(f.store, (root,),
        max_records=64, work=QueryWorkBudget(64))
    assert not incomplete
    assert terms[root]['kind'] == 'lift'
    assert terms[root]['operands'][0] == ('sym', f.noun('a'))
    assert terms[root]['reference'] != root


def test_a_later_word_definition_does_not_retype_a_retained_marker(monkeypatch):
    from PartRead import expression_components
    from QueryWork import QueryWorkBudget
    f = SentenceFixture(monkeypatch)
    marker = f.cs.interpret.lookup_word([81], [], form='marker-form')
    content = f.noun('content')
    edge = f.store.parts.add(marker, content)
    pos = f.language.parts_of_speech
    pos.add(marker, 'surface_argin1')
    # A legal numerical departure can use a product here. The marker's
    # admitted type still supplies no conceptual operand to the head read.
    pos.add(marker, 'product_argin1')
    pos.add(content, 'product_argin2')
    reference = ('sym', edge)
    before, _, missing = expression_components(f.store, (reference,),
        max_records=64, work=QueryWorkBudget(64))
    assert not missing and before[reference]['reference'] == ('sym', content)
    other = f.noun('another-object')
    f.cs.interpret.define(marker, other)
    assert not f.cs.interpret.no_object(marker)
    after, _, missing = expression_components(f.store, (reference,),
        max_records=64, work=QueryWorkBudget(64))
    assert not missing and after[reference]['reference'] == ('sym', content)


def test_answer_read_checks_a_named_composite_before_descending_its_head(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from BindingAnswers import matches
    from Meaning import ConceptualMeaning
    f = SentenceFixture(monkeypatch)
    row = f.store.write_clause(f.clause(('verb', ('verb', 'content', 'ending'), 'outer')))
    reference = f.store.raw_meaning_of(row).role_refs[0]
    named = int(f.store.parts.refs[reference[1], 0])
    word = f.cs.interpret.lookup_word([83], [], form='bound-name')
    f.cs.interpret.define(word, named)
    value = replace(ConceptualMeaning.from_description(f.store.slots[row, 0]),
        role_refs=(reference, None, None), bindings={'_bound_roles': (('referent', 0),)})
    model = SimpleNamespace(_concept_owner=lambda:f.cs,
                            symbolSpace=SimpleNamespace(ltm_store=f.store))
    assert matches(model, value, 'bound-name')


def test_answer_read_follows_admitted_surface_with_a_composed_marker(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from BindingAnswers import matches
    from Meaning import ConceptualMeaning
    f = SentenceFixture(monkeypatch)
    marker = f.cs.interpret.lookup_word([84], [], form='unbound-form')
    f.words['unbound-form'] = marker, marker
    row = f.store.write_clause(f.clause(('surface',
        ('surface', 'marker-object', 'unbound-form'),
        ('verb', 'content', 'ending'))))
    reference = f.store.raw_meaning_of(row).role_refs[0]
    value = replace(ConceptualMeaning.from_description(f.store.slots[row, 0]),
        role_refs=(reference, None, None), bindings={'_bound_roles': (('referent', 0),)})
    model = SimpleNamespace(_concept_owner=lambda:f.cs,
                            symbolSpace=SimpleNamespace(ltm_store=f.store))
    assert matches(model, value, 'content')
