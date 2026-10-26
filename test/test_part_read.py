"""Semantic chains and identity bindings replace retained derivations."""
import pytest
import torch

from test_clause_acceptance import SentenceFixture






def pending_chain(store, value, name, complement, *, sentence):
    from dataclasses import replace
    from Meaning import ConceptualMeaning
    from meaning_fixtures import unsupported as question
    from LTM1 import OCCURRENCE
    atom = replace(ConceptualMeaning.from_description(value),
                   bindings={'_forward_references': ((0, name),)})
    row = store.append_meaning(question(atom, (('referent', 0),)),
        kind='question', sentence_index=sentence, when=False)
    edge = store.parts.add((OCCURRENCE, int(store.address_keys[row])), complement,
                          truth_plus=1.)
    return ('sym', edge)


def test_closing_and_reads_keep_a_name_pending_without_a_resolver_choice():
    from ThoughtReferences import bindings, open_slots
    from part_fixtures import native_store
    from LTM1 import SYMBOL
    store, (name, predicate, answer) = native_store(points=torch.eye(4)[:3])
    store._concept_owner()._concept_allocator.singletons.add(answer)
    first = pending_chain(store, torch.eye(4)[0], name, (SYMBOL, answer), sentence=0)
    source = pending_chain(store, torch.eye(4)[0], name, (SYMBOL, answer), sentence=1)
    before = store.parts.refs.clone()
    for row in (first, source):
        meaning = store.meaning_of(row)
        assert open_slots(meaning) == (('referent', 0),)
        assert not bindings(meaning).get('_thought_witnesses')
        assert meaning.role_refs[2] == ('sym', answer)
    torch.testing.assert_close(store.parts.refs, before)


@pytest.mark.parametrize('relative', (False, True))
def test_forward_name_stays_pending_beside_a_witnessed_occurrence_complement(relative):
    from dataclasses import replace
    from Meaning import ConceptualMeaning
    from ThoughtReferences import bindings, open_slots
    from part_fixtures import native_store
    from LTM1 import OCCURRENCE, RELATION
    store, (name, predicate, answer) = native_store(points=torch.eye(4)[:3])
    if relative:
        complement = (RELATION, store.parts.add(predicate, answer))
    else:
        atom = replace(ConceptualMeaning.from_description(torch.eye(4)[2]),
                       role_refs=(('sym', answer), None, None))
        row = store.append_meaning(atom, evidence=(1., 0.), sentence_index=0, when=False)
        complement = (OCCURRENCE, int(store.address_keys[row]))
    store._concept_owner()._concept_allocator.singletons.add(complement)
    source = pending_chain(store, torch.eye(4)[0], name, complement, sentence=1)
    pending = pending_chain(store, torch.eye(4)[0], name, complement, sentence=2)
    from ChainRead import envelope
    for row in (source, pending):
        value = store.meaning_of(row)
        assert open_slots(value) == (('referent', 0),)
        assert bindings(value)['_forward_references'] == ((0, name),)
        assert value.role_refs[2] == envelope(store, complement)
        assert not bindings(value).get('_thought_witnesses')


def test_answer_read_checks_a_named_composite_before_descending_its_head(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from BindingAnswers import matches
    from Meaning import ConceptualMeaning
    f = SentenceFixture(monkeypatch)
    row = f.store.write_clause(f.clause(('verb', ('verb', 'content', 'ending'), 'outer')))
    reference = f.store.occurrence_of(row)
    named = f.store.raw_meaning_of(row).role_refs[0][1]
    word = f.cs.interpret.lookup_word([83], [], form='bound-name')
    f.cs.interpret.define(word, named)
    value = replace(ConceptualMeaning.from_description(f.store.slots[row]),
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
    value = replace(ConceptualMeaning.from_description(f.store.slots[row]),
        role_refs=(reference, None, None), bindings={'_bound_roles': (('referent', 0),)})
    model = SimpleNamespace(_concept_owner=lambda:f.cs,
                            symbolSpace=SimpleNamespace(ltm_store=f.store))
    assert matches(model, value, 'ending')
