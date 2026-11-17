"""An event departure reads the same member order as its retained occurrence."""
from dataclasses import replace

import pytest
import torch


@pytest.mark.parametrize('separate', [False, True])
def test_event_menu_uses_retained_order_instead_of_grammar_order(separate):
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from MemoryLink import Occurrence, preview
    from test_reference_close import fixture, known_concept
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    identity = known_concept(owner, atoms[0], order=1)
    result = atoms[1] if separate else atoms[0]
    expected_order = 0 if separate else 1
    existing = replace(ConceptualMeaning.from_description(result),
        role_refs=(None, None, None) if separate else (('sym', identity), None, None))
    row = store.append_meaning(existing)
    collection = owner.link(Occurrence(store.occurrence_of(row), kind='event', selected=None), {},
        value=result, where=None, when=1, stream=0)
    clause = Clause(replace(ConceptualMeaning.from_description(atoms[0]),
        role_refs=(('sym', identity), None, None)), point=atoms[0], order=1-expected_order,
        composition=Composition(atoms[0], reference=identity, order=1),
        identity=dict(event=(1, 1, 0, identity, 0, 0), action=0, result=result))
    resolver.select = lambda choices, logits, **kwargs: choices.index(collection) if collection in choices else None
    plan = dict(items=[], decisions=[], departure=-1, cost=atoms[0].new_zeros(()), stream=0)
    preview(resolver, plan, clause, None, stream=0, when=2)
    assert plan['link_choices']['event', 0] == collection
    prepared = resolver.prepare(clause, plan)
    written = {}
    store.write_clause(prepared, written_rows=written)
    address = written.get('root_event_occurrence') or written['root_occurrence']
    assert owner._concept_source_order(address) == expected_order
    owner.link(Occurrence(address, kind='event', selected=plan['link_choices']['event', 0]), {},
        value=result, where=None, when=2, stream=0)
    assert address in owner.mtm.entries[collection].occurrences


def test_event_menus_include_earlier_links_of_the_same_kept_close():
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from MemoryLink import Occurrence, preview
    from test_reference_close import fixture
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    original = .1*(atoms[0]+atoms[1])
    values = (.8*atoms[0]+.2*atoms[1], .2*atoms[0]+.8*atoms[1])
    row = store.append_meaning(ConceptualMeaning.from_description(original))
    collection = owner.link(Occurrence(store.occurrence_of(row), kind='event', selected=None), {},
        value=original, where=None, when=1, stream=0)
    def event(action, value):
        return Clause(ConceptualMeaning.from_description(value), point=value, order=0,
            composition=Composition(value, order=0),
            identity=dict(event=(1,1,0,1,0,0),action=action,result=value))
    clause = replace(event(0, values[0]), companions=(event(1, values[1]),))
    resolver.select = lambda choices, logits, **kwargs: choices.index(collection) if collection in choices else None
    plan = dict(items=[],decisions=[],departure=-1,cost=atoms[0].new_zeros(()),stream=0)
    preview(resolver,plan,clause,None,stream=0,when=2)
    assert plan['link_choices']['event',0] == collection
    assert plan['link_choices']['event',1] is None
    # Preview is a read; the prepared first link has not yet changed MTM.
    torch.testing.assert_close(owner.mtm.value_of(collection), original)
    for action,value in enumerate(values):
        row=store.append_meaning(ConceptualMeaning.from_description(value))
        target=owner.link(Occurrence(store.occurrence_of(row),kind='event',
            selected=plan['link_choices']['event',action]),{},value=value,where=None,when=2,stream=0)
        assert (target == collection) == (action == 0)
