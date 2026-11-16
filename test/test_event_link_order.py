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
