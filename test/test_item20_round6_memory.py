"""Retained reference paths remain typed when later input changes a reading."""
from dataclasses import replace
from types import SimpleNamespace as NS
import torch
from nomination_fixtures import recurrent_word


def test_recorded_compose_bind_uses_its_selected_introductions_order():
    from test_reference_close import fixture, known_concept
    from test_medium_term_memory import mention
    from chunk_fixtures import context
    from ClauseRow import Clause
    from Meaning import ConceptualMeaning
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    higher = known_concept(owner, atoms[0], order=1)
    identity = dict(handle=1, origins=torch.zeros(1, 3, dtype=torch.long))
    first = mention(atoms[0], mode='bind')
    first = replace(first, identity=identity,
        reference_request=dict(first.reference_request, compose_bind=0, sign_leaf=True))
    alias = mention(atoms[0], mode='bind')
    alias = replace(alias, subject_word_id=higher, identity=identity,
        reference_request=dict(alias.reference_request, compose_bind=1, forward_identity=higher))
    root = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0],
        children=(first, alias), refs=(('clause', 0), -1, ('clause', 1)))
    plan = resolver.preview(root, context(resolver, ()))
    assert owner._concept_source_order(higher) == 1
    assert [item['selected'] for item in plan['items']] == [('new', 0), ('new', 0)]
    assert [item['member_order'] for item in plan['items']] == [0, 0]
    kept = resolver.prepare(root, plan)
    written = {}
    row = store.write_clause(kept, written_rows=written)
    resolver.commit(plan, row, written, document='recorded-order', position=0, training=False)
    references = tuple(plan['mention_occurrences'].values())
    assert len(references) == 2
    key = owner.mtm.token_of(references[0])
    assert owner.mtm.token_of(references[1]) == key
    assert owner.mtm.entries[key].refinement['member_order'] == 0
