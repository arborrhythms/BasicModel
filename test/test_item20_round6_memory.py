"""Retained reference paths remain typed when later input changes a reading."""
from dataclasses import replace
from types import SimpleNamespace as NS
import torch


def test_expired_pending_path_reenters_a_later_native_definition():
    from test_reference_close import fixture, known_concept
    from FamiliarTier import commit_read_words
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    pool = owner.mtm
    word = owner.interpret.lookup_word([115], (), form='later-empty')
    key = pool.named(word, value=atoms[0], when=0)
    reading = NS(word_ids=torch.tensor([word]), leaves=atoms[0][None], symbol_where=None)
    def close(position):
        root = store.append_idea(atoms[0])
        return commit_read_words(owner, reading, batch=0, document='later-empty',
            position=position, written={'root_occurrence': store.occurrence_of(root)})[0]
    close(0)
    pool.entries[key] = replace(pool.entries[key], occurrences=())
    concept = known_concept(owner, atoms[0], order=1)
    owner.interpret.define(word, concept, trust=.1)
    second = close(1)
    assert not pool.entries[key].occurrences
    assert pool.entries[key].refinement['member_order'] == 0
    new = pool.token_of(second)
    assert new != key and pool.entries[new].native == concept
    assert pool.entries[new].refinement['member_order'] == 1
    assert store.order_of_row(store.location(second)) == 1




def test_rereading_the_same_position_at_a_new_order_preserves_the_original_member():
    from test_reference_close import fixture, known_concept
    from FamiliarTier import commit_read_words
    resolver, store, atoms = fixture()
    owner, pool = resolver.components.owner, resolver.components.owner.mtm
    word = owner.interpret.lookup_word([116], (), form='later-order')
    key = pool.named(word, value=atoms[0], when=0)
    reading = NS(word_ids=torch.tensor([word]), leaves=atoms[0][None], symbol_where=None)
    root = store.append_idea(atoms[0], name=601)
    def close():
        return commit_read_words(owner, reading, batch=0, document='same-position',
            position=0, written={'root_occurrence': store.occurrence_of(root)})[0]
    first = close()
    first_bindings = store.raw_meaning_of(store.location(first)).bindings
    concept = known_concept(owner, atoms[0], order=1)
    owner.interpret.define(word, concept, trust=.1)
    second = close()
    assert first != second
    assert store.raw_meaning_of(store.location(first)).bindings == first_bindings
    assert pool.entries[key].refinement['member_order'] == 0
    assert store.order_of_row(store.location(second)) == 1
    assert pool.entries[key].occurrences == (first,)
    assert pool.entries[pool.token_of(second)].native == concept


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
