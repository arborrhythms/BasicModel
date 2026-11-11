"""Direct certificates for failures exposed by the end-of-item measurements."""
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
import torch


def test_empty_named_bind_keeps_its_reading_order_through_retention():
    from test_reference_close import fixture, known_concept
    from test_medium_term_memory import mention
    from chunk_fixtures import context
    from FamiliarTier import commit_read_words
    resolver, store, atoms = fixture()
    owner, pool = resolver.components.owner, resolver.components.owner.mtm
    word = owner.interpret.lookup_word([117], (), form='pending-name')
    concept = known_concept(owner, atoms[0], order=1)
    owner.interpret.define(word, concept, trust=.1)
    key = pool.named(word, value=atoms[0], when=0)
    assert not pool.entries[key].occurrences
    identity = dict(handle=1, origins=torch.tensor([[6, word, 0]]))
    clause = mention(atoms[0], mode='bind')
    clause = replace(clause, subject_word_id=word, identity=identity,
        reference_request=dict(clause.reference_request, compose_bind=1, record=atoms[0]))
    plan = resolver.preview(clause, context(resolver, ()), stream=0)
    assert plan['items'][0]['selected'] == key
    assert plan['items'][0]['member_order'] == 1
    kept = resolver.prepare(clause, plan)
    written = {}
    row = store.write_clause(kept, written_rows=written)
    reading = NS(word_ids=torch.tensor([word]), leaves=atoms[0][None], symbol_where=None)
    commit_read_words(owner, reading, batch=0, document='pending-name', position=0,
                      written={'root_occurrence': store.occurrence_of(row)})
    resolver.commit(plan, row, written, document='pending-name', position=0, training=False)
    assert pool.entries[pool.token_of(store.occurrence_of(row))].refinement['member_order'] == 1
    assert store.order_of_row(row) == 1


@pytest.mark.parametrize('source', ['members', 'occurrences', 'candidate'])
def test_memory_centroids_preserve_common_rails_with_fractional_evidence(source):
    from MemoryConsolidation import _fold
    from FamiliarTier import centroid
    weights = torch.tensor([3., 2.999999999301692, 2., 1., .9999999997671821,
                            .9999999997671778, .9999999997671775], dtype=torch.float64)
    codes = torch.cat((torch.ones(7, 176, dtype=torch.float64),
                       torch.zeros(7, 176, dtype=torch.float64)), dim=1)
    bank = NS(W=codes, lookup_rows=lambda rows: codes[rows])
    owner = NS(similarity_codebook=bank)
    members = dict(enumerate(weights.tolist()))
    if source == 'candidate':
        result = centroid(owner, members)
    else:
        store = NS(slots=codes, truth_plus=weights, truth_minus=torch.zeros_like(weights))
        result = _fold(owner, store, list(members), members=members if source == 'members' else None)
    assert torch.equal(result, codes[0])
    assert ((result >= 0) & (result <= 1)).all()


def test_promoted_name_keeps_its_order_when_an_old_mention_reads_the_definition():
    from test_reference_close import fixture, known_concept
    from FamiliarTier import commit_read_words, promote_identity
    from Memory import OCCURRENCE, SYMBOL
    resolver, store, atoms = fixture()
    owner, pool = resolver.components.owner, resolver.components.owner.mtm
    alloc = owner._concept_allocator
    word = owner.interpret.lookup_word([118], (), form='retained-definition')
    key = pool.named(word, value=atoms[0], when=0)
    row = store.append_idea(atoms[0], address=701, when=False)
    reading = NS(word_ids=torch.tensor([word]), leaves=atoms[0][None], symbol_where=None)
    occurrence = commit_read_words(owner, reading, batch=0, document='retained-definition',
        position=0, written={'root_occurrence':store.occurrence_of(row)})[0]
    assert store.order_of_row(store.location(occurrence)) == 0
    concept = promote_identity(owner, key, word=word, trust=1.)
    assert pool.entries[key].refinement['member_order'] == 0
    assert store.order_of_row(store.location(occurrence)) == 0
    edge = alloc.parts.find((OCCURRENCE, occurrence[-1]), (SYMBOL, concept))
    assert edge is not None
    # Learning the name supplies a denotation without changing its earlier
    # occurrence or deleting that occurrence's membership witness.
    assert alloc.settle(concept) == 1
    assert ('occ', occurrence[-1]) in alloc.refs(concept, 'part')
    assert alloc.parts.find((OCCURRENCE, occurrence[-1]), (SYMBOL, concept)) == edge
    assert ('part', ('sym', word)) in alloc.records(concept)
    # An inconsistent explicit definition still fails at the original guard.
    higher = known_concept(owner, atoms[1], order=1)
    alloc.store_of(concept).add_constituent(concept, 'part', ('sym', higher))
    with pytest.raises(ValueError, match='recorded concept order disagrees'):
        alloc.settle(concept)


@pytest.mark.parametrize('named', [False, True])
def test_eventful_identity_keeps_the_checked_thought_as_a_member(named):
    from test_reference_close import fixture, known_concept
    from MemoryLink import Occurrence
    from Meaning import ConceptualMeaning
    from ThoughtReferences import with_slots
    from ThoughtStream import write
    resolver, store, atoms = fixture()
    owner, pool = resolver.components.owner, resolver.components.owner.mtm
    concept = known_concept(owner, atoms[0], order=1)
    initial = replace(ConceptualMeaning.from_description(atoms[0]),
                      role_refs=(('sym', concept), None, None))
    first = store.occurrence_of(store.append_meaning(initial, order=1, when=False))
    if named:
        word = owner.interpret.lookup_word([119], (), form='thought-member')
        owner.interpret.define(word, concept, trust=.1)
        key = pool.named(word, value=atoms[0])
    else:
        word, key = None, None
    key = owner.link(Occurrence(first, concept=word, referent=concept, order=1, selected=key), {},
        value=atoms[0], where=None, when=1, stream=0)
    event_value = atoms[1]
    event = store.occurrence_of(store.append_idea(event_value, when=False))
    pool.event(key, event, verb=concept, role=0, when=1, before=atoms[0])
    pool.close(store, stream=0, references=(first, event))
    history = pool.entries[key].events
    answer = store.occurrence_of(store.append_idea(atoms[2], when=False))
    identity = ('name', word) if named else key
    meaning = ConceptualMeaning(torch.stack((event_value, torch.zeros_like(event_value), atoms[2])),
        torch.ones(3,dtype=torch.bool),role_refs=(identity,('sym',concept),answer),
        bindings={'_equality':True,'_producing_operation':'query','_bound_roles':(('referent',2),)},
        sentence_kind='relation')
    row = write(NS(symbolSpace=NS(ltm_store=store)), with_slots(meaning, (), pair=(1.,0.)), row=0)
    written = store.occurrence_of(row)
    assert not hasattr(pool, 'update')
    assert pool.token_of(written) == key
    assert written in pool.entries[key].occurrences
    assert pool.entries[key].refinement['mentions'][written] == (0,2)
    assert store.order_of_row(row) == pool.entries[key].refinement['member_order'] == 1
    assert pool.entries[key].events == history
    torch.testing.assert_close(pool.value_of(key), torch.maximum(event_value, atoms[2]), rtol=0, atol=0)


@pytest.mark.parametrize('flag', ['sign_leaf', 'no_op_leaf'])
@pytest.mark.parametrize('mode', ['mint', 'bind'])
def test_surface_only_introduction_uses_its_written_order_for_later_aliases(flag, mode):
    from test_reference_close import fixture, known_concept
    from test_medium_term_memory import mention
    from chunk_fixtures import context
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    word = owner.interpret.lookup_word([120], (), form='surface-only')
    concept = known_concept(owner, atoms[0], order=1)
    owner.interpret.define(word, concept)
    identity = dict(handle=1, origins=torch.zeros(1, 3, dtype=torch.long))
    point = torch.zeros_like(atoms[0])
    first = Clause(ConceptualMeaning.from_description(point), point=point, order=0,
        subject_word_id=word, composition=Composition(point, order=0, word_id=word),
        identity=identity, reference_request=dict(mode=mode, leaves=(0,),
            **{flag: True}, addressed_introduction=True,
            **({} if mode=='mint' else {'compose_bind':0})))
    alias = mention(atoms[0], mode='bind')
    alias = replace(alias, identity=identity,
        reference_request=dict(alias.reference_request, compose_bind=1))
    root = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0],
        children=(first,alias), refs=(('clause',0),-1,('clause',1)))
    plan = resolver.preview(root, context(resolver, ()))
    assert owner._concept_source_order(owner.interpret.referent(word)) == 1
    assert [item['selected'] for item in plan['items']] == [('new',0),('new',0)]
    assert [item['member_order'] for item in plan['items']] == [0,0]
    kept=resolver.prepare(root,plan)
    written={}
    row=store.write_clause(kept,written_rows=written)
    resolver.commit(plan,row,written,document='surface-only',position=0,training=False)
    references=tuple(plan['mention_occurrences'].values())
    assert len(references)==2
    key=owner.mtm.token_of(references[0])
    assert owner.mtm.token_of(references[1])==key
    assert owner.mtm.entries[key].refinement['member_order']==0
    assert all(store.order_of_row(store.location(ref))==0 for ref in references)
