"""Bind owns anonymous names and their refinements; retained equalities do not."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from ClauseRow import Clause, Composition, ClausePredicate, predicate_point
from Meaning import ConceptualMeaning
from ReferenceContext import SituationFrame
from ThoughtReferences import bindings, open_slots, with_slots
from test_medium_term_memory import keep, snapshot, unchanged
from test_reference_close import fixture


def world():
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    words = {word: owner.interpret.lookup_word([unit], [], form=word)
             for word, unit in (('x', 120), ('three', 116), ('wrong', 119))}
    with torch.no_grad():
        for word, atom in (('three', atoms[0]), ('wrong', atoms[1])):
            owner.similarity_codebook.W[owner._csw_row_of(words[word])].copy_(atom)
    model = SimpleNamespace(_concept_owner=lambda: owner,
                            symbolSpace=SimpleNamespace(ltm_store=store))
    return resolver, store, atoms, words, model


def named(point, word):
    return Clause(ConceptualMeaning.from_description(point*0), point=point*0, order=1,
        composition=Composition(point*0, reference=word, word_id=word, leaves=(0,), order=1),
        reference_request=dict(mode='pronoun', forward_identity=word, leaves=(0,),
            cue=point*0, content=point*0, predicate=point*0, role=0))


def equality(point, word, complement, *, witness=1.):
    child = Clause(ConceptualMeaning.from_description(point), point=point,
        composition=Composition(point, reference=complement, word_id=complement, leaves=(1,)))
    pred = predicate_point('equal', point)
    value = ConceptualMeaning(torch.stack((point*0, pred, point)),
        torch.ones(3, dtype=torch.bool), bindings={'_equality': True,
            '_forward_references': ((0, word),),
            '_reference_sites': ((0, dict(role=0, leaves=(0,), cue=(point*0).tolist(),
                                          content=(point*0).tolist(), predicate=pred.tolist())),)})
    value = with_slots(value, (('referent', 0),), pair=(witness, 0.))
    # Supply the named participant's reading, as the journal does. Its
    # grammatical position and order license recovery of the open site.
    source = Composition(point*0, reference=word, word_id=word, leaves=(0,), order=1)
    composition = Composition(point, (source, child.composition), operation='equal')
    return Clause(value, relation='equal', refs=(-1, ClausePredicate(pred, 'equal'), ('clause', 0)),
                  children=(child,), evidence=(witness, 0.), composition=composition)


def force_named(resolver, monkeypatch, target):
    """Choose through the resolver seam, preserving score and choice_cost."""
    original = resolver.select
    def select(choices, logits, *, forward_identity=None, individuals=None):
        if forward_identity is None:
            return original(choices, logits, individuals=individuals)
        return next((i for i, key in enumerate(choices)
                     if individuals[key].get('kind', resolver._derivation_symbol(key)) == target), None)
    monkeypatch.setattr(resolver, 'select', select)


def test_named_first_bind_opens_anonymous_without_row_or_bank_writes():
    resolver, store, a, words, _ = world()
    row, plan = keep(resolver, named(a[0], words['x']))
    token = store.occurrence_of(row)
    pool = resolver.components.owner.mtm
    assert plan['items'][0]['selected'] == ('new', 0)
    assert plan['items'][0]['reference'] == token
    assert resolver.last_audit[0]['missing_reason'] is None
    assert pool.entries[token].kind is None
    assert pool.entries[token].occurrences == (token,) and pool.entries[token].count == 1
    assert not store.raw_meaning_of(row).roles.any()
    assert store.raw_meaning_of(row).role_refs == (None, None, None)
    assert not resolver.token_value(token).any()
    held = snapshot(resolver)
    groups = resolver.individuals(resolver.accessible((SituationFrame.from_row(store.row(row)),)))
    assert not groups[token]['value'].any()
    unchanged(resolver, held)


def test_arrival_bind_refines_and_reads_three_through_the_token(monkeypatch):
    from BindingAnswers import matches
    from PartRead import expression_components
    from QueryWork import QueryWorkBudget
    resolver, store, a, words, model = world()
    force_named(resolver, monkeypatch, words['three'])
    pending, _ = keep(resolver, named(a[0], words['x']))
    token = store.occurrence_of(pending)
    frame = SituationFrame.from_row(store.row(pending))
    raw, refs = store.slots.clone(), store.refs.clone()
    arrival, plan = keep(resolver, equality(a[0], words['x'], words['three']), (frame,), position=1)
    entry = resolver.components.owner.mtm.entries[token]
    assert entry.kind == words['three'] and entry.count == 3
    assert sum(item['token'] == token for item in plan['arrivals']) == 1
    assert all(item['features'].shape[-1] == 8 for item in plan['arrivals'])
    assert plan['cost'].requires_grad
    value = store.meaning_of(pending)
    assert not open_slots(value) and not bindings(value).get('_forward_references')
    assert matches(model, value, 'three')
    parent = store.meaning_of(arrival)
    assert not open_slots(parent) and not bindings(parent).get('_forward_references')
    assert matches(model, parent, 'three')
    assert resolver._derivation_symbol(token) == words['three']
    torch.testing.assert_close(resolver.token_value(token), a[0])
    groups = resolver.individuals(resolver.accessible((frame,)))
    torch.testing.assert_close(groups[token]['value'], a[0])
    terms, _, incomplete = expression_components(store, (token,), max_records=8, work=QueryWorkBudget(8))
    assert not incomplete and terms[token]['head_reference'] == ('sym', words['three'])
    torch.testing.assert_close(store.slots[:pending+1], raw[:pending+1], rtol=0, atol=0)
    torch.testing.assert_close(store.refs[:pending+1], refs[:pending+1], rtol=0, atol=0)


@pytest.mark.parametrize('when', ('before', 'after'))
def test_kept_wrong_equality_neither_binds_nor_reopens_a_token(monkeypatch, when):
    from BindingAnswers import matches
    resolver, store, a, words, model = world()
    force_named(resolver, monkeypatch, words['three'])
    pending, _ = keep(resolver, named(a[0], words['x']))
    token = store.occurrence_of(pending)
    frame = SituationFrame.from_row(store.row(pending))
    def wrong():
        before = dict(resolver.components.owner.mtm.entries)
        store.write_clause(equality(a[1], words['x'], words['wrong']),
                           document_key='wrong', sentence_index=0)
        value = store.meaning_of(pending)
        assert resolver.components.owner.mtm.entries == before
        return value
    if when == 'before':
        assert open_slots(wrong())
        assert resolver.components.owner.mtm.entries[token].kind is None
    keep(resolver, equality(a[0], words['x'], words['three']), (frame,), position=1)
    if when == 'after':
        assert matches(model, wrong(), 'three')
    assert matches(model, store.meaning_of(pending), 'three')
    assert not open_slots(store.meaning_of(pending))


def test_gap_without_a_word_and_candidate_stays_open():
    resolver, store, a, _, _ = world()
    site = named(a[0], 1)
    request = dict(site.reference_request)
    request.pop('forward_identity')
    row, plan = keep(resolver, replace(site, reference_request=request))
    assert plan['items'][0]['reference'] is None
    assert resolver.last_audit[0]['missing_reason'] == 'no_accessible_individual'
    assert ('referent', 0) in open_slots(store.meaning_of(row))
    assert not resolver.components.owner.mtm.entries


def test_unary_bind_keeps_the_objectless_word_site(monkeypatch):
    from ClauseJournal import finish_clause
    from reading_fixtures import record_reading
    from test_bind_thought import native
    f = native(monkeypatch)
    word = f.cs.interpret.lookup_word([101], (), form='m')
    from Spaces import _concept_alloc_of
    _concept_alloc_of(f.cs).reference_orders[word] = 1
    f.words['m'] = word, word
    program = replace(f.program(('definite_unary', 'm')), close_references=True,
                      leaf_orders=torch.tensor([f.cs._concept_source_order(word)]))
    clause = finish_clause(f.language, record_reading(f.language, program), registry=f.registry)
    assert clause.reference_request['forward_identity'] == word
    assert clause.reference_request['mode'] == 'bind'
    torch.testing.assert_close(clause.point, program.leaves[0], rtol=0, atol=0)


def test_bound_use_does_not_assign_a_kind_before_arrival(monkeypatch):
    resolver, store, a, words, model = world()
    pending, _ = keep(resolver, named(a[0], words['x']))
    token = store.occurrence_of(pending)
    frame = SituationFrame.from_row(store.row(pending))
    site = replace(named(a[0], words['x']), subject_word_id=words['x'],
        reference_request=dict(mode='bind', leaves=(0,), content=a[1], cue=a[1], predicate=a[1]*0),
        composition=Composition(a[1], (Composition(a[1], reference=words['x']),),
                                operation='definite_unary', head_role=1, order=1))
    # A positive prime makes this a selected bind under the zero-logit dummy.
    monkeypatch.setattr(resolver, 'prime', lambda signature, stream: signature.new_tensor(1.))
    used, plan = keep(resolver, site, (frame,), position=1)
    assert plan['items'][0]['reference'] == token
    assert resolver.components.owner.mtm.entries[token].kind is None
    assert store.raw_meaning_of(used).role_refs[0] is None
    force_named(resolver, monkeypatch, words['three'])
    _, refined = keep(resolver, equality(a[0], words['x'], words['three']), (frame,), position=2)
    assert sum(item['token'] == token for item in refined['arrivals']) == 1
    from BindingAnswers import matches
    assert matches(model, store.meaning_of(used), 'three')


def test_anonymous_parent_keeps_its_raw_open_role_and_query_predicate(monkeypatch):
    from ThoughtClosing import closing_question
    resolver, store, a, words, model = world()
    force_named(resolver, monkeypatch, None)
    clause = equality(a[0], words['x'], words['three'])
    plan = resolver.preview(clause, ())
    prepared = resolver.prepare(clause, plan)
    written = {}
    row = store.write_clause(prepared, document_key='parent', written_rows=written)
    resolver.commit(plan, row, written, document='parent', position=0, training=False)
    raw = store.raw_meaning_of(row)
    assert raw.role_refs[0][0] == 'ltm' and open_slots(raw) == (('referent', 0),)
    query = closing_question(store.meaning_of(row), prepared, store=store, written_rows=written)
    assert bindings(query)['_query_occurrence'] == store.occurrence_of(row)
    torch.testing.assert_close(query.roles[1], raw.roles[1])


@pytest.mark.parametrize('compiled', (False, True))
def test_named_trial_plans_only_refine_the_kept_close(monkeypatch, compiled):
    from Language import LowerLayer
    resolver, store, a, words, _ = world()
    force_named(resolver, monkeypatch, words['three'])
    pending, _ = keep(resolver, named(a[0], words['x']))
    pool, token = resolver.components.owner.mtm, store.occurrence_of(pending)
    frame = SituationFrame.from_row(store.row(pending))
    forward = LowerLayer(12, 12).compose
    if compiled:
        forward = torch.compile(forward, backend='eager', fullgraph=True)
    before, retained = dict(pool.entries), snapshot(resolver)
    trials = []
    for _ in range(2):
        point = forward(a[1][None], a[0][None])[0]
        clause = equality(point, words['x'], words['three'])
        plan = resolver.preview(clause, (frame,))
        trials.append((resolver.prepare(clause, plan), plan))
        assert pool.entries == before
        unchanged(resolver, retained)
    kept, plan = trials[0]
    written = {}
    row = store.write_clause(kept, written_rows=written)
    assert pool.entries == before
    retained = snapshot(resolver)
    resolver.commit(plan, row, written, document='trial', position=1, training=False)
    unchanged(resolver, retained)
    assert pool.entries[token].count == 3 and pool.entries[token].kind == words['three']


def test_named_root_can_retain_a_selected_lexical_constituent(monkeypatch):
    resolver, store, a, words, _ = world()
    force_named(resolver, monkeypatch, words['three'])
    leaf = Composition(a[0], reference=words['three'], word_id=words['three'], leaves=(0,))
    clause = replace(named(a[0], words['three']),
        composition=Composition(a[0]*0, (leaf,), operation='definite_unary', head_role=1, order=1))
    row, _ = keep(resolver, clause)
    assert resolver.components.owner.mtm.entries[store.occurrence_of(row)].kind == words['three']


def test_named_bind_to_a_native_individual_creates_no_pool_entry(monkeypatch):
    from IdentityRows import mint
    resolver, store, a, words, _ = world()
    native = mint(resolver.components.owner, a[0])
    frame = SituationFrame(1, torch.stack((a[0], a[0]*0, a[0]*0)),
        torch.tensor([True, False, False]), -1, a[0], symbol_ids=(native,))
    monkeypatch.setattr(resolver, 'prime', lambda signature, stream: signature.new_tensor(1.))
    row, plan = keep(resolver, named(a[0], words['x']), (frame,))
    assert plan['items'][0]['reference'] == native
    value = store.meaning_of(row)
    assert value.role_refs[0] == ('sym', native) and not open_slots(value)
    assert not bindings(value).get('_forward_references')
    assert not resolver.components.owner.mtm.entries


def test_named_bind_to_a_retained_unpooled_occurrence_keeps_its_kind(monkeypatch):
    resolver, store, a, words, model = world()
    antecedent = store.write_clause(Clause(ConceptualMeaning.from_description(a[0]),
        point=a[0], order=1, composition=Composition(a[0], reference=words['three'])))
    frame = SituationFrame.from_row(store.row(antecedent))
    monkeypatch.setattr(resolver, 'prime', lambda signature, stream: signature.new_tensor(1.))
    row, plan = keep(resolver, named(a[0], words['x']), (frame,), position=1)
    pool = resolver.components.owner.mtm
    token = pool.token_of(store.occurrence_of(row))
    assert plan['items'][0]['reference'] == store.occurrence_of(antecedent)
    assert token == store.occurrence_of(row) and pool.entries[token].kind == words['three']
    from BindingAnswers import matches
    assert matches(model, store.meaning_of(row), 'three')


@pytest.mark.parametrize('retained_occurrence', (False, True))
def test_native_arrival_refines_an_existing_anonymous_token(monkeypatch, retained_occurrence):
    from IdentityRows import mint
    resolver, store, a, words, _ = world()
    pending, _ = keep(resolver, named(a[0], words['x']))
    token = store.occurrence_of(pending)
    pool = resolver.components.owner.mtm
    native = mint(resolver.components.owner, a[0])
    force_named(resolver, monkeypatch, native)
    if retained_occurrence:
        row = store.write_clause(Clause(ConceptualMeaning.from_description(a[0]),
            point=a[0], order=1, composition=Composition(a[0], reference=native)))
        occurrence = store.occurrence_of(row)
        frame = replace(SituationFrame.from_row(store.row(row)), symbol_ids=(native,),
                        activations=((occurrence, native, 0),))
    else:
        frame = SituationFrame(1, torch.stack((a[0], a[0]*0, a[0]*0)),
            torch.tensor([True, False, False]), -1, a[0], symbol_ids=(native,))
    _, plan = keep(resolver, equality(a[1], words['x'], words['wrong']),
                   (SituationFrame.from_row(store.row(pending)), frame), position=1)
    refinement = next(item for item in plan['arrivals'] if item['token'] == token)
    assert refinement['selected'] == native
    assert pool.entries[token].kind == native and pool.entries[token].count == 2
    assert pool.entries[token].occurrences[-1] == refinement['reference']
    if retained_occurrence:
        assert refinement['reference'] == occurrence
    assert set(pool.entries) == {token}
    assert resolver._derivation_symbol(token) == native
    assert not open_slots(store.meaning_of(pending))


def test_cached_word_mention_still_binds_to_its_accessible_token(monkeypatch):
    from BindingAnswers import matches
    resolver, store, a, words, model = world()
    force_named(resolver, monkeypatch, words['three'])
    pending, _ = keep(resolver, named(a[0], words['x']))
    token = store.occurrence_of(pending)
    frame = SituationFrame.from_row(store.row(pending))
    keep(resolver, equality(a[0], words['x'], words['three']), (frame,), position=1)
    # The cached concept carries no provisional request. Its order-one
    # participant reading still owns this occurrence's bind to the token.
    with torch.no_grad():
        owner = resolver.components.owner
        owner.similarity_codebook.W[owner._csw_row_of(words['x'])].copy_(a[1])
    cached = Clause(ConceptualMeaning.from_description(a[1]), point=a[1], order=1,
        composition=Composition(a[1], reference=words['wrong'], word_id=words['x'], leaves=(0,), order=1))
    row, plan = keep(resolver, cached, (frame,), position=2)
    assert len(plan['items']) == 1 and plan['items'][0]['lexical']
    assert plan['items'][0]['reference'] == token
    assert resolver._derivation_symbol(store.occurrence_of(row)) == words['three']
    used = next(item['occurrence'] for item in resolver.last_audit if not item.get('refinement'))
    assert matches(model, store.meaning_of(store.index_of_row(used)), 'three')
    pool = resolver.components.owner.mtm
    # An arrival bind can itself reassign the selected introduction. This use
    # stays attached to that exact antecedent through the reassignment.
    assert pool.token_of(('ltm', store.address_domain, used)) == pool.token_of(token)
