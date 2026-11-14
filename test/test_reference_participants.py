"""Only the kept parse's participant constituents can enter a bind menu."""
from order_fixtures import force_order
from chunk_fixtures import context as chunk_context
from dataclasses import replace
from types import SimpleNamespace
import weakref

import pytest
import torch

from ClauseRow import Clause, Composition
from Meaning import ConceptualMeaning
from ReferenceContext import SituationFrame
from test_medium_term_memory import keep, snapshot, unchanged
from test_reference_arrivals import world, named, described


@pytest.mark.parametrize('attachment', ('surface', 'predicate', 'punctuation'))
@pytest.mark.parametrize('lexical_copy', (False, True))
def test_positive_marker_or_predicate_is_never_an_arrival_candidate(monkeypatch, attachment, lexical_copy):
    resolver, store, atoms, words, _ = world()
    pending, _ = keep(resolver, named(atoms[0], words['x']))
    noun = Composition(atoms[0], reference=words['three'], word_id=words['three'], leaves=(2,))
    from test_reference_close import known_concept as mint
    marker_reference = mint(resolver.components.owner, atoms[1]) if lexical_copy else words['wrong']
    marker = Composition(atoms[1], reference=marker_reference, word_id=words['wrong'], leaves=(1,))
    x = Composition(atoms[0]*0, reference=words['x'], word_id=words['x'], leaves=(0,))
    if attachment == 'predicate':
        phrase = Composition(atoms[0], (marker, noun), operation='verb', head_role=1)
        tree = Composition(atoms[0], (x, phrase), operation='lift', head_role=2)
    else:
        phrase = Composition(atoms[0], (marker, noun) if attachment == 'surface' else (noun, marker),
            operation='surface' if attachment == 'surface' else 'verb',
            head_role=2 if attachment == 'surface' else 1)
        tree = Composition(atoms[0], (x, phrase), operation='equal')
    # Even an explicit provisional word request at this leaf cannot turn
    # its surface/predicate position into an object.
    wrong = replace(named(atoms[1], words['wrong']), composition=marker)
    clause = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0],
        composition=tree, children=(wrong,))
    def prediction(identity, predicate, role):
        return predicate.new_tensor(100. if resolver._derivation_symbol(identity)
                                    in (marker_reference, words['wrong']) else 1.)
    monkeypatch.setattr(resolver, 'prediction', prediction)
    assert float(resolver.prediction(('sym', words['wrong']), atoms[0], 0)) > 0
    held = snapshot(resolver)
    names, candidates = resolver._arrival_constituents(clause)
    assert words['wrong'] not in names
    assert candidates and all(node.subject_word_id != words['wrong'] for node in candidates)
    assert all(node.composition.reference not in (marker_reference, words['wrong']) for node in candidates)
    plan = resolver.preview(clause, chunk_context(resolver, (SituationFrame.from_row(store.row(pending)),)))
    assert all(item['node'].reference_request.get('forward_identity') != words['wrong']
               for item in plan['items'])
    assert plan['arrivals']
    for item in plan['arrivals']:
        for key in item['candidates']:
            if key[:1] == ('arrival',):
                assert candidates[key[1]].composition.reference not in (marker_reference, words['wrong'])
    unchanged(resolver, held)


def test_an_objectless_marker_leaf_without_a_participant_position_has_no_site():
    resolver, store, atoms, words, _ = world()
    leaf = named(atoms[1], words['wrong'])
    leaf = replace(leaf, reference_request=dict(leaf.reference_request, leaf=True))
    held = snapshot(resolver)
    plan = resolver.preview(leaf, chunk_context(resolver, ()))
    assert not plan['items'] and resolver._arrival_constituents(leaf) == (set(), [])
    assert not resolver.components.owner.mtm.entries
    unchanged(resolver, held)


def test_the_journal_never_adds_a_standalone_provisional_word_to_the_object_menu(monkeypatch):
    from ClauseJournal import finish_clause
    from reading_fixtures import record_reading
    from test_bind_thought import native
    f = native(monkeypatch)
    word = f.cs.interpret.lookup_word([101], (), form='m')
    f.words['m'] = word, word
    program = replace(f.program('m'), close_references=True,
                      leaf_orders=torch.zeros(1, dtype=torch.long))
    clause = finish_clause(f.language, record_reading(f.language, program), registry=f.registry)
    assert clause.reference_request is None and clause.order == 0
    row, plan = keep(f.resolver, clause)
    assert not plan['items'] and not f.cs.mtm.entries
    held = snapshot(f.resolver)
    frame = SituationFrame.from_row(f.store.row(row))
    assert not f.resolver.accessible(chunk_context(f.resolver, (frame,)))
    assert not f.resolver.preview(named(f.atoms[0], word), chunk_context(f.resolver, (frame,)))['items'][0]['candidates']
    unchanged(f.resolver, held)


def test_ordinary_menu_uses_the_retained_argument_occupations(monkeypatch):
    from PartsOfSpeech import PartsOfSpeech
    resolver, store, atoms, words, _ = world()
    owner = resolver.components.owner
    rule = SimpleNamespace(method_name='lift', surface_name=None, head_role=2,
                           rhs_symbols=('NP', 'VP'), clause_form='S', relation_kind=None)
    from test_reference_close import known_concept as mint
    words = dict(three=mint(owner, atoms[0]), wrong=mint(owner, atoms[1]))
    occupations = PartsOfSpeech(('lift_argin1', 'lift_argin2'), owner)
    labels = {name: owner.interpret.lookup_word([120+i], (), form=name)
              for i, name in enumerate(words)}
    for name, word in labels.items():
        owner.interpret.define(word, words[name])
    occupations.add(labels['three'], 'lift_argin1')
    occupations.add(labels['wrong'], 'lift_argin2')
    object.__setattr__(store, '_occupation_owner', weakref.ref(occupations))
    object.__setattr__(store, '_part_read_rules', (rule,))
    # Explicitly minted native identities; the predicate position in this
    # admitted pair does not make it a participant.
    tree = Composition(atoms[0], (
        Composition(atoms[0], reference=words['three']),
        Composition(atoms[1], reference=words['wrong'])), operation='lift', head_role=2)
    written = {}
    clause = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0], composition=tree)
    row = store.write_clause(clause,  written_rows=written)
    frame = replace(SituationFrame.from_row(store.row(row)),
                    symbol_ids=written['symbols', id(clause)])
    # The ordinary field reads an identity's retained role history. Writing
    # a resolved expression alone no longer creates an accessible individual.
    occurrence = store.occurrence_of(row)
    pool = resolver.components.owner.mtm
    pool.open(occurrence,kind=words['three'],native=words['three'],when=0,value=atoms[0])
    pool.record_link(occurrence, occurrence, None, kind='reference', order=0)
    pool.event(occurrence,occurrence,verb=words['wrong'],role=0,when=0)
    held = snapshot(resolver)
    records = resolver.accessible(chunk_context(resolver, (frame,)))
    identities = resolver.individuals(records)
    assert words['three'] in identities and words['wrong'] not in identities
    unchanged(resolver, held)


@pytest.mark.parametrize('position', ('participant', 'marker', 'predicate'))
@pytest.mark.parametrize('order', (0, 1))
def test_journal_and_close_use_the_reading_order_and_argument_position(monkeypatch, order, position):
    from ClauseJournal import finish_clause
    from reading_fixtures import record_reading
    from Spaces import _concept_alloc_of
    from test_bind_thought import native
    f = native(monkeypatch)
    word = f.cs.interpret.lookup_word([101], (), form='m')
    force_order(f.cs, word, order)
    f.words['m'] = word, word
    tree = (('lift', 'm', 'v') if position == 'participant' else
            ('surface', 'm', 'v') if position == 'marker' else ('lift', 'v', 'm'))
    program = f.program(tree)
    orders = torch.zeros(2, dtype=torch.long)
    orders[program.word_ids == word] = f.cs._concept_source_order(word)
    program = replace(program, leaf_orders=orders, close_references=True)
    clause = finish_clause(f.language, record_reading(f.language, program), registry=f.registry)
    expected = int(order == 1 and position == 'participant')
    held = snapshot(f.resolver)
    plan = f.resolver.preview(clause, chunk_context(f.resolver, ()))
    assert len(plan['items']) == expected
    if expected:
        node = plan['items'][0]['node']
        assert node.order == order and node.reference_request['mode'] == 'bind'
        torch.testing.assert_close(node.point, program.leaves[0], rtol=0, atol=0)
    unchanged(f.resolver, held)
    _, committed = keep(f.resolver, clause)
    assert len(committed['items']) == expected
    assert sum(entry.refinement.get('member_kind') != 'event'
               for entry in f.cs.mtm.entries.values()) == expected


def test_bind_wrapper_keeps_the_verb_participant_and_excludes_its_predicate():
    from ReferenceParticipants import participant_terms
    predicate = Composition(None, word_id=101)
    participant = Composition(None, word_id=102)
    phrase = Composition(None, (predicate, participant), operation='verb', head_role=1)
    root = Composition(None, (phrase,), operation='bind', head_role=1)
    eligible, _ = participant_terms(((root, False),))
    assert id(participant) in eligible
    assert id(predicate) not in eligible


def test_a_pending_lexical_site_cannot_bind_to_its_enclosing_introduction(monkeypatch):
    resolver, store, atoms, words, _ = world(pending_atom=0)
    word = words['x']
    earlier, _ = keep(resolver, described(atoms[0], word))
    frame = SituationFrame.from_row(store.row(earlier))
    point = atoms[0]
    leaf = Composition(point, reference=word, word_id=word, leaves=(0,), order=1)
    phrase = Composition(point, (leaf,), operation='compound_unary', head_role=1,
                         leaves=(0,), order=1)
    clause = Clause(ConceptualMeaning.from_description(point), point=point, order=1,
                    composition=phrase, subject_word_id=word)
    # The enclosing phrase takes its own object. At the lexical site choose
    # that introduction if offered, otherwise the earlier pending occurrence.
    def select(choices, logits, *, forward_identity=None, individuals=None):
        if forward_identity != word:
            return None
        return choices.index(('new', 1)) if ('new', 1) in choices else 0

    monkeypatch.setattr(resolver, 'select', select)
    _, plan = keep(resolver, clause, (frame,), position=1)
    # Operand preparation now precedes its enclosing introduction.
    site, intro = plan['items']
    assert intro['own_object'] and intro['selected'] == ('new', 1)
    assert site['lexical'] and ('new', 1) not in site['candidates']
    assert site['selected'] in site['candidates']
    assert resolver.components.owner.mtm.token_of(site['reference']) == (
        resolver.components.owner.mtm.token_of(store.occurrence_of(earlier)))


def test_two_retained_roles_can_reuse_the_same_leaf_introduction():
    resolver, _, atoms, words, _ = world()
    leaf = named(atoms[0], words['x'])
    first = replace(leaf, identity=dict(handle=1, action=None), reference_request=dict(
        leaf.reference_request, mode='bind', addressed_introduction=True, compose_bind=0))
    second = replace(first, reference_request=dict(first.reference_request, role=2))
    clause = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0],
                    children=(first, second), refs=(('clause', 0), -1, ('clause', 1)))
    plan = resolver.preview(clause, chunk_context(resolver, ()))
    prepared = resolver.prepare(clause, plan)
    first_item, second_item = plan['items']
    assert first_item['own_object'] and not second_item['own_object']
    assert first_item['selected'] == second_item['selected'] == ('new', 0)
    term = prepared.children[1].composition
    assert term is prepared.children[0].composition
    assert term.binding is not term and term.binding.children == ()
