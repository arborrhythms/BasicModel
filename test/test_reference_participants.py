"""Only the kept parse's participant constituents can enter a bind menu."""
from dataclasses import replace
from types import SimpleNamespace
import weakref

import pytest
import torch

from ClauseRow import Clause, Composition
from Meaning import ConceptualMeaning
from ReferenceContext import SituationFrame
from test_medium_term_memory import keep, snapshot, unchanged
from test_reference_arrivals import world, named


@pytest.mark.parametrize('attachment', ('surface', 'predicate', 'punctuation'))
@pytest.mark.parametrize('lexical_copy', (False, True))
def test_positive_marker_or_predicate_is_never_an_arrival_candidate(monkeypatch, attachment, lexical_copy):
    resolver, store, atoms, words, _ = world()
    pending, _ = keep(resolver, named(atoms[0], words['x']))
    noun = Composition(atoms[0], reference=words['three'], word_id=words['three'], leaves=(2,))
    from IdentityRows import mint
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
    plan = resolver.preview(clause, (SituationFrame.from_row(store.row(pending)),))
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
    plan = resolver.preview(leaf, ())
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
    assert not f.resolver.accessible((frame,))
    assert not f.resolver.preview(named(f.atoms[0], word), (frame,))['items'][0]['candidates']
    unchanged(f.resolver, held)


def test_ordinary_menu_uses_the_retained_argument_occupations(monkeypatch):
    from PartsOfSpeech import PartsOfSpeech
    resolver, store, atoms, words, _ = world()
    owner = resolver.components.owner
    rule = SimpleNamespace(method_name='lift', surface_name=None, head_role=2,
                           rhs_symbols=('NP', 'VP'), clause_form='S', relation_kind=None)
    from IdentityRows import mint
    words = dict(three=mint(owner, atoms[0]), wrong=mint(owner, atoms[1]))
    occupations = PartsOfSpeech(('lift_argin1', 'lift_argin2'))
    occupations.add(words['three'], 'lift_argin1')
    occupations.add(words['wrong'], 'lift_argin2')
    object.__setattr__(store, '_occupation_owner', weakref.ref(occupations))
    object.__setattr__(store, '_part_read_rules', (rule,))
    # Explicitly minted native identities; the predicate position in this
    # admitted pair does not make it a participant.
    tree = Composition(atoms[0], (
        Composition(atoms[0], reference=words['three']),
        Composition(atoms[1], reference=words['wrong'])), operation='lift', head_role=2)
    row = store.write_clause(Clause(ConceptualMeaning.from_description(atoms[0]),
        point=atoms[0], composition=tree), document_key='participants')
    frame = replace(SituationFrame.from_row(store.row(row)),
                    symbol_ids=(words['three'], words['wrong']))
    held = snapshot(resolver)
    records = resolver.accessible((frame,))
    assert words['three'] in records and words['wrong'] not in records
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
    _concept_alloc_of(f.cs).reference_orders[word] = order
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
    plan = f.resolver.preview(clause, ())
    assert len(plan['items']) == expected
    if expected:
        node = plan['items'][0]['node']
        assert node.order == order and node.reference_request['mode'] == 'bind'
        torch.testing.assert_close(node.point, program.leaves[0], rtol=0, atol=0)
    unchanged(f.resolver, held)
    _, committed = keep(f.resolver, clause)
    assert len(committed['items']) == expected
    assert len(f.cs.mtm.entries) == expected


def test_retained_argument_classification_uses_only_its_own_occupations(monkeypatch):
    from PartRead import argument_reader
    from Spaces import _concept_alloc_of
    from test_bind_thought import native
    f = native(monkeypatch)
    marker, head = f.noun('m'), f.noun('n')
    occupations = f.language.parts_of_speech
    occupations.add(marker, 'surface_argin1')
    occupations.add(head, 'surface_argin2')
    _concept_alloc_of(f.cs).symbol_types.setdefault(marker, set()).add('object')
    def no_definition_read(*args, **kwargs):
        raise AssertionError('argument classification read lexical definitions')
    monkeypatch.setattr(type(f.cs.definitions), 'words', no_definition_read)
    operation, is_marker = argument_reader(f.store)
    assert is_marker(marker, set())
    assert operation((marker, head)).method_name == 'surface'
    word = f.words['m'][0]
    assert not is_marker(word, set())
    assert operation((word, head)) is None
