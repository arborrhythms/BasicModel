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
