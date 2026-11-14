"""§54: a named forward referent is not supplied by its own mention.

These mechanism certificates use native word definitions, occurrence rows,
MTM operations and the thought binding menu. They do not require learned parsing.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Meaning import ConceptualMeaning
from ReferenceContext import SituationFrame
from ThoughtReferences import bindings, open_slots, with_slots
from order_fixtures import force_order
from test_reference_close import fixture


def identity(owner, point, *, order=1):
    symbol = owner.new_concept()
    force_order(owner, symbol, order)
    row = owner._csw_concept_row(order, symbol)
    with torch.no_grad():
        owner.similarity_codebook.W[row].copy_(point)
    return symbol


def forward_row(*, event=False, object_name=False):
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    word = owner.interpret.lookup_word([118], (), form='v')
    obj = identity(owner, atoms[0])
    owner.interpret.define(word, obj)
    name = obj if object_name else word
    point = atoms[0]
    value = replace(ConceptualMeaning.from_description(point), bindings={
        '_forward_references': ((0, name),),
        '_reference_sites': ((0, dict(role=0, leaves=(0,), content=point.tolist(),
            cue=(point*0).tolist(), predicate=(point*0).tolist())),)})
    value = with_slots(value, (('referent', 0),), pair=(0., 0.))
    row = store.append_meaning(value, evidence=(0., 0.))
    occurrence = store.occurrence_of(row)
    pool = owner.mtm
    pool.open(occurrence, kind=None, when=0, value=point, reading=word)
    pool.promote(occurrence, obj)
    if event:
        verb = identity(owner, atoms[6], order=0)
        changed = store.append_idea(point + atoms[1])
        pool.event(occurrence, store.occurrence_of(changed), verb=verb, role=0, when=1, before=point)
    entry = pool.entries[occurrence]
    assert entry.kind is None and entry.symbol == obj and not entry.pending
    assert entry.occurrences == (occurrence,)
    assert owner.definitions.objects(word) == (obj,)
    assert store.row(row)['evidence'] == (0.,0.)
    return SimpleNamespace(resolver=resolver, store=store, atoms=atoms, owner=owner,
        pool=pool, row=row, occurrence=occurrence, word=word, obj=obj, name=name)


def assert_open(value, name):
    assert ('referent', 0) in open_slots(value)
    assert (0, name) in bindings(value).get('_forward_references', ())
    assert ('referent', 0) not in bindings(value).get('_bound_roles', ())
    assert value.role_refs[0] is None


def selected_same_token(f):
    """Arrival selection can refine a token without replacing its key/name."""
    other = identity(f.owner, f.atoms[0] + f.atoms[1])
    f.pool.bind(f.occurrence, f.occurrence, kind=other, when=2,
                value=f.atoms[0] + f.atoms[1], reading=f.word)
    assert f.pool.entries[f.occurrence].symbol == f.obj
    assert f.pool.entries[f.occurrence].kind == other
    return other


def selected_external_token(f):
    other = identity(f.owner, f.atoms[0] + f.atoms[1])
    value = replace(ConceptualMeaning.from_description(f.atoms[0] + f.atoms[1]),
                    role_refs=(('sym', other), None, None))
    row = f.store.append_meaning(value,  evidence=(.6, .2),
                                 )
    occurrence = f.store.occurrence_of(row)
    f.pool.open(occurrence, kind=other, when=2, value=f.atoms[0] + f.atoms[1])
    f.pool.bind(occurrence, f.occurrence, kind=other, when=3,
                value=f.atoms[0], reading=f.word)
    assert f.pool.token_of(f.occurrence) == occurrence
    return row, occurrence, other


@pytest.mark.parametrize('event', (False, True))
@pytest.mark.parametrize('object_name', (False, True))
def test_forward_name_keeps_open_when_only_its_own_named_token_exists(event, object_name):
    f = forward_row(event=event, object_name=object_name)
    before = f.store.raw_meaning_of(f.row)
    entries = dict(f.pool.entries)
    value = f.store.meaning_of(f.row)
    assert_open(value, f.name)
    assert not bindings(value).get('_thought_witnesses')
    assert f.pool.entries == entries
    assert f.store.raw_meaning_of(f.row).metadata() == before.metadata()
    torch.testing.assert_close(f.store.raw_meaning_of(f.row).roles, before.roles, rtol=0, atol=0)


def test_selected_arrival_on_same_token_reads_the_new_kind_despite_its_old_name():
    f = forward_row()
    other = selected_same_token(f)
    value = f.store.meaning_of(f.row)
    assert ('referent', 0) not in open_slots(value)
    assert value.role_refs[0] == ('sym', other)
    assert not bindings(value).get('_forward_references')
    assert f.pool.entries[f.occurrence].symbol == f.obj
    assert_open(f.store.raw_meaning_of(f.row), f.name)


@pytest.mark.parametrize('address', (False, True))
def test_closing_exclusion_does_not_use_its_own_complement_as_a_witness(address):
    f = forward_row()
    selected_same_token(f)
    # This is ThoughtClosing's exclude=(index,) call, also expressed by its
    # canonical occurrence. The row is not independent evidence for itself.
    excluded = f.occurrence if address else f.row
    value = f.store.read_reference_bindings(f.row, f.store.raw_meaning_of(f.row),
                                           exclude=(excluded,))
    assert_open(value, f.name)
    assert not bindings(value).get('_thought_witnesses')


def test_excluding_the_request_retains_its_independently_selected_antecedent():
    f = forward_row()
    _, antecedent, other = selected_external_token(f)
    value = f.store.read_reference_bindings(f.row, f.store.raw_meaning_of(f.row),
                                           exclude=(f.row,))
    assert ('referent', 0) not in open_slots(value)
    assert value.role_refs[0] == ('sym', other)
    assert bindings(value)['_thought_witnesses'] == (antecedent,)
    assert not bindings(value).get('_forward_references')
    assert_open(f.store.raw_meaning_of(f.row), f.name)


def test_excluding_every_retained_use_keeps_the_forward_reference_open():
    f = forward_row()
    antecedent_row, _, _ = selected_external_token(f)
    value = f.store.read_reference_bindings(f.row, f.store.raw_meaning_of(f.row),
                                           exclude=(f.row, antecedent_row))
    assert_open(value, f.name)
    assert not bindings(value).get('_thought_witnesses')


def thought_plan(f, frames):
    value = f.store.raw_meaning_of(f.row)
    value = replace(value, bindings=dict(bindings(value), _query_occurrence=f.occurrence))
    plans = []
    f.resolver.bind_site(value, 0, frames, decisions=plans)
    assert len(plans) == 1
    return plans[0]


@pytest.mark.parametrize('departure', (False, True))
def test_finish_thought_bind_does_not_close_a_name_from_its_own_token(departure):
    f = forward_row(event=True)
    frame = SituationFrame.from_row(f.store.row(f.row))
    plan = thought_plan(f, (frame,))
    assert plan['candidates'] == (f.occurrence, f.obj)
    assert plan['base']['binding_occurrence'] == f.occurrence
    # Positive logits choose the candidate greedily; negative logits make it
    # the departure from the zero-logit dummy. Neither licenses self-naming.
    # The promoted concept is a separate menu entry; this probe selects
    # the original occurrence to test self-naming in both choice modes.
    logits = f.resolver.policy.new_tensor([-1. if departure else 1., -torch.inf])
    result = f.resolver.finish_thought_bind(plan, departure=departure, logits=logits)
    assert plan['decision']['candidates'] == (None, f.occurrence, f.obj)
    assert plan['decision']['selected'] == 1
    assert plan['decision']['greedy'] == (0 if departure else 1)
    assert plan['decision']['departed'] is departure
    assert_open(result['meaning'], f.name)
    assert result['incomplete'] == ('pending_antecedent',)
    assert not result['witnesses']


@pytest.mark.parametrize('departure', (False, True))
def test_finish_thought_bind_keeps_a_legitimate_independent_selected_antecedent(departure):
    f = forward_row()
    antecedent_row, antecedent, _ = selected_external_token(f)
    frame = SituationFrame.from_row(f.store.row(antecedent_row))
    plan = thought_plan(f, (frame,))
    assert plan['candidates'] == (antecedent,)
    logits = f.resolver.policy.new_full((1,), -1. if departure else 1.)
    result = f.resolver.finish_thought_bind(plan, departure=departure, logits=logits)
    assert ('referent', 0) not in open_slots(result['meaning'])
    assert result['meaning'].role_refs[0] == antecedent
    assert result['witnesses'] == (antecedent,)
    assert not result['incomplete']
