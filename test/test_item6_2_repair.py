"""Constituent ownership at thought filling, before any index or write."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from ThoughtReferences import bindings, evidence_pair, fill, open_slots, question
from ThoughtStream import write


def atom(value=1.):
    return ConceptualMeaning.from_description(torch.tensor([value, 0., 0., 0.]))


def supply(meaning):
    return dict(meaning=meaning, support_true=.8, support_false=.2)


def test_saved_construction_binds_and_writes_the_child():
    child = atom()
    supplied = replace(child, role_refs=(('constituent', 0), None, None),
                       constituents=(child,))
    goal = question(child, (('referent', 0),))
    resolved = fill(goal, supply(supplied), operation='ask')
    assert resolved.constituents == (child,)
    assert not open_slots(resolved)
    assert evidence_pair(resolved) == (.8, .2)
    store = TernaryTruthStore(4, capacity=8)
    model = SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=store))
    index = write(model, resolved, row=0)
    assert len(store) == 2
    assert store.row(index)['kind'] == 'inference'
    assert store.meaning_of(index).role_refs[0] == store.occurrence_of(0)
    assert bindings(store.meaning_of(index))['_producing_operation'] == 'ask'


def test_fill_rebases_into_nonempty_goal_and_keeps_nested_ownership():
    existing, leaf = atom(.25), atom(.5)
    nested = replace(atom(), role_refs=(('constituent', 0), None, None),
                     constituents=(leaf,), scope=(('child', ('constituent', 0)),))
    supplied = replace(atom(), role_refs=(('constituent', 0), None, None),
                       constituents=(nested,))
    goal = ConceptualMeaning.from_description(torch.ones(3, 4))
    goal = question(replace(goal, role_refs=(None, None, ('constituent', 0)),
                            constituents=(existing,)), (('referent', 0),))
    resolved = fill(goal, supply(supplied))
    assert resolved.role_refs == (('constituent', 1), None, ('constituent', 0))
    assert resolved.constituents == (existing, nested)
    assert nested.constituents == (leaf,)
    assert nested.role_refs[0] == ('constituent', 0)
    assert goal.constituents == (existing,) and goal.role_refs[0] is None
    store = TernaryTruthStore(4, capacity=8)
    model = SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=store))
    index = write(model, resolved, row=0)
    assert len(store) == 4
    root = store.meaning_of(index)
    assert root.role_refs[2] == store.occurrence_of(0)
    assert root.role_refs[0] == store.occurrence_of(2)
    assert store.meaning_of(2).role_refs[0] == store.occurrence_of(1)
    assert store.meaning_of(2).scope == (('child', store.occurrence_of(1)),)


def test_multiple_roles_share_the_imported_child_without_reindexing_goal():
    child = atom()
    base = ConceptualMeaning.from_description(torch.ones(3, 4))
    supplied = replace(base, role_refs=(('constituent', 0), None, ('constituent', 1)),
                       constituents=(child, child))
    goal = question(replace(base, constituents=(child,)),
                    (('referent', 0), ('referent', 2)))
    resolved = fill(goal, supply(supplied))
    assert resolved.constituents == (child,)
    assert resolved.role_refs == (('constituent', 0), None, ('constituent', 0))


def test_occurrence_binding_also_carries_the_supplied_constituent_table():
    child = atom(.5)
    supplied = replace(atom(), constituents=(child,))
    store = TernaryTruthStore(4, capacity=8)
    index = store.append_meaning(supplied, kind='fact', evidence=(.8, .2))
    occurrence = store.occurrence_of(index)
    result = fill(question(atom(), (('referent', 0),)),
                  dict(**supply(supplied), frames=({'meaning': supplied,
                                                   'occurrence': occurrence},)))
    assert result.role_refs[0] == occurrence
    assert result.constituents == (child,)
    assert not open_slots(result)


@pytest.mark.parametrize('reference', [
    ('constituent', 0), ('constituent', -1), ('constituent', True),
    ('constituent', 0.,), ('constituent',), ('constituent', 0, 1),
])
def test_dangling_or_malformed_supplied_local_reference_raises_at_fill(reference):
    supplied = replace(atom(), role_refs=(reference, None, None))
    with pytest.raises(ValueError, match='local constituent reference at fill'):
        fill(question(atom(), (('referent', 0),)), supply(supplied))


def test_nested_metadata_and_goal_local_references_are_checked_at_fill():
    bad = replace(atom(), scope=(('target', ('constituent', 0)),))
    supplied = replace(atom(), constituents=(bad,))
    goal = question(atom(), (('referent', 0),))
    with pytest.raises(ValueError, match='local constituent reference at fill'):
        fill(goal, supply(supplied))
    with pytest.raises(ValueError, match='local constituent reference at fill'):
        fill(question(bad, (('referent', 0),)), supply(atom()))


def test_carried_constituent_keeps_its_gradient():
    parameter = torch.tensor([1., 0., 0., 0.], requires_grad=True)
    child = ConceptualMeaning.from_description(parameter)
    supplied = replace(atom(), role_refs=(('constituent', 0), None, None),
                       constituents=(child,))
    result = fill(question(atom(), (('referent', 0),)), supply(supplied))
    result.constituents[0].roles.sum().backward()
    torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
