"""Numerical idea evaluation requires context, equality witnesses and work."""
from types import SimpleNamespace

import pytest
import torch

from IdeaEvaluation import isomorphic
from Language import LiftLayer
from Meaning import ConceptualMeaning
from QueryWork import QueryWorkBudget
from ThoughtBinding import integrate
from ThoughtReferences import bindings, with_slots


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('rail', [False, True])
def test_common_context_evaluates_distinct_inputs_without_mutating_kernel(native, rail):
    op = LiftLayer(nInput=4, nOutput=4)
    a = torch.tensor([.12, .23, .34, .45])
    b = torch.tensor([1., 0., 1., 0.]) if rail else torch.tensor([.3, .1, .4, .2])
    context = torch.tensor([.2, .3, .1, .4])
    left = dict(before=a, after=op.compose(a[None], context[None])[0], native=False)
    right = dict(before=b, after=op.compose((.8*b if native else b)[None], context[None])[0], native=native)
    activation = op._sigma.activation
    parameters = {name: value.detach().clone() for name, value in op.named_parameters()}
    work = QueryWorkBudget(256)
    assert isomorphic((op,), left, right, work)
    assert work.spent > 0 and op._sigma.activation is activation
    for name, value in op.named_parameters():
        torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
    assert not isomorphic((op,), left, right, QueryWorkBudget(0))
    assert not isomorphic((op,), dict(left, native=True), dict(right, native=True), QueryWorkBudget(8))
    # On a rail the same output can hide different latent contexts. Change
    # the observable output so the negative case remains distinguishable.
    different = dict(right, after=right['after'].roll(1))
    assert not isomorphic((op,), left, different, QueryWorkBudget(256))


def test_equal_outputs_without_input_equality_do_not_supply_a_proof():
    a, b, x, y, answer, witness = [('sym', i) for i in range(1, 7)]
    def equality(left, right, *, open_role=None):
        value = ConceptualMeaning(torch.ones(3, 4), torch.ones(3, dtype=torch.bool),
            role_refs=(left, None, right), bindings={'_equality': True})
        return with_slots(value, () if open_role is None else (('referent', open_role),), pair=(1., 0.))
    goal = equality(a, None, open_role=2)
    frame = dict(meaning=equality(b, answer), occurrence=witness, evidence=(1., 0.))
    result = SimpleNamespace(evidence=dict(frames=(frame,),
        evaluations={a: dict(identity=x), b: dict(identity=y)}))
    calls = []
    def checked(left, right):
        calls.append((left, right))
        return True
    assert integrate(goal, result, (), isomorphic=checked) is goal
    assert not calls
    proof = SimpleNamespace(evidence=dict(frames=(dict(meaning=equality(x, y),
        occurrence=('sym', 7), evidence=(1., 0.)),)))
    found = integrate(goal, result, (proof,), isomorphic=checked)
    assert found.role_refs[2] == answer and calls
    assert bindings(found)['_thought_witnesses'] == (('sym', 7), witness)
