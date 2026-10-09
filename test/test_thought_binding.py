"""Region alignment and proof-carrying equality substitution, without math."""
from types import SimpleNamespace
import torch

from Meaning import ConceptualMeaning
from ThoughtReferences import bindings, evidence_pair, question, with_slots
from ThoughtBinding import integrate


def equality(left, right, pair=(.8, 0.)):
    value = ConceptualMeaning(torch.eye(3), torch.ones(3, dtype=torch.bool),
        role_refs=(left, ('sym', 99), right), bindings={'_equality': True})
    return with_slots(value, (), pair=pair)


def read(left, right, occurrence, *, components=None, pair=(.8, 0.)):
    frame = dict(meaning=equality(left, right, pair), occurrence=occurrence, evidence=pair)
    return SimpleNamespace(evidence=dict(frames=(frame,), components=components or {}))


def test_reverse_equality_binds_the_other_operand_not_the_subject():
    subject, description = ('sym', 1), ('ltm', 'test', 2)
    goal = question(equality(None, subject), (('referent', 0),))
    result = integrate(goal, read(subject, description, ('ltm', 'test', 3)), ())
    assert result.role_refs == (description, ('sym', 99), subject)
    assert bindings(result)['_bound_roles'] == (('referent', 0),)
    assert bindings(result)['_direct_binding']


def test_an_unrelated_positive_row_does_not_fill_the_question():
    goal = question(equality(None, ('sym', 1)), (('referent', 0),))
    result = integrate(goal, read(('sym', 7), ('sym', 8), ('ltm', 'test', 3)), ())
    assert result is goal


def test_substitution_requires_the_returned_premise_and_keeps_its_witness():
    name, operand, value, answer, modifier = [('sym', n) for n in range(1, 6)]
    compound, rewritten = ('ltm', 'test', 6), ('ltm', 'test', 7)
    premise, binding, license = [('ltm', 'test', n) for n in (8, 9, 10)]
    components = {compound: dict(kind=0, operands=(operand, modifier), constructor=('lift', True)),
                  rewritten: dict(kind=0, operands=(value, modifier), constructor=('lift', True))}
    first = read(name, compound, premise, components=components)
    goal = integrate(question(equality(None, name), (('referent', 0),)), first, ())
    step = read(rewritten, answer, license, components=components, pair=(.6, .1))
    assert integrate(goal, step, (first,)) is goal
    bound = integrate(goal, step, (first, read(operand, value, binding, pair=(.7, 0.))))
    assert bound.role_refs[0] == answer
    assert bindings(bound)['_derived_binding']
    assert bindings(bound)['_thought_witnesses'] == (premise, binding, license)
    assert evidence_pair(bound) == (.6, .1)
    weak = integrate(goal, read(rewritten, answer, license, components=components, pair=(.9, .9)),
                     (first, read(operand, value, binding, pair=(.4, 0.))))
    assert evidence_pair(weak) == (.4, .4)
    wrong_operation = {**components, rewritten: dict(components[rewritten], constructor=('other', True))}
    wrong = read(rewritten, answer, license, components=wrong_operation)
    assert integrate(goal, wrong, (first, read(operand, value, binding))) is goal


def test_missing_cached_operand_is_not_a_structural_proof():
    name, a, b, answer = [('sym', n) for n in range(1, 5)]
    first = read(name, a, ('ltm', 'test', 8))
    goal = integrate(question(equality(None, name), (('referent', 0),)), first, ())
    step = read(b, answer, ('ltm', 'test', 9), components={
        a: dict(kind=0, operands=()), b: dict(kind=0, operands=())})
    assert integrate(goal, step, (first,)) is goal
