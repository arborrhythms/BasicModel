"""Region alignment and proof-carrying equality substitution, without math."""
from types import SimpleNamespace
import torch
import pytest

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
    components = {compound: dict(kind='lift', operands=(operand, modifier)),
                  rewritten: dict(kind='lift', operands=(value, modifier))}
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
    wrong_operation = {**components, rewritten: dict(components[rewritten], kind='other')}
    wrong = read(rewritten, answer, license, components=wrong_operation)
    assert integrate(goal, wrong, (first, read(operand, value, binding))) is goal


def test_missing_cached_operand_is_not_a_structural_proof():
    name, a, b, answer = [('sym', n) for n in range(1, 5)]
    first = read(name, a, ('ltm', 'test', 8))
    goal = integrate(question(equality(None, name), (('referent', 0),)), first, ())
    step = read(b, answer, ('ltm', 'test', 9), components={
        a: dict(kind=0, operands=()), b: dict(kind=0, operands=())})
    assert integrate(goal, step, (first,)) is goal


def test_conclusion_does_not_publish_unreferenced_working_constituents(monkeypatch):
    from dataclasses import replace
    from part_fixtures import native_store
    from ThoughtStream import write
    import Occurrence
    store, (a, b, predicate) = native_store(width=3, points=torch.eye(3))
    model = SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=store))
    monkeypatch.setattr(Occurrence, 'source_at', lambda *args: ('document', 0))
    unused = with_slots(ConceptualMeaning.from_description(torch.eye(3)[1]), (), pair=(1., 0.))
    concluded = replace(equality(('sym', a), ('sym', b)), constituents=(unused,))
    before = len(store)
    index = write(model, concluded, row=0)
    assert len(store) == before + 1
    assert store.meaning_of(index).role_refs == concluded.role_refs


@pytest.mark.parametrize('producer', ('query', 'bind'))
def test_a_read_binding_publishes_a_proof_occurrence_without_composing_a_part(monkeypatch, producer):
    from dataclasses import replace
    from part_fixtures import native_store
    from ThoughtStream import write
    import Occurrence
    store, (a, b, predicate) = native_store(width=3, points=torch.eye(3))
    registry = SimpleNamespace(signature_for=lambda *args, **kwargs:
        SimpleNamespace(operation=SimpleNamespace(semantic_id='isEqual')))
    model = SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=store), grammatical_thoughts=registry)
    monkeypatch.setattr(Occurrence, 'source_at', lambda *args: ('document', 0))
    meaning = equality(('sym', a), ('sym', b))
    meaning = replace(meaning, bindings=dict(bindings(meaning),
        _producing_operation=producer, _bound_roles=(('referent', 0),),
        _thought_witnesses=(('sym', a),)))
    counts = len(store), len(store.parts)
    index = write(model, meaning, row=0)
    assert (len(store), len(store.parts)) == (counts[0] + 1, counts[1])
    assert store.row(index)['kind'] == 'inference'
    assert store.meaning_of(index).role_refs == meaning.role_refs
    assert bindings(store.meaning_of(index))['_thought_witnesses'] == (('sym', a),)


@pytest.mark.parametrize('reverse', (False, True))
@pytest.mark.parametrize('repeated', (False, True))
def test_unresolved_operand_is_not_an_answer_but_keeps_its_equality(reverse, repeated):
    from dataclasses import replace
    from ThoughtReferences import open_slots
    name, unresolved = ('sym', 1), ('ltm', 'test', 2)
    result = read(name, unresolved, ('ltm', 'test', 3))
    frame = result.evidence['frames'][0]
    frame['meaning'] = replace(frame['meaning'], bindings=dict(
        bindings(frame['meaning']), _unresolved_operands=((2, unresolved),)))
    target = unresolved if reverse else name
    goal = question(equality(None, target), (('referent', 0),))
    updated = integrate(goal, result, (result,) if repeated else ())
    if reverse:
        assert updated.role_refs[0] == name and not open_slots(updated)
        assert bindings(updated)['_thought_witnesses'] == (frame['occurrence'],)
    else:
        assert updated is goal and open_slots(updated) == (('referent', 0),)
        assert '_bound_roles' not in bindings(updated)


def test_unresolved_operand_guard_applies_to_ordinary_fill_without_changing_features():
    from dataclasses import replace
    from ThoughtFeatures import semantic_metadata
    from ThoughtReferences import fill, open_slots
    name, unresolved = ('sym', 1), ('ltm', 'test', 2)
    supplied = equality(name, unresolved)
    marked = replace(supplied, bindings=dict(bindings(supplied),
        _unresolved_operands=((2, unresolved),)))
    torch.testing.assert_close(semantic_metadata((supplied,))[0],
                               semantic_metadata((marked,))[0], rtol=0, atol=0)
    goal = question(equality(name, None), (('referent', 2),))
    evidence = dict(frames=(dict(meaning=marked, occurrence=('ltm', 'test', 3)),),
                    support_true=.8, support_false=0.)
    updated = fill(goal, evidence, operation='query')
    assert updated.role_refs[2] is None and open_slots(updated) == (('referent', 2),)
    assert '_bound_roles' not in bindings(updated)
    evidence['frames'] = (dict(meaning=supplied, occurrence=('ltm', 'test', 3)),)
    assert fill(goal, evidence, operation='query').role_refs[2] == unresolved


def test_explicit_reflexive_equality_remains_a_direct_witness():
    from ThoughtReferences import open_slots
    name = ('sym', 1)
    goal = question(equality(None, name), (('referent', 0),))
    updated = integrate(goal, read(name, name, ('ltm', 'test', 3)), ())
    assert updated.role_refs[0] == name and not open_slots(updated)
    assert bindings(updated)['_thought_witnesses'] == (('ltm', 'test', 3),)


@pytest.mark.parametrize('role', (0, 2))
def test_concluded_relation_keeps_its_occurrence_argument(monkeypatch, role):
    from dataclasses import replace
    from ClauseRow import Clause, predicate_identity, predicate_point
    from ThoughtStream import write
    from test_clause_acceptance import SentenceFixture
    import Occurrence
    f = SentenceFixture(monkeypatch)
    a, b = f.noun('a'), f.noun('b')
    point = f.registry._payload(('sym', a))
    row = f.store.write_clause(Clause(ConceptualMeaning.from_description(point),
        point=point, refs=(a, -1, -1)), document_key='initial', sentence_index=0)
    address = f.store.occurrence_of(row)
    refs = [('sym', b), ('sym', predicate_identity('part')), ('sym', b)]
    refs[role] = address
    values = torch.stack((f.registry._payload(('sym', b)), predicate_point('part', point),
                          f.registry._payload(('sym', b))))
    values[role] = point
    concluded = with_slots(ConceptualMeaning(values, torch.ones(3, dtype=torch.bool),
        role_refs=tuple(refs)), (), pair=(.8, .1))
    f.model.symbolSpace = SimpleNamespace(ltm_store=f.store)
    monkeypatch.setattr(Occurrence, 'source_at', lambda *args: ('conclusion', 0))
    original = f.store.slots[row].clone()
    result = write(f.model, concluded, row=0)[1]
    assert f.store.parts.contains(result)
    pair = f.store.parts.row(result)
    assert pair['refs'].tolist() == ([a, b] if role == 0 else [b, a])
    assert (pair['truth_plus'], pair['truth_minus']) == pytest.approx((.8, .1))
    assert f.store.raw_meaning_of(len(f.store)-1).role_refs == (('sym', result), None, None)
    torch.testing.assert_close(f.store.slots[row], original, rtol=0, atol=0)
