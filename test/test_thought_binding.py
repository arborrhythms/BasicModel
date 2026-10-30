"""Region alignment and proof-carrying equality substitution, without math."""
from types import SimpleNamespace
import torch
import pytest

from Meaning import ConceptualMeaning
from ThoughtReferences import bindings, evidence_pair, with_slots
from meaning_fixtures import unsupported as question
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
    components = {compound: dict(kind='lift', identity=operand),
                  rewritten: dict(kind='lift', identity=value)}
    first = read(name, compound, premise)
    first.evidence['evaluations'] = components
    def apply(goal, result, previous):
        return integrate(goal,result,previous,isomorphic=lambda a,b:a['kind']==b['kind'])
    goal = apply(question(equality(None, name), (('referent', 0),)), first, ())
    step = read(rewritten, answer, license, components=components, pair=(.6, .1))
    assert apply(goal, step, (first,)) is goal
    bound = apply(goal, step, (first, read(operand, value, binding, pair=(.7, 0.))))
    assert bound.role_refs[0] == answer
    assert bindings(bound)['_derived_binding']
    assert bindings(bound)['_thought_witnesses'] == (premise, binding, license)
    assert evidence_pair(bound) == (.6, .1)
    weak = apply(goal, read(rewritten, answer, license, components=components, pair=(.9, .9)),
                     (first, read(operand, value, binding, pair=(.4, 0.))))
    assert evidence_pair(weak) == (.4, .4)
    wrong_operation = {**components, rewritten: dict(components[rewritten], kind='other')}
    wrong = read(rewritten, answer, license)
    wrong.evidence['evaluations'] = wrong_operation
    assert apply(goal, wrong, (first, read(operand, value, binding))) is goal


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
    def signature(value, **kwargs):
        if int(value.role_mask.sum()) != 3:
            raise ValueError('an idea is not a relative clause')
        return SimpleNamespace(operation=SimpleNamespace(semantic_id='isEqual'))
    registry = SimpleNamespace(signature_for=signature)
    model = SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=store), grammatical_thoughts=registry)
    monkeypatch.setattr(Occurrence, 'source_at', lambda *args: ('document', 0))
    unused = with_slots(ConceptualMeaning.from_description(torch.eye(3)[1]), (), pair=(1., 0.))
    unused = replace(unused, role_refs=(('sym', b),None,None))
    concluded = replace(equality(('sym', a), ('sym', b)), constituents=(unused,))
    before = len(store)
    index = write(model, concluded, row=0)
    assert len(store) == before
    assert store.parts.references(index[1])[0][1] == a
    assert store.parts.references(index[1])[1][1] == b


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
    assert store.meaning_of(index).role_refs == (meaning.role_refs[2], None, None)
    assert bindings(store.meaning_of(index))['_bound_references'] == ((0, meaning.role_refs[0]),)
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
    assert pair['refs'].tolist() == ([address[2], b] if role == 0 else [b, address[2]])
    from LTM1 import OCCURRENCE, SYMBOL
    assert pair['ref_kinds'].tolist() == ([OCCURRENCE, SYMBOL] if role == 0 else [SYMBOL, OCCURRENCE])
    assert (pair['truth_plus'], pair['truth_minus']) == pytest.approx((.8, .1))
    assert len(f.store) == row + 1  # the relation is owned by LTM1
    assert f.store.raw_meaning_of(row).role_refs == (('sym', a), None, None)
    torch.testing.assert_close(f.store.slots[row], original, rtol=0, atol=0)


def test_empty_query_keeps_a_closed_equality_on_the_semantic_chain_path(monkeypatch):
    from test_item6_2_thinking import world, run
    model, registry, store, (a, b, c) = world()
    root = registry.form('equal', a, a)
    selected = []
    def choose(_root, active, actions, **kwargs):
        if selected:
            return None
        action = next(item for item in actions if item is not None
            and item.semantic_id == 'query' and item.request.constituents
            and item.request.constituents[0].role_refs == root.role_refs)
        selected.append(action)
        return action
    monkeypatch.setattr(model, '_choose_selected_thought_action', choose)
    result = run(model, root, work_budget=20)
    assert len(selected) == 1
    assert bindings(result.meaning)['_producing_operation'] == 'query'
    assert not bindings(result.meaning).get('_bound_roles')
    torch.testing.assert_close(result.meaning.roles, root.roles, rtol=0, atol=0)
    assert evidence_pair(result.meaning) == (0., 0.)
    saved = store.part_row(store.parts.find(a[1], a[1]))
    assert saved['evidence'] == (0., 0.)
    assert saved['refs'].tolist() == [a[1], a[1]]

@pytest.mark.parametrize('producer', ('query', 'bind'))
def test_unbound_read_of_unrecognized_multi_role_content_still_fails(monkeypatch, producer):
    from ThoughtStream import write
    from Meaning import ConceptualMeaning
    from test_item6_2_thinking import world
    model, registry, store, (a, b, c) = world()
    meaning = ConceptualMeaning(torch.eye(3, 8), torch.ones(3, dtype=torch.bool),
        role_refs=(a, b, c), bindings={'_producing_operation': producer})
    before = len(store), len(store.parts)
    with pytest.raises(ValueError, match='multi-role thought requires'):
        write(model, meaning, row=0)
    assert (len(store), len(store.parts)) == before


@pytest.mark.parametrize('pair', [(1., 0.), (0., 1.)])
def test_unrelated_query_evidence_does_not_rewrite_a_closed_unknown_chain(pair):
    """Captured curriculum crash: an object's negative evidence reached a clause."""
    from ThoughtStream import integrate as integrate_thought, materialize
    from Queries import ThoughtResult, _freeze_boundary_value
    goal = with_slots(ConceptualMeaning(torch.eye(3), torch.tensor([True, True, False]),
        role_refs=(('ltm', 'test', 1), ('ltm', 'test', 2), None),
        bindings={'_chain_view': True, '_query_occurrence': ('sym', 47)},
        sentence_kind='relation'), (), pair=(0., 0.), source_evidence=(0., 0.))
    point = ConceptualMeaning.from_description(torch.tensor([.2, .3, .4]))
    raw = ThoughtResult('query', 'conceptual-lookup', 'set', 'retrieval', goal,
        _freeze_boundary_value(dict(frames=(dict(meaning=point, occurrence=('ltm', 'test', 1)),),
            support_true=pair[0], support_false=pair[1], witnesses=(('ltm', 'test', 1),))))
    def signature(*args, **kwargs):
        raise ValueError('an ordinary predicate is not a registered thought operation')
    registry = SimpleNamespace(signature_for=signature, space=SimpleNamespace())
    result = integrate_thought(registry, goal, materialize(raw), ())
    assert evidence_pair(result) == (0., 0.) and result.role_refs == goal.role_refs
    torch.testing.assert_close(result.roles, goal.roles, rtol=0, atol=0)
    assert bindings(result)['_producing_operation'] == 'query'
    assert not bindings(result).get('_thought_witnesses')

@pytest.mark.parametrize('producer', ('query', 'bind'))
def test_unchanged_read_chain_still_reuses_its_witness(monkeypatch, producer):
    from dataclasses import replace
    from ThoughtStream import write
    from test_item6_2_thinking import world
    model, registry, store, (a, b, c) = world()
    reference = store.append_relation(a[1], b[1], truth_plus=.7)
    meaning = store.meaning_of(reference)
    meaning = replace(meaning, bindings=dict(bindings(meaning),
        _producing_operation=producer, _thought_witnesses=(reference,)))
    before = len(store), len(store.parts)
    assert write(model, meaning, row=0) == reference
    assert (len(store), len(store.parts)) == before


@pytest.mark.parametrize('producer', ('ask', 'query'))
@pytest.mark.parametrize('incomplete', ((), ('work_budget',)))
def test_empty_read_does_not_bind_its_temporary_question_address(producer, incomplete):
    """Captured native Part crash: a cutoff ask materialized its own request."""
    from Queries import ThoughtResult, _freeze_boundary_value
    from ThoughtReferences import open_slots, validate_local_references
    from ThoughtStream import integrate as integrate_thought, materialize, write
    goal = with_slots(ConceptualMeaning(torch.eye(3), torch.ones(3, dtype=torch.bool),
        role_refs=(None, ('sym', 99), ('sym', 75)), sentence_kind='relation',
        bindings={'_forward_references': ((0, 15),)}), (('referent', 0),),
        pair=(1., 1.), source_evidence=(1., 1.))
    request = ConceptualMeaning(torch.eye(3), torch.tensor([True, True, False]),
        role_refs=(('thought', 'address-v1', 123), ('sym', 1), None))
    raw = ThoughtResult(producer, 'conceptual-subgoal', 'subgoal', 'subgoal', request,
        _freeze_boundary_value(dict(value=None, frames=(), support_true=0., support_false=0.,
                                    incomplete=incomplete, records_scanned=0)))
    materialized = materialize(raw)
    assert materialized.evidence.get('meaning') is None
    assert materialized.incomplete == incomplete
    def signature(value, **kwargs):
        if value.role_refs[1] is None:
            raise ValueError('a point is not a registered relation')
        return SimpleNamespace(operation=SimpleNamespace(
            semantic_id='part' if value.role_refs[1] == ('sym', 99) else producer))
    registry = SimpleNamespace(signature_for=signature, space=SimpleNamespace())
    result = integrate_thought(registry, goal, materialized, ())
    assert result.role_refs == goal.role_refs and open_slots(result) == (('referent', 0),)
    torch.testing.assert_close(result.roles, goal.roles, rtol=0, atol=0)
    assert evidence_pair(result) == (1., 1.)
    assert bindings(result)['_producing_operation'] == producer
    assert not bindings(result).get('_bound_roles')
    store = SimpleNamespace(_validate_local_references=validate_local_references,
        write_clause=lambda clause, **kwargs: (clause, kwargs))
    model = SimpleNamespace(grammatical_thoughts=registry, symbolSpace=SimpleNamespace(ltm_store=store))
    clause, metadata = write(model, result, row=0)
    assert metadata['kind'] == 'question'
    assert clause.refs[0] == ('clause', 0) and clause.refs[2] == 75


@pytest.mark.parametrize('payload', ('meaning', 'frames'))
@pytest.mark.parametrize('pair', ((0., .7), (.4, .7)))
def test_returned_ask_content_keeps_both_evidence_poles(payload, pair):
    from dataclasses import replace
    from Queries import ThoughtResult, _freeze_boundary_value
    from ThoughtReferences import open_slots
    from ThoughtStream import integrate as integrate_thought, materialize
    goal = with_slots(ConceptualMeaning(torch.eye(3), torch.ones(3, dtype=torch.bool),
        role_refs=(None, ('sym', 99), ('sym', 75)), sentence_kind='relation'),
        (('referent', 0),), pair=(0., 0.))
    returned = with_slots(replace(goal, role_refs=(('sym', 77), ('sym', 99), ('sym', 75))), ())
    request = replace(goal, role_refs=(('thought', 'address-v1', 123), ('sym', 1), None))
    evidence = dict(support_true=pair[0], support_false=pair[1], witnesses=(('ltm', 'test', 3),))
    evidence[payload] = returned if payload == 'meaning' else (
        dict(meaning=returned, occurrence=('ltm', 'test', 3), evidence=pair),)
    result = materialize(ThoughtResult('ask', 'conceptual-subgoal', 'subgoal', 'subgoal',
        request, _freeze_boundary_value(evidence)))
    registry = SimpleNamespace(signature_for=lambda *args, **kwargs:
        SimpleNamespace(operation=SimpleNamespace(semantic_id='part')), space=SimpleNamespace())
    bound = integrate_thought(registry, goal, result, ())
    assert bound.role_refs == returned.role_refs and not open_slots(bound)
    assert evidence_pair(bound) == pair
    assert bindings(bound)['_thought_witnesses'] == (('ltm', 'test', 3),)
