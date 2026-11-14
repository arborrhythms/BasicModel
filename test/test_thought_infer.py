"""Round A: addressed, trusted serial derivations over native rows."""
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import pytest
import torch

from Memory import STMChunks as chunks
from Meaning import ConceptualMeaning
from QueryWork import QueryWorkBudget
from ThoughtReferences import bindings, evidence_pair, open_slots, with_slots
from ThoughtChunks import attach, put, read, held, stage
from ThoughtStream import materialize
from test_item6_2_thinking import world, run


def idea(registry, reference, pair=(1., 0.)):
    return with_slots(replace(ConceptualMeaning.from_description(registry._payload(reference)),
        role_refs=(reference, None, None)), (), pair=pair)


def setup():
    model, registry, store, refs = world()
    memory, stm = model._what_memory(), model.conceptualSpace.stm
    attach(memory, stm)
    root = with_slots(ConceptualMeaning.from_description(registry._payload(refs[0])), (('referent', 0),))
    memory.begin_thought_episode(root, work_budget=256)
    return model, registry, store, refs


def invoke(model, registry, request):
    with model._query_boundary_scope((0,)):
        context = model._thought_grammar_context(request, row=0, work=QueryWorkBudget(256), continuation=None)
        result = materialize(registry.execute(request, context))
        return stage(model._what_memory(), result, 0)


def operands(model, registry, store, refs, *, premise_pair=(.8, .2), rule_pair=(.9, .3)):
    p, q, _ = refs
    premise = idea(registry, p, premise_pair)
    source = store.append_meaning(premise, evidence=premise_pair)
    rule = store.append_relation(p[1], q[1], truth_plus=rule_pair[0], truth_minus=rule_pair[1])
    rule_meaning = store.part_row(rule[1], reading='part')['meaning']
    premise = replace(premise, bindings=dict(bindings(premise), _thought_witnesses=(store.occurrence_of(source),)))
    rule_meaning = with_slots(replace(rule_meaning, bindings=dict(bindings(rule_meaning),
        _rule_row=rule, _rule_kind='part')), (), pair=rule_pair)
    stm = model.conceptualSpace.stm
    return (put(stm, premise, row=0), put(stm, rule_meaning, row=0), rule)


def test_chunk_truth_exact_address_clear_and_episode_pin():
    model, registry, store, refs = setup()
    stm, memory = model.conceptualSpace.stm, model._what_memory()
    focus = memory.thought_state().focus_when
    pointer = stm._live_chunk_truth.data_ptr()
    assert len(focus) == 4
    index = int(chunks.locate(stm.state, focus, band=chunks.WHEN)[0])
    assert int(stm.state[chunks.STACK][0, index]) >= 0
    deposited = []
    for number in range(10, 35):
        deposited.append(put(stm, idea(registry, refs[0], (.6, .2)), row=0))
    assert int(chunks.locate(stm.state, focus, band=chunks.WHEN)[0]) == index
    assert int(chunks.locate(stm.state, bindings(deposited[0])['_chunk_when'], band=chunks.WHEN)[0]) == -1
    memory.finish_thought(memory.thought_state().contexts[0].meaning)
    assert stm.state[chunks.STACK][0, index] == -1
    for number in range(40, 50):
        put(stm, idea(registry, refs[0]), row=0)
    assert int(chunks.locate(stm.state, focus, band=chunks.WHEN)[0]) == -1
    state = chunks.clear(stm.state, torch.ones_like(stm.state[chunks.READ], dtype=torch.bool))
    assert not state[chunks.TRUTH].any()
    assert state[chunks.TRUTH].shape == (1, 8, 2)
    assert stm._live_chunk_truth.data_ptr() == pointer


def test_trial_restore_keeps_active_streams_inside_the_constructed_batch():
    from Memory import ShortTermMemory
    from WalkTrials import ThoughtTrialState
    stm = ShortTermMemory(batch=3, concept_dim=8)
    stm.ensure_batch(1)
    from clock_fixtures import for_stm
    model = for_stm(stm)
    value = with_slots(ConceptualMeaning.from_description(torch.ones(8)), (), pair=(.7, .2))
    put(stm, value, row=0, pin=True)
    before = tuple(t.clone() for t in stm.state)
    pointer = stm._live_chunk_truth.data_ptr()
    trial = ThoughtTrialState(model)
    stm.ensure_batch(2)
    put(stm, value, row=1, pin=True)
    trial.restore()
    assert stm._active_batch == 1 and stm.snapshot().shape == (1, 1, 8)
    assert stm._live_buffer.shape == (3, 8, 8)
    assert stm._live_chunk_truth.data_ptr() == pointer
    for original, restored in zip(before, stm.state):
        torch.testing.assert_close(original, restored)
    assert not stm._live_chunk_truth[1:].any()


def test_duplicate_occupied_thought_address_fails_before_mutation():
    model, registry, store, refs = setup()
    stm = model.conceptualSpace.stm
    value = put(stm, idea(registry, refs[0], (.7, .2)), row=0)
    when = bindings(value)['_chunk_when']
    before = tuple(t.clone() for t in stm.state)
    contents = stm._thought_contents
    clock = model.when_time.clone()
    # Corrupt the producer clock to offer the already occupied stamp again.
    model.when_time.sub_(1)
    with pytest.raises(ValueError, match='already occupied'):
        put(stm, idea(registry, refs[1]), row=0)
    model.when_time.copy_(clock)
    assert stm._thought_contents is contents
    for original, unchanged in zip(before, stm.state):
        torch.testing.assert_close(original, unchanged)
    put(stm, with_slots(value, (), pair=(.6, .3)), when, 0, existing=True)
    assert evidence_pair(read(stm, when, 0)) == pytest.approx((.6, .3))
    other = bindings(put(stm, value, row=0))['_chunk_when']
    first, second = (int(chunks.locate(stm.state, item, band=chunks.WHEN)[0]) for item in (when, other))
    corrupt = list(stm.state)
    corrupt[chunks.WHEN] = corrupt[chunks.WHEN].clone()
    corrupt[chunks.WHEN][0, second] = corrupt[chunks.WHEN][0, first]
    stm.set_state(tuple(corrupt))
    # The traceable tensor primitive still returns its first occupied match.
    assert int(chunks.locate(stm.state, when, band=chunks.WHEN)[0]) == min(first, second)
    with pytest.raises(ValueError, match='multiple thought chunks'):
        read(stm, when, 0)
    with pytest.raises(ValueError, match='multiple thought chunks'):
        put(stm, value, when, 0, existing=True)


def test_membership_match_uses_the_existing_part_cache(monkeypatch):
    from Queries import ThoughtTaxonomyCapability
    from test_priming_diffusion import owner
    import Queries
    native = owner(torch.ones(4, 2))
    native._concept_allocator.parts.add(1, 2, symbol=100, truth_plus=.2)
    calls, original = [], Queries.capture_taxonomy
    def edges(value, **kwargs):
        calls.append(value)
        return original(value, **kwargs)
    monkeypatch.setattr(Queries, 'capture_taxonomy', edges)
    model, registry, store, refs = setup()
    premise, rule, row = operands(model, registry, store, refs, rule_pair=(.9, .7))
    # These fixtures name the cache's actual member and class rows.
    premise = put(model.conceptualSpace.stm,
        replace(premise, role_refs=(('sym', 1), None, None)), bindings(premise)['_chunk_when'], 0, existing=True)
    rule = put(model.conceptualSpace.stm,
        replace(rule, role_refs=(('sym', 2), rule.role_refs[1], refs[1])), bindings(rule)['_chunk_when'], 0, existing=True)
    request = registry.form('infer', premise, rule)
    with model._query_boundary_scope((0,)):
        context = model._thought_grammar_context(request, row=0, work=QueryWorkBudget(32), continuation=None)
        context = replace(context, taxonomy=ThoughtTaxonomyCapability(native))
        result = registry.execute(request, context)
    assert calls == [native]
    assert result.evidence['reason'] == 'membership'
    assert result.evidence['meaning'].role_refs[0] == ('sym', 1)
    assert result.evidence['derivation']['match_support_true'] == pytest.approx(.2)
    assert (result.support_true, result.support_false) == pytest.approx((.2, .2))
    from ThoughtInference import replay_pair
    assert replay_pair(result.evidence) == pytest.approx((.2, .2))


def test_unused_derived_idea_is_forgotten_by_existing_configuration():
    from Forgetting import forget, ForgettingConfig
    from Memory import LTM0
    from ThoughtStream import write
    model, registry, store, refs = world()
    store = LTM0(8, capacity=4)
    model.symbolSpace.ltm_store = store
    value = replace(idea(registry, refs[0], (.6, .2)), bindings={
        '_evidence_pair': (.6, .2), '_producing_operation': 'infer',
        '_thought_witnesses': (), '_evidence_kind': 'inference'})
    index = write(model, value, row=0)
    occurrence = store.occurrence_of(index)
    for number in range(store.capacity - len(store)):
        store.append_meaning(ConceptualMeaning.from_description(torch.ones(8)), trust=1., surprise=1.)
    forget(store, ForgettingConfig(), document_boundary=True)
    assert store.location(occurrence) is None


def test_forward_identity_lanes_alternative_max_and_no_extra_chunk():
    model, registry, store, refs = setup()
    premise, rule, row = operands(model, registry, store, refs)
    first = invoke(model, registry, registry.form('infer', premise, rule))
    assert first.evidence['reason'] == 'identity'
    assert first.evidence['derivation']['match_support_true'] == 1.
    assert evidence_pair(first.evidence['meaning']) == pytest.approx((.8, .3))
    assert first.evidence['meaning'].role_refs[0] == refs[1]
    assert first.evidence['sources'] == (bindings(premise)['_chunk_when'], row)
    count = int((model.conceptualSpace.stm.state[chunks.READ] > 0).sum())
    premise = put(model.conceptualSpace.stm, with_slots(premise, (), pair=(.95, .1)), bindings(premise)['_chunk_when'], 0, existing=True)
    second = invoke(model, registry, registry.form('infer', premise, rule))
    assert evidence_pair(second.evidence['meaning']) == pytest.approx((.9, .3))
    assert int((model.conceptualSpace.stm.state[chunks.READ] > 0).sum()) == count


def test_backward_writes_unknown_question_then_forward_uses_filled_premise():
    model, registry, store, refs = setup()
    premise, rule, row = operands(model, registry, store, refs)
    goal = put(model.conceptualSpace.stm, idea(registry, refs[1]), row=0)
    result = invoke(model, registry, registry.form('inferBackward', rule, goal))
    assert result.evidence_kind == 'subgoal'
    assert evidence_pair(result.evidence['meaning']) == (0., 0.)
    assert open_slots(result.evidence['meaning']) == (('referent', 0),)
    assert bindings(result.evidence['meaning'])['_subgoal_subject'] == refs[0]
    forward = invoke(model, registry, registry.form('infer', premise, rule))
    assert forward.evidence_kind == 'inference'


def test_unmatched_inference_keeps_the_open_focus():
    model, registry, store, refs = setup()
    _premise, rule, _row = operands(model, registry, store, refs)
    premise = put(model.conceptualSpace.stm, idea(registry, refs[2]), row=0)
    focus = model._what_memory().thought_state().focus_when
    before = read(model.conceptualSpace.stm, focus, 0)
    result = invoke(model, registry, registry.form('infer', premise, rule))
    assert result.evidence['incomplete'] == ('antecedent_mismatch',)
    assert result.evidence.get('meaning') is None
    after = read(model.conceptualSpace.stm, focus, 0)
    assert after.role_refs == before.role_refs and open_slots(after) == open_slots(before)
    assert evidence_pair(after) == (0., 0.)


def test_identity_match_reads_a_native_equality_witness():
    model, registry, store, refs = setup()
    premise, rule, _row = operands(model, registry, store, refs, rule_pair=(.9, .7))
    model.conceptualSpace._concept_allocator.singletons.add(refs[0][1])
    identity = store.append_relation(refs[2][1], refs[0][1], truth_plus=.2)
    premise = put(model.conceptualSpace.stm, idea(registry, refs[2], (.6, .1)), row=0)
    result = invoke(model, registry, registry.form('infer', premise, rule))
    assert result.evidence['reason'] == 'identity'
    assert result.evidence['meaning'].role_refs[0] == refs[1]
    assert identity in result.evidence['witnesses']
    assert result.evidence['derivation']['match_support_true'] == pytest.approx(.2)
    assert evidence_pair(result.evidence['meaning']) == pytest.approx((.2, .2))
    from ThoughtInference import replay_pair
    assert replay_pair(result.evidence) == pytest.approx((.2, .2))


def test_relation_rule_instantiates_its_consequent_with_the_member():
    from ThoughtStream import write
    from order_fixtures import force_order
    model, registry, store, (kind, middle, prop) = setup()
    token = ('sym', model.conceptualSpace.new_concept())
    force_order(model.conceptualSpace, token[1], 0)
    model.conceptualSpace._csw_concept_row(0, token[1])
    model.conceptualSpace.add_whole(token[1], kind)
    antecedent = registry.form('part', kind, middle)
    consequent = registry.form('part', kind, prop)
    rule = registry.form('implies', kind, prop)
    rule = with_slots(replace(rule,
        role_refs=(('constituent', 0), rule.role_refs[1], ('constituent', 1)),
        constituents=(antecedent, consequent)), (), pair=(.9, .3))
    index = write(model, rule, row=0)
    rule = replace(rule, bindings=dict(bindings(rule), _rule_row=store.occurrence_of(index), _rule_kind='implies'))
    premise = with_slots(registry.form('part', token, middle), (), pair=(.7, .2))
    premise = put(model.conceptualSpace.stm, premise, row=0)
    rule = put(model.conceptualSpace.stm, rule, row=0)
    result = invoke(model, registry, registry.form('infer', premise, rule))
    assert result.evidence['reason'] == 'membership'
    assert result.evidence['meaning'].role_refs == (token, consequent.role_refs[1], prop)
    assert evidence_pair(result.evidence['meaning']) == pytest.approx((.7, .3))


def test_negated_operand_and_equality_use_the_same_operator():
    model, registry, store, refs = setup()
    premise, rule, row = operands(model, registry, store, refs)
    negated = invoke(model, registry, registry.form('not', idea(registry, refs[1], (.1, .7))))
    result = invoke(model, registry, registry.form('infer', negated.evidence['meaning'], rule))
    assert result.evidence['meaning'].role_refs[0] == refs[0]
    assert evidence_pair(result.evidence['meaning']) == pytest.approx((.8, .7))
    from ThoughtInference import replay_pair
    assert replay_pair(result.evidence) == pytest.approx((.8, .7))
    assert result.evidence['alternatives'] == ((bindings(premise)['_chunk_when'], pytest.approx((.8, .2))),)
    model._what_memory().commit_thought(result.evidence['meaning'], operation='infer',
        result=result, support_true=result.support_true, support_false=result.support_false,
        evidence_kind='inference')
    equality = put(model.conceptualSpace.stm, replace(rule,
        bindings=dict(bindings(rule), _rule_kind='equal')), bindings(rule)['_chunk_when'], 0, existing=True)
    result = invoke(model, registry, registry.form('infer', premise, equality))
    assert result.evidence['reason'] == 'equality'
    assert result.evidence['meaning'].role_refs[0] == refs[1]


def test_intuition_may_answer_and_retired_face_is_absent():
    from ThoughtAnswer import accepted
    from Queries import THOUGHT_EXECUTORS
    model, registry, store, refs = setup()
    value = replace(idea(registry, refs[0]), bindings={
        '_evidence_kind': 'intuition', '_producing_operation': 'intuit', '_evidence_pair': (.8, .2)})
    assert accepted(value)  # Item 1.8 §3.6: intuition may answer.
    _premise, rule, _row = operands(model, registry, store, refs)
    value = put(model.conceptualSpace.stm, value, row=0)
    selected = invoke(model, registry, registry.form('infer', value, rule))
    assert accepted(selected.evidence['meaning'])
    assert 'intuit' in THOUGHT_EXECUTORS and 'intuit' in registry.executable_operation_ids
    result = invoke(model, registry, registry.form('intuit', value))
    assert result.evidence_kind == 'intuition'
    assert accepted(result.evidence['meaning'])  # A closed intuition may answer (§3.6).
    retired = 'is' + 'Implied'
    assert retired not in THOUGHT_EXECUTORS
    assert ('is_' + 'implied') not in Path('bin/ThoughtFaces.py').read_text()
    for path in Path('data').glob('*.grammar'):
        assert retired not in path.read_text()


def test_question_thought_menu_offers_built_intuit(monkeypatch):
    model, registry, store, refs = world()
    question = registry.form('isPart', refs[0], open_roles=('argin2',))
    observed = []
    choose = model._choose_selected_thought_action
    def inspect(root, active, actions, **kwargs):
        actions = tuple(actions)
        offered = {action.semantic_id for action in actions if action is not None}
        observed.append(offered)
        assert 'intuit' in offered
        return choose(root, active, actions, **kwargs)
    monkeypatch.setattr(model, '_choose_selected_thought_action', inspect)
    run(model, question, work_budget=64)
    assert observed
    # D9 retires with the executor, including an explicit current request.
    request = registry.form('intuit', question)
    assert any(action.semantic_id == 'intuit' for action in
               registry.controller_candidates(question, question, request))
    assert invoke(model, registry, request).evidence_kind == 'intuition'


def test_binding_cost_grades_two_incorrect_text_trials_and_credits_better(monkeypatch):
    from BindingAnswers import cost
    from whole_path_fixtures import thought_surrogate as surrogate
    model, registry, store, refs = world()
    owner = model._concept_owner()
    target = refs[2]
    owner.word_concepts = lambda word: (target[1],)
    row = owner._csw_row_of(target[1])
    with torch.no_grad():
        owner.similarity_codebook.W[row].fill_(1.)
    def answer(value):
        result = replace(idea(registry, refs[0]), roles=torch.full((3, 8), value))
        return replace(result, bindings={'_bound_roles': (('referent', 0),)})
    from Models import BasicModel
    model._sentence_answer_questions = ('question',)
    object.__setattr__(model, 'inputSpace', SimpleNamespace(data=SimpleNamespace(
        what=lambda question: SimpleNamespace(available=True, kind='referent', what='target'))))
    from whole_path_fixtures import answer as score_answer
    costs = [float(score_answer(model, [answer(value)], torch.ones(1))[0]) for value in (.2, .8)]
    assert costs == pytest.approx([float(cost(model, answer(value), 'target')) + 1. for value in (.2, .8)])
    assert costs[1] < costs[0]
    parameter = torch.nn.Parameter(torch.tensor([.2, -.1]))
    probability = parameter.softmax(0)
    credit = surrogate(dict(eligible=[True]),
        dict(choices=[dict(probability=probability[1], alternatives=1)]), 0, costs)
    credit.backward()
    assert parameter.grad[1] < 0 < parameter.grad[0]


def test_worked_token_property_line_queries_backward_and_forward_then_replays(monkeypatch):
    """Felix is a cat; cats are furry; is Felix furry? Words are test data."""
    from Thoughts import replay_thoughts
    model, registry, store, (token, kind, prop) = world()
    membership = store.append_relation(token[1], kind[1], truth_plus=.7, truth_minus=.2)
    rule_row = store.append_relation(kind[1], prop[1], truth_plus=.9, truth_minus=.3)
    goal = registry.form('isPart', token, prop)
    goal = with_slots(replace(goal, bindings={'_query_relation': 'part',
        '_forward_references': ((0, token),)}), (('referent', 0),))
    selected = []
    def choose(root, active, actions, **kwargs):
        step = len(selected)
        if step == 4:
            assert None in actions
            return None
        resident = dict(held(model.conceptualSpace.stm, 0))
        for action in actions:
            if action is None:
                continue
            request = action.request
            if step in (0, 2) and action.semantic_id == 'query' and request.constituents:
                pattern = request.constituents[0]
                if ((step == 0 and pattern.role_refs[0] is None and pattern.role_refs[2] == prop)
                        or (step == 2 and bindings(pattern).get('_subgoal_subject') == token
                            and pattern.role_refs[2] == kind)):
                    selected.append(action)
                    return action
            if step in (1, 3) and action.semantic_id == 'infer':
                premise = resident.get(request.role_refs[0][1])
                rule = resident.get(request.role_refs[2][1])
                direction = bindings(request).get('_infer_direction')
                if (premise is not None and rule is not None
                        and bindings(rule).get('_rule_row') == rule_row
                        and premise.role_refs[0] == token
                        and premise.role_refs[2] == (prop if step == 1 else kind)
                        and direction == ((2, 0) if step == 1 else (0, 2))):
                    selected.append(action)
                    return action
        raise AssertionError(f'worked line step {step} missing from {[(a.semantic_id, a.request.role_refs) for a in actions if a]}')
    monkeypatch.setattr(model, '_choose_selected_thought_action', choose)
    result = run(model, goal, work_budget=256)
    assert [item.semantic_id for item in selected] == ['query', 'infer', 'query', 'infer']
    forward = next(record.result for record in result.records
                   if record.operation == 'infer' and record.evidence_kind == 'inference')
    assert forward.evidence['reason'] == 'instantiation'
    assert result.meaning.role_refs[0] == token and result.meaning.role_refs[2] == prop
    assert evidence_pair(result.meaning) == pytest.approx((.7, .3))
    assert bindings(result.meaning)['_thought_witnesses'] == (membership, rule_row)
    replay = replay_thoughts(result.records, chunks_state=model.conceptualSpace.stm.state)
    assert replay.finished
    assert evidence_pair(replay.contexts[0].meaning) == pytest.approx((.7, .3))
    # §17.12a keeps the two signed claims separately; §17.21–22 retires
    # the row kind and the stored proof. The replay above checks the proof.
    rows = [store.row(i) for i in range(len(store))]
    assert len(rows) >= 2
    assert [row['trust'] for row in rows[-2:]] == pytest.approx((.7, -.3))
    tampered = list(result.records)
    position = next(i for i, record in enumerate(tampered) if record.operation == 'infer' and record.evidence_kind == 'inference')
    tampered[position] = replace(tampered[position], support_true=.1)
    with pytest.raises(ValueError, match='replayed supports'):
        replay_thoughts(tampered)
