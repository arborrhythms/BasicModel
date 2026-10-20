"""Generated binding-site certificates, with native rows and thought faces.

The generator supplies the structure and antecedent. These are mechanism
gates, not claims about learned parsing or reference resolution.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ClauseRow import Clause, Composition
from Meaning import ConceptualMeaning
from ReferenceContext import SituationFrame, reference_requests
from ThoughtReferences import bindings, evidence_pair, open_slots, with_slots
from constituent_corpus import KINDS, corpus
from test_clause_acceptance import SentenceFixture
from test_item6_2_thinking import world, run
from reading_fixtures import record_reading


def native(monkeypatch):
    f = SentenceFixture(monkeypatch, native_cube=True)
    model, _, _, _ = world()
    object.__setattr__(model, 'conceptualSpace', f.cs)
    object.__setattr__(model, 'grammatical_thoughts', f.registry)
    frames = []
    model.symbolSpace.ltm_store = f.store
    model.symbolSpace.grammatical_thoughts = f.registry
    model.symbolSpace.expectation = SimpleNamespace(situation_references=lambda row: tuple(frames))
    f.model = model
    f.frames = frames
    f.next_sentence = 0
    f.identities = {}
    f.atoms = torch.eye(8)
    f.resolver = f.cs.components.reference_close
    return f


def keep(f, clause, *, training=False):
    plan = f.resolver.preview(clause, tuple(f.frames))
    kept = f.resolver.prepare(clause, plan)
    written = {}
    sentence = f.next_sentence
    f.next_sentence += 1
    row = f.store.write_clause(kept, document_key='bind-corpus', sentence_index=sentence,
                               written_rows=written)
    f.resolver.commit(plan, row, written, document='bind-corpus', position=sentence,
                      training=training)
    plan['written'] = written
    f.frames.append(replace(SituationFrame.from_row(f.store.row(row)),
        symbol_ids=written.get(('symbols', id(kept)), ()),
        activations=written.get(('activations', id(kept)), ())))
    for item in plan['items']:
        target = written[id(item['written_node'])]
        f.identities[target] = item['reference']
    return row, plan


def occurrence(f, content, predicate, *, mode='mint', role=0, training=False):
    request = dict(mode=mode, role=role, leaves=(0, 1), cue=f.atoms[7],
                   predicate=predicate, content=content)
    term = Composition(content, order=1)
    constituent = Clause(ConceptualMeaning.from_description(content), point=content, order=1,
                         reference_request=request, composition=term)
    verb = Composition(predicate, order=0)
    point = torch.maximum(content, predicate)
    tree = Composition(point, (term, verb) if role == 0 else (verb, term),
                       operation='lift' if role == 0 else 'verb', order=1)
    clause = Clause(ConceptualMeaning.from_description(point), point=point, order=1,
                    children=(constituent,), composition=tree)
    row, plan = keep(f, clause, training=training)
    return plan['written'][id(plan['items'][0]['written_node'])], plan


def site(f, content, *, role=0, predicate=None, leaves=(0, 1), cue=None, occurrence=None):
    roles = torch.zeros(3, 8)
    roles[role] = content
    mask = roles.ne(0).any(-1)
    mask[role] = True
    metadata = dict(role=role, leaves=leaves,
        cue=None if cue is None else cue.tolist(),
        predicate=None if predicate is None else predicate.tolist())
    if occurrence is not None:
        metadata['occurrence'] = occurrence
    value = ConceptualMeaning(roles, mask, sentence_kind='idea',
        bindings={'_reference_sites': ((role, metadata),)})
    return with_slots(value, (('referent', role),), pair=(0., 0.))


def execute(f, meaning, role=0):
    from QueryWork import QueryWorkBudget
    operand = replace(meaning, bindings=dict(bindings(meaning), _bind_role=role))
    request = f.registry.form('bind', operand)
    with f.model._query_boundary_scope((0,)):
        context = f.model._thought_grammar_context(request, row=0,
            work=QueryWorkBudget(128), continuation=None)
        return f.registry.execute(request, context)


def receipt(gate, case, **values):
    directory = os.environ.get('ITEM35_BIND_RECEIPT')
    if directory:
        target = Path(directory) / gate
        target.mkdir(parents=True, exist_ok=True)
        (target / (case + '.json')).write_text(json.dumps(dict(case=case, **values), indent=2)+'\n')


PAIRS = [(kind, (kind+distance) % len(KINDS)) for kind in range(len(KINDS))
         for distance in range(1, len(KINDS))]
VERBS = sorted({row[3]['words'][2] for row in corpus()
                if row[3]['family'] == 'transitive'})


@pytest.mark.parametrize('kind,other', PAIRS)
def test_g1_definite_generator_referent_with_distractor(monkeypatch, kind, other):
    from ClauseJournal import finish_clause
    f = native(monkeypatch)
    target, _ = occurrence(f, f.atoms[kind], f.atoms[4])
    occurrence(f, f.atoms[other], f.atoms[5])
    expected = f.identities[target]
    # The same unary definite rule emits the close request. An unresolved
    # instance becomes the identical bind site in a thought episode.
    program = f.program(('lift', ('definite_unary', KINDS[kind]), VERBS[0]))
    program = replace(program, leaves=torch.stack((f.atoms[kind], f.atoms[4])),
        leaf_orders=torch.zeros(2, dtype=torch.long), close_references=True)
    clause = finish_clause(f.language, record_reading(f.language, program), registry=f.registry)
    from ReferenceClose import constituents
    node = constituents(clause)[0]
    assert node.reference_request['mode'] == 'bind'
    result = execute(f, site(f, node.point, predicate=node.reference_request['predicate'],
                            leaves=node.reference_request['leaves']))
    actual = result.value.role_refs[0]
    assert actual == ('sym', expected)
    assert not open_slots(result.value)
    assert evidence_pair(result.value) == pytest.approx((result.support_true, result.support_false))
    assert result.support_true + result.support_false == pytest.approx(1.)
    torch.testing.assert_close(f.registry.reverse(result, program.leaves),
                               program.leaves[:1], rtol=0, atol=0)
    receipt('G1', f'{kind}-{other}', text=f'a {KINDS[kind]} ; a {KINDS[other]} ; the {KINDS[kind]}',
            expected=expected, chosen=actual[1], pair=[result.support_true, result.support_false])


@pytest.mark.parametrize('kind,other', PAIRS)
def test_g2_generated_leaf_without_object_uses_same_resolver(monkeypatch, kind, other):
    from ClauseJournal import finish_clause
    f = native(monkeypatch)
    target, _ = occurrence(f, f.atoms[kind], f.atoms[4])
    occurrence(f, f.atoms[other], f.atoms[5])
    expected = f.identities[target]
    marker = f'm{kind}{other}'
    word = f.cs.interpret.lookup_word([100+kind*4+other], (), form=marker)
    assert f.cs.interpret.forward(word) == word and f.cs.interpret.no_object(word)
    from Spaces import _concept_alloc_of
    assert word in _concept_alloc_of(f.cs).word_forms[marker]
    assert word in f.cs.word_concepts(marker)
    f.words[marker] = word, word
    program = f.program(('lift', marker, VERBS[0]))
    program = replace(program, leaves=torch.stack((f.atoms[7], f.atoms[4])),
                      leaf_orders=torch.zeros(2, dtype=torch.long), close_references=True)
    program = record_reading(f.language, program)
    requests = reference_requests(f.language, program.actions,
        leaf_orders=program.leaf_orders, no_object=[True, False])
    assert requests[0] == (1, 'pronoun')
    clause = finish_clause(f.language, program, registry=f.registry)
    from ReferenceClose import constituents
    nodes = constituents(clause)
    assert len(nodes) == 1 and nodes[0].reference_request['mode'] == 'pronoun'
    assert not nodes[0].point.any()
    # Commit while the antecedents are outside the current situation, then
    # cue them. The real closing must open the NP, not invent a missing
    # object role in its containing intransitive clause.
    frames = tuple(f.frames)
    f.frames.clear()
    row, plan = keep(f, clause)
    from ThoughtClosing import closing_question
    query = closing_question(f.store.meaning_of(row), plan['prepared_root'],
        program=program, store=f.store, written_rows=plan['written'])
    assert query is not None and open_slots(query) == (('referent', 0),)
    assert bindings(query)['_forward_references'] == ((0, word),)
    f.frames[:] = frames
    result = execute(f, query)
    assert result.value.role_refs[0] == ('sym', expected)
    emitted = f.registry.reverse(result, program.leaves)
    torch.testing.assert_close(emitted, program.leaves[:1], rtol=0, atol=0)
    receipt('G2', f'{kind}-{other}', text=f'a {KINDS[kind]} ; a {KINDS[other]} ; {marker} {VERBS[0]}',
            expected=expected, chosen=result.value.role_refs[0][1],
            pair=[result.support_true, result.support_false], emitted=len(emitted))


@pytest.mark.parametrize('kind,accessible', [(k, a) for k in range(len(KINDS)) for a in (True, False)])
def test_g3_four_role_witnesses_open_elision_then_bind_or_remain_open(monkeypatch, kind, accessible):
    from ClauseJournal import finish_clause
    f = native(monkeypatch)
    for witness in range(4):
        target, _ = occurrence(f, f.atoms[kind], f.atoms[4], role=2,
                              mode='mint' if witness == 0 else 'bind', training=True)
    assert 2 in f.resolver.expected_roles(f.atoms[4])
    expected = f.identities[target] if accessible else None
    program = f.program(('lift', KINDS[(kind+1) % len(KINDS)], VERBS[0]))
    program = replace(program, leaves=torch.stack((f.atoms[(kind+1) % 4], f.atoms[4])),
                      leaf_orders=torch.zeros(2, dtype=torch.long))
    program = record_reading(f.language, program)
    clause = finish_clause(f.language, program, registry=f.registry)
    assert ('referent', 2) in open_slots(clause.meaning)
    if not accessible:
        f.frames.clear()
    result = execute(f, clause.meaning, 2)
    actual = result.value.role_refs[2]
    if accessible:
        assert actual == ('sym', expected)
        assert ('referent', 2) not in open_slots(result.value)
    else:
        assert actual is None and ('referent', 2) in open_slots(result.value)
        assert (result.support_true, result.support_false) == (0., 0.)
        assert result.incomplete == ('no_accessible_individual',)
    assert len(f.registry.reverse(result, program.leaves)) == 0
    receipt('G3', f'{kind}-{int(accessible)}', text=f'{KINDS[(kind+1)%4]} {VERBS[0]}',
            expected=expected, chosen=None if actual is None else actual[1],
            open_roles=open_slots(result.value), pair=[result.support_true, result.support_false])


def test_episode_has_one_bind_per_site_and_kept_write_preserves_np_content(monkeypatch):
    from ThoughtStream import candidates
    f = native(monkeypatch)
    missing, _ = occurrence(f, f.atoms[0], f.atoms[4], mode='bind')
    before = f.store.slots[missing].clone()
    target, _ = occurrence(f, f.atoms[0], f.atoms[4])
    meaning = f.store.meaning_of(missing)
    meaning = replace(meaning, bindings=dict(bindings(meaning), _query_occurrence=f.store.occurrence_of(missing)))
    offered = candidates(f.registry, meaning, meaning, meaning, ())
    assert [a.open_roles for a in offered if a.semantic_id == 'bind'] == [(0,)]
    selected = []
    def choose(root, active, actions, **kwargs):
        action = next((a for a in actions if a is not None and a.semantic_id == 'bind'), None)
        if action is not None:
            selected.append(action)
        return action
    monkeypatch.setattr(f.model, '_choose_selected_thought_action', choose)
    result = run(f.model, meaning, work_budget=128)
    assert len(selected) == 1
    assert result.meaning.role_refs[0] == ('sym', f.identities[target])
    assert not open_slots(result.meaning)
    assert open_slots(f.store.meaning_of(missing)) == (('referent', 0),)
    torch.testing.assert_close(f.store.slots[missing], before, rtol=0, atol=0)
    assert not [a for a in candidates(f.registry, meaning, result.meaning, result.meaning,
                                    result.records) if a.semantic_id == 'bind']


def test_participant_cache_rebuilds_from_retained_identity_rows(monkeypatch):
    f = native(monkeypatch)
    for _ in range(4):
        occurrence(f, f.atoms[0], f.atoms[4], role=2)
    assert f.resolver.expected_roles(f.atoms[4]) == (2,)
    saved = f.resolver._participants()
    f.store._participant_cache = None
    assert f.resolver._participants() == saved
    f.store.parts.retain([])
    assert f.resolver.expected_roles(f.atoms[4]) == ()


@pytest.mark.parametrize('has_antecedent', (False, True))
def test_bind_keeps_self_and_containing_identities_without_rewriting_occurrences(monkeypatch, has_antecedent):
    f = native(monkeypatch)
    missing, _ = occurrence(f, f.atoms[0], f.atoms[4], mode='bind')
    target = f.store.occurrence_of(missing)
    before = f.store.slots[missing].clone()
    # Self-reference is legal. Every candidate has a finite content point;
    # reading an identity must not recursively expand its own reference.
    # Both the open NP and its containing sentence are retained concepts.
    expected = {target, f.store.occurrence_of(f.store.location(f.frames[-1].row_id))}
    child = target
    for position in range(2):
        meaning = replace(ConceptualMeaning.from_description(f.atoms[0]),
                          role_refs=(child,None,None))
        row = f.store.append_meaning(meaning, order=1, document_key='ancestors',
                                    sentence_index=position)
        child = f.store.occurrence_of(row)
        expected.add(child)
        f.frames.append(SituationFrame.from_row(f.store.row(row)))
    if has_antecedent:
        row = f.store.append_meaning(ConceptualMeaning.from_description(f.atoms[0]*.5),
                                    order=1, document_key='independent')
        expected.add(f.store.occurrence_of(row))
        f.frames.append(SituationFrame.from_row(f.store.row(row)))
    query = f.store.meaning_of(missing)
    query = replace(query,bindings=dict(bindings(query),_query_occurrence=target))
    result = execute(f,query)
    assert set(result.evidence['binding_candidates']) == expected
    assert result.value.role_refs[0] in expected
    # The selected, potentially cyclic identity is a working interpretation.
    # It never changes the retained occurrence, even after thought completes.
    assert f.store.meaning_of(missing).role_refs[0] is None
    assert open_slots(f.store.meaning_of(missing)) == (('referent', 0),)
    view = f.store.read_structure(target, max_nodes=4, max_depth=4, max_records=len(f.store))
    assert len(view['rows']) <= 3
    torch.testing.assert_close(f.store.slots[missing],before,rtol=0,atol=0)


def test_distinct_open_roles_are_distinct_bind_sites_even_after_no_access(monkeypatch):
    from ThoughtStream import candidates
    f = native(monkeypatch)
    meaning = site(f, f.atoms[0])
    meaning = with_slots(meaning, (('referent', 0), ('referent', 2)))
    actions = candidates(f.registry, meaning, meaning, meaning, ())
    assert [a.open_roles for a in actions if a.semantic_id == 'bind'] == [(0,), (2,)]
    result = execute(f, meaning, 0)
    assert open_slots(result.value) == open_slots(meaning)
    records = (SimpleNamespace(kind='thought', result=result),)
    actions = candidates(f.registry, meaning, meaning, result.value, records)
    assert [a.open_roles for a in actions if a.semantic_id == 'bind'] == [(2,)]


def test_bind_cost_trains_existing_form_and_content_only_after_kept_commit(monkeypatch):
    from Layers import Error
    f = native(monkeypatch)
    occurrence(f, f.atoms[0], f.atoms[4])
    occurrence(f, f.atoms[1], f.atoms[5])
    from test_reference_close import form_exposure
    form_exposure(f.resolver, f.atoms[7], f.atoms[0])
    before = {name: value.clone() for name, value in f.resolver.named_parameters()}
    result = execute(f, site(f, f.atoms[0], predicate=f.atoms[4], cue=f.atoms[7]))
    for name, value in f.resolver.named_parameters():
        assert value.grad is None
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    record = SimpleNamespace(kind='thought', result=result)
    errors = Error()
    f.resolver.train_thought(SimpleNamespace(records=(record,)), errors)
    loss = errors.total(objective='expectation')
    assert loss is not None and loss.requires_grad
    loss.backward()
    assert f.resolver.policy.grad.abs().sum() > 0
    assert f.resolver.policy.grad[4].abs() > 0


def test_paired_bind_trials_use_r_plus_a_with_no_work_reward(monkeypatch):
    f = native(monkeypatch)
    occurrence(f, f.atoms[0], f.atoms[4])
    occurrence(f, f.atoms[1], f.atoms[5])
    from test_reference_close import form_exposure
    form_exposure(f.resolver, f.atoms[7], f.atoms[0])
    meaning = site(f, f.atoms[0], predicate=f.atoms[4], cue=f.atoms[7])
    # Force only the operation ordering, retaining native logits, departure,
    # execution, scoring, comparison, and kept commit.
    original_candidates = f.registry.controller_candidates
    def negation(*args, **kwargs):
        return tuple(action for action in original_candidates(*args, **kwargs)
                     if action.semantic_id == 'not')
    monkeypatch.setattr(f.registry, 'controller_candidates', negation)
    chooser = f.model.shared_grammar
    original_logits = chooser.thought_logits
    def prefer_binding(active, requests):
        logits = original_logits(active, requests)
        preferred = [request is None or f.registry.signature_for(
            request, verify_reference=False).operation.semantic_id == 'bind' for request in requests]
        return logits + logits.new_tensor(preferred) * (2*logits.detach().abs().sum()+1)
    monkeypatch.setattr(chooser, 'thought_logits', prefer_binding)
    f.model.train()
    f.model.errors.clear()
    measured = []
    def score(result):
        measured.append(f.cs.components.weight * sum(
            record.result.evidence.get('reference_cost', 0.)
            for record in result.records if record.kind == 'thought'
            and record.result is not None))
        return dict(reconstruction=0., answer=0.)
    result = run(f.model, meaning, work_budget=128, score=score)
    audit = f.model._last_thought_comparison
    assert audit['departure'] >= 0
    assert audit['keep_costs'] == pytest.approx(audit['costs'])
    assert audit['costs'] == pytest.approx(measured)
    assert audit['work_spent'][0] != audit['work_spent'][1]
    reference_cost = sum(record.result.evidence['reference_cost']
        for record in result.records if record.kind == 'thought'
        and record.result is not None and record.result.semantic_id == 'bind')
    assert min(audit['costs']) == pytest.approx(f.cs.components.weight*reference_cost)
    assert audit['explore_kept'] == (measured[1] < measured[0])
    loss = f.model.errors.total(objective='expectation')
    assert loss is not None and loss.requires_grad
    loss.backward()
    assert f.resolver.policy.grad[4].abs() > 0


def test_selected_open_formation_keeps_one_site_and_its_objectless_cue(monkeypatch):
    from ClauseJournal import finish_clause
    from ReferenceClose import constituents
    f = native(monkeypatch)
    word = f.cs.interpret.lookup_word([101], (), form='m')
    f.words['m'] = word, word
    program = f.program(('lift', 'm', VERBS[0]))
    program = replace(program, binding_choices=True,
        leaf_orders=torch.zeros(2, dtype=torch.long))
    program = record_reading(f.language, program)
    refs = program.operation_refs.clone()
    refs[-1, 0] = 0
    program = replace(program, operation_refs=refs)
    clause = finish_clause(f.language, program, registry=f.registry)
    assert ('referent', 0) in open_slots(clause.meaning)
    assert not constituents(clause), 'an explicit open role is not a second NP request'
    metadata = dict(dict(bindings(clause.meaning)['_reference_sites'])[0])
    assert metadata['leaves'] == (0,) and not any(metadata['content'])
    assert tuple(metadata['cue']) == tuple(program.leaves[0].tolist())


def test_pending_subject_keeps_its_composed_predicate_when_stored(monkeypatch):
    from ClauseJournal import finish_clause
    f = native(monkeypatch)
    word = f.cs.interpret.lookup_word([101], (), form='m')
    f.words['m'] = word, word
    program = f.program(('lift', 'm', ('verb', VERBS[0], KINDS[0])))
    program = replace(program, binding_choices=True,
        leaf_orders=torch.zeros(3, dtype=torch.long))
    program = record_reading(f.language, program)
    refs = program.operation_refs.clone()
    refs[-1, 0] = 0
    program = replace(program, operation_refs=refs)
    clause = finish_clause(f.language, program, registry=f.registry)
    assert clause.meaning.roles[1].any()
    row = f.store.write_clause(clause)
    stored = f.store.meaning_of(row)
    assert ('referent', 0) in open_slots(stored)
    assert ('relation', 1) not in open_slots(stored)
    assert stored.role_refs[1] is not None


@pytest.mark.parametrize('unresolved', (False, True))
def test_objectless_predicate_keeps_its_identity_in_the_ended_compound(monkeypatch, unresolved):
    from ClauseJournal import finish_clause
    f = native(monkeypatch)
    word = f.cs.interpret.lookup_word([101], (), form='v')
    f.words['v'] = word, word
    program = f.program(('equal', ('lift', KINDS[0], ('verb', 'v', KINDS[1])), KINDS[2]))
    program = record_reading(f.language, program)
    refs = program.operation_refs.clone()
    verb = next(i for i, (kind, operation, _) in enumerate(program.actions.tolist())
                if kind == 1 and f.language._compose_binary_rules[operation].method_name == 'verb')
    if unresolved:
        # The form has an identity while its object reference is still
        # unknown. A selected raw form is not an admitted-singleton alias.
        program = replace(program, reference_ids=torch.full_like(program.concept_ids, -1),
                          reference_orders=torch.zeros_like(program.concept_ids))
    else:
        refs[verb, 0] = -1  # the predicate has no reference-choice menu
    program = replace(program, operation_refs=refs)
    clause = finish_clause(f.language, program, registry=f.registry)
    row = f.store.write_clause(clause)
    # The relative parent ends the complete subject and its VP. A later
    # decomposition must be able to read the shared predicate's identity.
    subject = int(f.store.row(row)['refs'][0])
    vp = int(f.store.parts.refs[subject, 1])
    head = next(rule.head_role for rule in f.binary if rule.method_name == 'verb')
    assert int(f.store.parts.refs[vp, head - 1]) == word
