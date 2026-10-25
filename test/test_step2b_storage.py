"""Storage-only 2b contracts; no parsing or learning claim."""
import copy

import pytest
import torch

from Layers import LTM0
from Meaning import ConceptualMeaning


def test_definition_rejects_a_word_object_and_keeps_both_forms():
    from test_definition_rows import model as definition_model, admit
    model = definition_model()
    try:
        owner = model._concept_owner()
        identity = model.perceptualSpace.percept_store.identity
        word, obj = admit(model, '12')
        other, _ = admit(model, 'plus')
        assert owner.word_surface_for_row(owner._csw_row_of(word)) == b'12'
        with pytest.raises(ValueError, match='word as its object'):
            owner.interpret.define(other, word)
        assert owner.definitions.words(word) == ()
        assert owner.word_surface_for_row(owner._csw_row_of(word)) == b'12'
        assert owner.word_surface_for_row(owner._csw_row_of(other)) == b'plus'
        assert owner.word_surface_for_row(owner._csw_row_of(obj)) == b'12'
        for symbol in (word, obj):
            row = owner._csw_row_of(symbol)
            assert identity.read(owner.similarity_codebook.lookup_rows(row)[:identity.width]) == b'12'
        assert owner.definitions.words(obj) == (word,)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.parametrize('relative', (False, True))
@pytest.mark.parametrize('parsed', (False, True))
def test_live_zero_evidence_question_opens_after_its_row_is_written(relative, parsed):
    from dataclasses import replace
    from ThoughtClosing import closing_question
    from ThoughtReferences import evidence_pair, needs_episode, open_slots
    from test_clause_storage import clause_store, idea_clause, part_clause
    store, _ = clause_store()
    clause = part_clause() if relative else idea_clause()
    clause = replace(clause, meaning=replace(clause.meaning, mode='interrogative'))
    written = {}
    row = store.write_clause(clause, evidence=(0., 0.), written_rows=written)
    query = closing_question(store.meaning_of(row), clause,
        clause.meaning if parsed else None, store=store, written_rows=written)
    assert query is not None and query.mode == 'interrogative'
    assert needs_episode(query) and open_slots(query) == ()
    assert evidence_pair(query) == (0., 0.)
    assert 'grammatical_mode' not in store.state_dict()


def test_stored_question_is_refused_by_kind_without_a_mode_column():
    store = LTM0(4, capacity=4)
    question = ConceptualMeaning.from_description(torch.ones(4))
    from dataclasses import replace
    row = store.append_meaning(replace(question, mode='interrogative'), kind='question')
    state = copy.deepcopy(store.state_dict())
    assert 'grammatical_mode' not in state and not hasattr(store, 'MODES')
    restored = LTM0(4, capacity=4)
    restored.load_state_dict(state)
    restored.load_semantic_extras(store.semantic_extras())
    for owner in (store, restored):
        assert owner.row(row)['kind'] == 'question'
        assert owner.row(row)['evidence'] == (0., 0.)
        with pytest.raises(ValueError, match='question, estimate or cover'):
            owner.accept_fact(row)
        ordinary = owner.append_meaning(question, sentence_index=1)
        owner.accept_fact(ordinary)
        assert owner.row(ordinary)['kind'] == 'fact'


@pytest.mark.parametrize('relation', (False, True))
def test_held_content_without_a_thought_signature_has_no_expectation_holes(relation):
    from dataclasses import replace
    from Queries import QueryWorkBudget
    from test_negative_expectation import _anticipating_model
    model, _, meaning = _anticipating_model()
    registry = model.grammatical_thoughts
    if relation:
        # An ordinary concept in the VP slot is not a registered operation.
        roles = meaning.roles.clone()
        roles[1] = roles[0]
        held = replace(meaning, roles=roles,
            role_refs=(meaning.role_refs[0], meaning.role_refs[0], meaning.role_refs[2]))
    else:
        held = ConceptualMeaning.from_description(meaning.roles[0])
    store = model.symbolSpace.ltm_store
    index = store.append_meaning(held, kind='observation', sentence_index=len(store))
    with model._query_boundary_scope((0,)):
        context = model._thought_grammar_context(meaning, row=0,
            work=QueryWorkBudget(64), continuation=None)
        request = registry.form('ask', store.occurrence_of(index), context=context)
        work = QueryWorkBudget(64)
        result = model._selected_thought_expectation(request, row=0, work=work)
    assert result[-4:-1].tolist() == [0., 0., 0.]
    assert work.spent == 1


def test_negative_trust_derived_pair_reverses_only_on_the_relation_edge():
    from dataclasses import replace
    from test_clause_storage import clause_store, part_clause
    store, _ = clause_store()
    clause = part_clause()
    clause = replace(clause, meaning=replace(clause.meaning, polarity=False))
    edge = store.write_clause(clause, trust=.7)
    assert len(store) == 1
    assert store.row(edge)['evidence'] == pytest.approx((0., .7))
    assert store.row(0)['evidence'] == pytest.approx((.7, 0.))
    assert not store.raw_meaning_of(0).polarity
    torch.testing.assert_close(store.slots[0, 0], clause.meaning.roles[2])


def test_pending_forgetting_pressure_survives_restore_below_high_water():
    from Forgetting import forget
    from test_forgetting import linked_store, config
    store, child, parent = linked_store(child_is_relation=True)
    cfg = config(trust_weight=1, utility_weight=0, luminosity_weight=0, age_weight=0)
    with pytest.warns(RuntimeWarning, match='low water'):
        report = forget(store, cfg, document_boundary=True)
    assert report['after'] == 3 and store.index_of_row(parent[2]) is None
    assert store.index_of_row(child[2]) is not None and bool(store.forget_pending)
    restored = LTM0(4, capacity=4)
    restored.load_state_dict(store.state_dict())
    restored.load_semantic_extras(store.semantic_extras())
    assert bool(restored.forget_pending)
    report = forget(restored, cfg, document_boundary=True)
    assert report['target_reached'] and not bool(restored.forget_pending)
    assert restored.index_of_row(child[2]) is None


@pytest.mark.parametrize('failure', (False, True))
def test_truth_ingestion_restores_caller_recall_and_isolates_each_source(failure):
    from contextlib import contextmanager, nullcontext
    from types import SimpleNamespace, MethodType
    from Models import BasicModel
    from MediumTermMemory import MediumTermMemory
    caller = {4: [object()]}
    caller_pool = MediumTermMemory()
    owner = SimpleNamespace(mtm=caller_pool)
    model = SimpleNamespace(_what_recall_sentence_history=caller, _what_recall_batch_size=5,
        _concept_owner=lambda: owner,
        eval=lambda: None, set_sigma=lambda value: None, _what_memory=lambda: None,
        conceptualSpaces=(), data=SimpleNamespace(train_input=['source']),
        _runtime_batch=lambda *args, **kwargs: nullcontext(),
        inputSpace=SimpleNamespace(prepInput=lambda values: torch.zeros(1, 4)))
    model._clear_recall_history = MethodType(BasicModel._clear_recall_history, model)
    store = SimpleNamespace()
    @contextmanager
    def assertions(**kwargs):
        store.rows = []
        yield store.rows
    store.clause_assertions = assertions
    def present(**kwargs):
        assert model._what_recall_sentence_history == {}
        assert model._what_recall_batch_size is None
        assert owner.mtm is not caller_pool
        model._what_recall_sentence_history[0] = [object()]
        model._what_recall_batch_size = 1
        if failure:
            raise RuntimeError('failed presentation')
        store.rows.append(12)
    model.runBatch = present
    if failure:
        with pytest.raises(RuntimeError, match='failed presentation'):
            BasicModel._ltm_ingest_truth_texts(model, store, ['first', 'second'])
    else:
        assert BasicModel._ltm_ingest_truth_texts(model, store, ['first', 'second']) == [(12,), (12,)]
    assert model._what_recall_sentence_history is caller
    assert model._what_recall_batch_size == 5
    assert owner.mtm is caller_pool
