"""Opaque-word corpus and native model boundaries for thinking spec section 10."""
from pathlib import Path
import re

import pytest
import torch

from math_chain_corpus import (BEYOND_PAIRS, HELD_OUT_PAIRS, NUMBER_WORDS,
                               MathChainCorpus, flatten)

ROOT = Path(__file__).resolve().parents[1]


def test_counting_facts_have_both_forms_and_owned_statement_references():
    facts = MathChainCorpus().counting()
    assert len(facts) == 40
    for value in range(20):
        direct, referenced = facts[2 * value:2 * value + 2]
        assert direct.sentences == (f'{NUMBER_WORDS[value]} plus one is {NUMBER_WORDS[value+1]}.',)
        assert referenced.sentences == (f'{NUMBER_WORDS[value].capitalize()} plus one.',
                                       f'Ref(1) is {NUMBER_WORDS[value+1]}.')
        assert referenced.statement_references == ((1, 0),)
        assert direct.question is None and referenced.answer is None


def test_held_out_and_beyond_sources_end_at_question_and_have_no_worked_answers():
    corpus = MathChainCorpus()
    assert len(corpus.train_pairs) == 117
    assert set(corpus.train_pairs).isdisjoint(HELD_OUT_PAIRS)
    for split, pairs in (('test', HELD_OUT_PAIRS), ('beyond', BEYOND_PAIRS)):
        docs = corpus.presentation()[split]
        assert tuple(doc.pair for doc in docs) == pairs
        for doc in docs:
            assert doc.question == len(doc.sentences) - 1 == 4
            assert doc.sentences[-1] == 'what is y ?'
            assert all('the answer' not in line for line in doc.sentences)
            assert len(doc.steps) == doc.pair[1]
            assert doc.steps[-1][2] == doc.answer


def test_worked_steps_are_chronological_and_the_zero_case_has_no_successor_step():
    corpus = MathChainCorpus()
    doc = corpus.problem((3, 2), split='train', training=True)
    assert doc.sentences[doc.question + 1:] == (
        'two is one plus one.', 'three plus one is four.',
        'four plus one is five.', 'the answer is five.')
    zero = corpus.problem((3, 0), split='train', training=True)
    assert zero.steps == () and zero.sentences[-1] == 'the answer is three.'
    assert not any(re.search(r'\d', line) for line in doc.sentences)


def test_expectation_ablation_withholds_the_answer_line_and_all_answer_labels():
    corpus = MathChainCorpus()
    docs = corpus.presentation(answer_line=False)['train']
    texts, targets, _ = flatten(docs, supplied=False)
    assert all(value is None for value in targets)
    assert not any('the answer is' in line for line in texts)
    assert any('plus one is' in line for line in texts)


def test_premises_shuffle_without_reordering_worked_steps():
    corpus = MathChainCorpus()
    # Observe the shuffle call, not a probabilistic assertion on sampled order.
    calls = []
    def rotate(values):
        calls.append(tuple(values))
        values[:] = values[1:] + values[:1]
    corpus.generator.rng.shuffle = rotate
    doc = corpus.problem((3, 2), split='train', training=True)
    assert len(calls) == 1 and len(calls[0]) == 4
    assert doc.sentences[0] == 'y is x plus two.'
    assert doc.steps == (('three', 'one', 'four'), ('four', 'one', 'five'))


def test_text_answers_are_loss_side_and_not_numeric_output_vectors():
    from data import Data
    from What import What
    data = Data()
    data.load('math_chain')
    question = next(i for i, value in enumerate(data.text_answers['train']) if value is not None)
    assert data.what(What.supervised(question)).what in NUMBER_WORDS
    assert not data.what(What.supervised(next(i for i, value in enumerate(
        data.text_answers['train']) if value is None))).available
    assert all(value.shape == (1,) and not bool(value.any()) for value in data.train_output)
    assert not hasattr(data, 'math_range')
    assert data.grammar_lessons == {'train': [], 'validation': [], 'test': []}
    data.refresh_math_chain(answer_line=False)
    assert not data.has_supervised_outputs
    assert all(value is None for value in data.text_answers['train'])
    data.load('xor')
    assert data.text_answers is None


def build_model(config=None):
    from data import TheData
    from Models import BaseModel
    from util import init_config
    config = config or ROOT / 'data/MM_math_chain.xml'
    init_config(path=str(config), defaults_path=str(ROOT / 'data/model.xml'))
    import Language
    Language.TheGrammar._configured = False
    TheData.load('math_chain')
    return BaseModel.from_config(str(config), data=TheData)[0]


def test_zero_budget_loads_and_trains_the_ordinary_vp_without_attention(tmp_path, eager_reading):
    from MathChainTraining import present
    config = tmp_path / 'zero.xml'
    config.write_text((ROOT / 'data/MM_math_chain.xml').read_text().replace(
        '<attentionBudget>32</attentionBudget>', '<attentionBudget>0</attentionBudget>'))
    model = build_model(config)
    data = model.inputSpace.data
    texts, labels, addresses = flatten(data.math_chain_corpus.counting()[:2], supplied=False)
    data.train_input, data.train_output = texts, [torch.zeros(1) for _ in texts]
    data.text_answers['train'] = labels
    data.source_addresses['train'] = [dict(value, split='train') for value in addresses]
    observations = []
    def observe(model, *_args):
        observations.append((int(model.inputSpace._word_active_mask.sum()),
                             bool(model._tensor_pushed_ideas.abs().any())))
        assert model._attention_words is None
        assert all(meter.spent == 0 for meter in model._attention_meters)
    try:
        assert model.attention_budget == 0
        report = present(model, split='train', optimizer=model.getOptimizer(lr=.001),
                         after_batch=observe)
        assert report['sentences'] == 3
        assert all(words > 0 and composed for words, composed in observations)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_math_chain_model_has_no_numeric_answer_head_and_uses_existing_operations(monkeypatch):
    monkeypatch.setattr(torch, 'manual_seed', lambda *_: pytest.fail('fixed learner seed'))
    model = build_model()
    try:
        assert tuple(model.outputSpace.outputShape) == (1, 1)
        assert model.attention_budget >= 12
        assert 'plus' not in model.grammatical_thoughts.executable_operation_ids
        assert model.inputSpace.data.text_answers is not None
        assert model.inputSpace.data.grammar_lessons['train'] == []
        assert not any(getattr(module, 'out_features', None) == len(NUMBER_WORDS)
                       for module in model.modules())
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_document_batches_keep_every_sentence_and_do_not_mix_document_context():
    from MathChainTraining import document_batches
    docs = MathChainCorpus().presentation()['train']
    _, _, addresses = flatten(docs, supplied=False)
    batches = list(document_batches(addresses, 8))
    seen, streams = [], {}
    for indices, end in batches:
        for row, index in enumerate(indices):
            address = addresses[index]
            assert address['sentence'] == streams.get(row, (address['document'], -1))[1] + 1
            assert address['document'] == streams.get(row, (address['document'],))[0]
            streams[row] = address['document'], address['sentence']
            seen.append(index)
        if end:
            streams.clear()
    assert sorted(seen) == list(range(len(addresses))) and not streams


def test_native_worked_document_trains_unforced_without_numeric_oracles(monkeypatch, eager_reading):
    """Construction/optimizer smoke, not a learning-gate attempt."""
    import exact
    from MathChainTraining import present
    model = build_model()
    data = model.inputSpace.data
    docs = (*data.math_chain_corpus.counting()[:2],
            data.math_chain_corpus.problem((3, 2), split='train', training=True))
    texts, targets, addresses = flatten(docs, supplied=True)
    data.train_input, data.train_output = texts, [torch.zeros(1) for _ in texts]
    data.text_answers['train'] = targets
    data.source_addresses['train'] = [dict(value, split='train') for value in addresses]
    # Generation is finished. The runtime has only the ordinary source stream.
    def forbidden(*_args, **_kwargs):
        pytest.fail('the learner called an exact-arithmetic oracle')
    monkeypatch.setattr(exact, 'MathProblemGenerator', forbidden)
    data.math_chain_corpus = data.math_chain_documents = None
    monkeypatch.setattr(torch, 'manual_seed', forbidden)
    reference_reads = []
    reference_bank = model._sentence_reference_bank
    def read_references(*args, **kwargs):
        value = reference_bank(*args, **kwargs)
        reference_reads.append(value)
        return value
    monkeypatch.setattr(model, '_sentence_reference_bank', read_references)
    try:
        report = present(model, split='train', optimizer=model.getOptimizer(lr=.001))
        assert report['sentences'] == len(texts)
        # Both suffixes use the bank rebuilt before the greedy walk. The
        # detached fork does not re-enumerate an independently changed bank.
        assert len(reference_reads) == report['batches']
        assert model._last_equality_learning['counts'] is not None
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_selected_equality_credits_the_current_verb_and_not_an_arithmetic_label(monkeypatch):
    from types import SimpleNamespace
    from EqualityLearning import cost
    from Language import VerbLayer
    verb = VerbLayer(8, 8)
    # Ordinary opaque codes. There is no numeral index or arithmetic target.
    head, obj, subject, target = torch.randn(4, 8).tanh()
    phrase = verb(subject[None], verb(head[None], obj[None]))[0]
    program = SimpleNamespace(actions=torch.tensor([[1, 0, -1]]),
        operation_values=torch.stack((phrase, target, phrase * 0))[None])
    language = SimpleNamespace(_compose_binary_rules=(SimpleNamespace(method_name='equal'),))
    value, counts = cost(language, [program], torch.tensor([True]), phrase[None])
    value.sum().backward()
    assert counts == [1]
    assert verb._verb_shift.grad is not None and verb._verb_shift.grad.abs().sum() > 0


def test_bound_answer_is_an_identity_not_a_nearest_numeral_class():
    from types import SimpleNamespace
    from Meaning import ConceptualMeaning
    from ThoughtReferences import question, fill
    from BindingAnswers import matches
    goal = question(ConceptualMeaning.from_description(torch.ones(8)), (('referent', 0),))
    answer = fill(goal, dict(reference=('sym', 81), value=torch.ones(8),
                            support_true=1., support_false=0.))
    owner = SimpleNamespace(word_concepts=lambda word: (81,) if word == 'alpha' else (82,))
    model = SimpleNamespace(_concept_owner=lambda: owner, symbolSpace=SimpleNamespace(ltm_store=None))
    assert matches(model, answer, 'alpha')
    assert not matches(model, answer, 'beta')  # even with identical numeric content
    assert not matches(model, goal, 'alpha')


def test_text_target_preserves_the_live_batch_attention_and_source(monkeypatch):
    import ModelAttention
    model = build_model()
    try:
        model._lex_embed_stem(model.inputSpace.prepInput(['one plus one.', 'two plus one.']))
        meters = model._attention_meters
        spent = [(meter.spent, dict(meter.counts)) for meter in meters]
        words, prior = model._attention_words, model._word_expectation
        objective = model._objective_query
        lesson = model.teacher.lesson
        stats = list(model._word_unit_stats)
        live = model.inputSpace._ar_embedded.detach().clone()
        memory = model._what_memory()
        histories = memory._what_slots
        def forbidden(*_args, **_kwargs):
            pytest.fail('an answer target entered the live attention episode')
        monkeypatch.setattr(ModelAttention, 'stage_input', forbidden)
        monkeypatch.setattr(model.symbolSpace, 'ensure_microbatch', forbidden)
        target = model._embed_answer_texts(['twenty'])
        assert target.shape[0] == 1 and not target.requires_grad
        assert model._attention_meters is meters and len(meters) == 2
        assert [(meter.spent, dict(meter.counts)) for meter in meters] == spent
        assert model._attention_words is words and model._word_expectation is prior
        assert model._objective_query is objective and model.teacher.lesson is lesson
        assert model._word_unit_stats == stats
        assert memory.batch == 2 and memory._what_slots is histories
        torch.testing.assert_close(model.inputSpace._ar_embedded, live)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.parametrize('has_point', [True, False])
def test_prior_statement_is_a_reference_candidate_before_noun_columns_exist(has_point):
    from types import SimpleNamespace
    from Models import BasicModel
    from ReferenceContext import SituationFrame
    from WhereRegistry import WhereRegistry
    frame = SituationFrame(1, torch.eye(4)[:3], torch.tensor([True, False, False]),
                           12345, torch.eye(4)[0] if has_point else None)
    discourse = SimpleNamespace(_inter_chain_window=2, _inter_last_meaning=[None],
                                situation_references=lambda row: (frame,))
    owner = SimpleNamespace(components=None)
    model = SimpleNamespace(symbolSpace=SimpleNamespace(expectation=discourse),
        languageSpace=SimpleNamespace(_compose_binary_rules=()),
        _concept_owner=lambda: owner, _sentence_reference_types=lambda _: None,
        conceptualSpace=SimpleNamespace(),
        where_registry=WhereRegistry((('conceptual', 8),)))
    bank = BasicModel._sentence_reference_bank(model, torch.tensor([True]), torch.zeros(1, 4))
    assert bank.ids[bank.valid].tolist() == [12345]
    assert bank.relations[bank.valid].tolist() == [not has_point]
    assert bank.column_ids.tolist() == [12345]
    if has_point:
        torch.testing.assert_close(bank.values[bank.valid][0], frame.point)


def test_thought_menu_checks_open_reference_types_before_choice():
    from dataclasses import replace
    from test_normal_thought_controller import _catalog_world
    from Occurrence import ADDRESS_DOMAIN
    model, registry, memory, part, whole = _catalog_world()
    root = registry.form('isPart', part, whole)
    # An existing statement occurrence is a valid closed relation operand,
    # but the open taxonomy face requires a native concept endpoint.
    source = replace(root, role_refs=(('ltm', ADDRESS_DOMAIN, 12345),
                                      root.role_refs[1], ('ltm', ADDRESS_DOMAIN, 67890)))
    menu = registry.controller_candidates(source, source, source)
    assert menu
    for item in menu:
        registry.signature_for(item.request, verify_reference=False)
    assert not any(item.open_roles for item in menu)
