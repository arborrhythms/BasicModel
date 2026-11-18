"""Sentence admission owns its walk; non-sentence reads retain their owner."""
from types import SimpleNamespace

import pytest
import torch

from test_attention_thinking_handoff import first_legal


def prepared_model(monkeypatch, *, sentence):
    import ModelAttention
    chooser = first_legal()
    chooser.d_model = 2
    owner = SimpleNamespace(definitions={}, similarity_codebook=None,
        priming_weights=lambda **kwargs: None,
        interpret=SimpleNamespace(word_for_form=lambda form: form))
    model = SimpleNamespace(
        inputSpace=SimpleNamespace(_ar_embedded_N=torch.eye(2)[None],
            _word_active_mask=torch.ones(1, 2, dtype=torch.bool),
            _ar_word_part_offsets=torch.tensor([[[0, 3], [4, 7]]]),
            _packed_sentence_ids=torch.zeros(1, 2, dtype=torch.long)),
        conceptualSpace=SimpleNamespace(_word_column=torch.ones(1, 2, 2)),
        perceptualSpace=SimpleNamespace(_forward_input={'word_texts': [['red', 'cat']],
            'word_intervals': [[(0, 3), (4, 7)]]}),
        languageSpace=torch.nn.Module(),
        _staged_concepts_in=torch.tensor([list(b'red cat')]),
        _concept_owner=lambda: owner, _stm_reducer=lambda: chooser,
        _sentence_ends=sentence, attention_budget=32,
        grammar_phases=SimpleNamespace(examination=True, vector=True))
    monkeypatch.setattr(ModelAttention, 'stage_percept_expectation', lambda *args: None)
    monkeypatch.setattr(ModelAttention, 'native_word_poles', lambda model, spans, forms, known:
                        torch.stack((known, torch.zeros_like(known)), -1).float())
    return model


def test_sentence_preparation_does_not_walk_or_publish_admission(monkeypatch):
    import ModelAttention
    model = prepared_model(monkeypatch, sentence=True)
    def forbidden(*args, **kwargs):
        pytest.fail('a row walk ran during sentence preparation')
    monkeypatch.setattr(ModelAttention, 'narrow_words', forbidden)
    ModelAttention.stage_input(model)
    assert model._attention_words is None
    assert model._attention_greedy is None
    assert model._word_expectation is None
    assert model.conceptualSpace._word_column is None
    assert model._attention_meters[0].spent == 0
    walk = model._attention_open(sentence=0)
    assert walk.state.steps_taken == 0
    from Attention import narrow_step, interpretation_at, record_interpretation
    narrow_step(walk.state)
    word, allowed = interpretation_at(walk.state, walk.state.look_slot)
    assert allowed.all() and not walk.state.accepted.any()
    record_interpretation(walk.state, walk.state.look_slot, allowed)
    assert walk.state.accepted[0,word].all()
    assert walk.state.steps_taken > 0


def test_non_sentence_configuration_keeps_one_staged_charged_walk(monkeypatch):
    import ModelAttention
    model = prepared_model(monkeypatch, sentence=False)
    read = ModelAttention.narrow_words
    calls = []
    def counted(*args, **kwargs):
        calls.append(None)
        return read(*args, **kwargs)
    monkeypatch.setattr(ModelAttention, 'narrow_words', counted)
    ModelAttention.stage_input(model)
    assert len(calls) == 1
    assert not model._attention_words.accepted.any()  # examination alone never interprets
    assert model._attention_meters[0].spent == int(model._attention_words.table.spent[0])


def test_disabled_examination_does_not_offer_a_sentence_walk(monkeypatch):
    import ModelAttention
    model = prepared_model(monkeypatch, sentence=True)
    model.grammar_phases.examination = False
    ModelAttention.stage_input(model)
    assert model._attention_open is None
    assert model._attention_words is None


def test_expectation_preserves_the_kept_prefix_and_row_candidate_bank(monkeypatch):
    import ModelAttention
    from WordExpectation import ExpectedWords
    calls = []
    codes = torch.eye(4)[None]
    current = torch.tensor([[False, False, True, False]])
    earlier = torch.tensor([[True, False, False, False]])
    lanes = torch.tensor([[[0., 0.], [0., 0.], [1., 0.], [0., 0.]]])
    prefix = SimpleNamespace(accepted=earlier,
        poles=torch.tensor([[[0., 1.], [0., 0.], [0., 0.], [0., 0.]]]))
    def expect(kind, words, bank, valid, targets, **kwargs):
        calls.append((kind, bank, valid, targets, kwargs))
        active = kwargs['active']
        field = torch.zeros(1, 4, 4, 2)
        scalar = torch.zeros(1, 4)
        return ExpectedWords(torch.zeros(1, 4, 4), words, words*active[..., None],
            active.float(), field, field, scalar, active, scalar, field)
    model = SimpleNamespace(
        _attention_forms=([['a', 'b', 'c', 'd']], torch.arange(4)[None], codes,
                          torch.ones(1, 4, dtype=torch.bool)),
        _attention_words=SimpleNamespace(accepted=current), _attention_poles=lanes,
        inputSpace=SimpleNamespace(_ar_word_object_atoms=codes),
        conceptualSpace=SimpleNamespace(),
        symbolSpace=SimpleNamespace(expectation=SimpleNamespace(enabled_levels=('word',), expect=expect)),
        word_expectation_gain=1., expectation_epsilon=1e-6, expectation_saturation='clip')
    monkeypatch.setattr(ModelAttention, 'stage_symbol_bank', lambda *args: None)
    ModelAttention.stage_expectation(model, prefix=prefix)
    assert len(calls) == 1
    _, bank, valid, targets, kwargs = calls[0]
    assert torch.equal(bank, codes)
    assert valid.all()
    assert targets.tolist() == [[0, -1, 2, -1]]
    assert kwargs['lanes'].tolist() == [[[0., 1.], [0., 0.], [1., 0.], [0., 0.]]]
    assert torch.equal(model._word_expectation_mask, earlier | current)


def test_ordinary_composition_keeps_its_unrestricted_leaf_mask(monkeypatch):
    """Exercise the zero-attention control at the native lesson capacity."""
    import inspect
    import ModelAttention
    from Models import BasicModel
    from attention_thinking_certificate import lesson
    from reading_fixtures import use_eager_reading

    use_eager_reading(monkeypatch)
    stage = ModelAttention.stage_input
    def ordinary(model):
        model.attention_budget = 0
        return stage(model)
    monkeypatch.setattr(ModelAttention, 'stage_input', ordinary)

    class ReachedComposition(Exception):
        pass

    restore = BasicModel._restore_sentence_state
    def restoring(model, *args, **kwargs):
        caller = inspect.currentframe().f_back.f_code.co_name
        if caller == 'start_compose':
            assert model._attention_open is None
            assert model.inputSpace._ar_grammar_leaf_mask is None
            raise ReachedComposition
        return restore(model, *args, **kwargs)
    monkeypatch.setattr(BasicModel, '_restore_sentence_state', restoring)
    observer = SimpleNamespace(reference_seconds=0., records=[], calls=[], actions=[], costs=[])
    with pytest.raises(ReachedComposition):
        lesson(observer)
