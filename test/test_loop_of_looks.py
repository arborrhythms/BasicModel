"""Certificates of item 1.8 §60: one causal loop of looks."""
from dataclasses import fields
import torch
import pytest
from chooser_steering import stand_in


def explore_configuration(tmp_path):
    """The whole-paths benchmark's configuration (a receipt under doc/), less
    the key conceptualPi that sigma-pi §21, rule 6 removed: a basic percept
    is a row of the first Pi in every model."""
    from pathlib import Path
    source = Path('doc/benchmarks/2026-11-11-whole-paths/configurations/explore-1.xml').read_text()
    path = tmp_path / 'explore-1.xml'
    path.write_text(source.replace('<conceptualPi>true</conceptualPi>', '')
                    .replace('<transformChooser>mlp</transformChooser>', ''))
    return str(path)


def test_7_incremental_word_forecast_follows_arrival_order(monkeypatch):
    from WordExpectation import WordArrivals, word_distribution
    from Layers import SymbolExpectation
    owner = SymbolExpectation(n_symbols=4, max_depth=4, n_dim=4, concept_dim=4)
    bank = torch.eye(4)[None]
    owner.set_symbol_bank(bank[0])
    valid = torch.ones(1, 4, dtype=torch.bool)
    order = torch.tensor([[2, 0, 3]])
    sequence = bank[:, order[0]]
    expected = word_distribution(owner, sequence, bank, valid, order)
    reader = WordArrivals(owner, bank, valid)
    other = WordArrivals(owner, bank, valid)
    # The opening owns its initial STM context. No later arrival rereads it.
    monkeypatch.setattr(owner, '_read_context', lambda *a: (_ for _ in ()).throw(
        AssertionError('word arrivals reopened their initial context')))
    readings = [reader.step(sequence[:, i], order[:, i]) for i in range(3)]
    for field in fields(expected):
        actual = torch.cat([getattr(reading, field.name) for reading in readings], 1)
        torch.testing.assert_close(actual, getattr(expected, field.name), rtol=0, atol=0)
    first = other.step(bank[:, 1], torch.tensor([1]))
    # A different first arrival changes no forecast before that arrival.
    torch.testing.assert_close(first.activation, readings[0].activation, rtol=0, atol=0)
    second = other.step(bank[:, 0], torch.tensor([0]))
    assert not torch.equal(second.activation, readings[1].activation)


@pytest.mark.usefixtures('eager_reading')
@pytest.mark.parametrize('leave_first', [False, True])
def test_1_2_3_words_join_before_the_walk_finishes(leave_first, monkeypatch, tmp_path):
    import Attention
    import ModelAttention
    import SentenceLooks
    from SentenceAttention import SentenceField
    from Models import BaseModel
    from data import TheData
    from Corpus import admit_training_forms
    text = 'red cat sleeps'
    TheData.processLM({split: dict(text=[text] if split=='train' else [], label=[])
                      for split in ('train','validation','test')})
    model = BaseModel.from_config(
        explore_configuration(tmp_path),data=TheData)[0]
    admit_training_forms(model,[text])
    # MTM nominates a form on recurrence (sigma-pi §21, rule 20).
    from nomination_fixtures import nominated_sentences; nominated_sentences(model, [text])
    model.eval()
    chooser = model._stm_reducer()
    chooser.interpret_bias = 0.  # Exercise the learned choice, including abstention.
    interpret = chooser.choose_interpret
    from reading_fixtures import SourceOrder
    model._attention_supplied_reading = SourceOrder()
    events, extents = [], []
    declined = False

    def optional(word, allowed, **kw):
        nonlocal declined
        take = allowed.clone()
        if leave_first and not declined and bool(allowed.any()):
            take.zero_()
            declined = True
        kw['replay_action'] = take.long()
        return interpret(word,allowed,**kw)

    observe = ModelAttention.observe_percept
    def brick(current,forecast,payload,sub,index,active):
        events.append(('brick',int(index)))
        return observe(current,forecast,payload,sub,index,active)

    remember = SentenceField.remember
    def word_step(field,index,rows,language,*,closing=False):
        if not closing and bool(rows.any()):
            events.append(('word',int(index)))
        return remember(field,index,rows,language,closing=closing)

    look = SentenceLooks.look
    def counted(*args):
        result = look(*args)
        events.append(('look',len(events)))
        extents.append(result[1].detach().clone())
        return result

    monkeypatch.setattr(chooser,'choose_interpret',optional)
    monkeypatch.setattr(ModelAttention,'observe_percept',brick)
    monkeypatch.setattr(SentenceField,'remember',word_step)
    monkeypatch.setattr(SentenceLooks,'look',counted)
    try:
        with torch.no_grad():
            model.understand(model.inputSpace.prepInput([text]))
        calls = [i for i,event in enumerate(events) if event[0]=='look']
        words = [event[1] for event in events if event[0]=='word']
        bricks = [event[1] for event in events if event[0]=='brick']
        assert bricks == words and len(set(words)) == len(words)
        # A declined word is asked for again at the next look.
        assert words == [0,2,4]
        assert len(calls) == len(words) + (1 if leave_first else 0)
        assert next(i for i,event in enumerate(events) if event[0]=='word') < calls[-1]
        readings = model._sentence_look_fields[0][0]
        assert len(readings) == len(calls)+1
        placed = model._last_sentence_field['trials'][0]['admission'].bool()
        assert int(placed.sum()) == len(words)
    finally:
        model.End()


def test_10_interpret_departure_keeps_the_focus_that_reached_its_word(tmp_path, monkeypatch, eager_reading):
    """Depart at the second word's grammar decision on a real training path."""
    from test_compiled_word_chunk import _tiny_canonical_model
    from Corpus import admit_training_forms
    from What import WhatAnswer
    import SentenceCredit
    from SentenceLooks import Looks
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32,
        concept_rows=32, training_overrides={'reconstructionPlacement': 'eager'},
        architecture_overrides={'explorePaths': 1, 'omissionTarget': 1})
    model._tensor_peer_while_eager = True
    text = 'red cat sleeps'
    admit_training_forms(model, [text])
    # MTM nominates a form on recurrence (sigma-pi §21, rule 20).
    from nomination_fixtures import nominated_sentences; nominated_sentences(model, [text])
    model.inputSpace.data.has_supervised_outputs = True
    monkeypatch.setattr(model.inputSpace.data, 'what', WhatAnswer.silence)
    chooser = model._stm_reducer()
    chooser.interpret_bias = 0.  # The departure is in the learned interpretation.
    interpret, depart, run = chooser.choose_interpret, SentenceCredit.departure, Looks.run
    paths, arrivals = [], []
    from reading_fixtures import SourceOrder
    model._attention_supplied_reading = SourceOrder()

    def consume(word, allowed, **options):
        if options.get('masked_action') is None:
            options['replay_action'] = allowed.long()
        return interpret(word, allowed, **options)

    def choose(*args, **options):
        draw = depart(*args, **options)
        if 'interpretations' in options:
            eligible = options['interpretations'][0].nonzero().flatten()
            assert len(eligible) >= 2
            for name in SentenceCredit.WALKS:
                draw[name + '_round'].fill_(-1)
            draw['walk'].fill_(1)
            draw['narrowing'].zero_()
            draw['interpret_round'] = eligible[1:2]
        return draw

    def traced(path, arrive, *args, **kwargs):
        words = []
        def word_step(index, rows):
            words.append((len(path.decisions)-1, int(index)))
            return arrive(index, rows)
        result = run(path, word_step, *args, **kwargs)
        paths.append(path)
        arrivals.append(words)
        return result

    monkeypatch.setattr(chooser, 'choose_interpret', consume)
    monkeypatch.setattr(SentenceCredit, 'departure', choose)
    monkeypatch.setattr(Looks, 'run', traced)
    try:
        source = model.inputSpace.prepInput([text])
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.003), batchSize=1,
                       batch_override=(source, torch.empty(1, 0)))
        base, other = paths
        turn = int(other.draw['interpret_round'])
        assert arrivals[0][0] == arrivals[1][0] and arrivals[0][0][0] < turn
        assert [a for a in arrivals[0] if a[0] < turn] == [a for a in arrivals[1] if a[0] < turn]
        for i in range(turn+1):
            torch.testing.assert_close(base.fields[i]['extent'], other.fields[i]['extent'])
            assert base.decisions[i]['index'].equal(other.decisions[i]['index'])
            if i < turn:
                assert base.decisions[i]['action'].equal(other.decisions[i]['action'])
        assert base.decisions[turn]['interpreted'].item()
        assert not other.decisions[turn]['interpreted'].item()
    finally:
        model.End()


def test_8_fourth_packed_sentence_can_use_the_shared_allowance(monkeypatch):
    import ModelAttention
    from test_sentence_owned_attention import prepared_model

    model = prepared_model(monkeypatch, sentence=True)
    model.attention_budget = 2
    source = ['red', 'cat'] * 4
    spans = [(4 * i, 4 * i + 3) for i in range(len(source))]
    isp = model.inputSpace
    isp._ar_embedded_N = torch.eye(2).repeat(4, 1)[None]
    isp._word_active_mask = torch.ones(1, 8, dtype=torch.bool)
    isp._packed_sentence_slot_mask = torch.ones(1, 4, dtype=torch.bool)
    isp._packed_sentence_ids = torch.arange(4)[None].repeat_interleave(2, dim=1)
    model.perceptualSpace._forward_input = dict(word_texts=[source], word_intervals=[spans])
    ModelAttention.stage_input(model)
    meter, = model._attention_meters
    assert meter.remaining == 8
    for sid in range(4):
        state = model._attention_open(sentence=sid, spent_override=torch.tensor([meter.spent]))
        live = (state.spans[..., 1] > state.spans[..., 0])[0].nonzero().flatten().tolist()
        assert live == [2 * sid, 2 * sid + 1]
        assert state.limit.item() == 2
        meter.require('look', 2)
        assert meter.remaining == 2 * (3 - sid)
    assert meter.spent == 8


def test_4_verbal_xor_offers_its_declared_interpretation():
    from test_mm_xor import _fresh_model
    model, _, _ = _fresh_model('data/XOR_grammar.xml')
    try:
        chooser = model._stm_reducer()
        pure = torch.tensor([True, False])
        word = torch.zeros(2, chooser.d_model)
        choice = chooser.choose_interpret(word, pure)
        assert choice['eligible'].tolist() == [True, False]
        assert torch.isfinite(choice['logits'][:, 0]).all()
        assert torch.isfinite(choice['logits'][:, 1]).tolist() == [True, False]
        # Lexical naming is performed with the word bank at a terminal.
        # A grammar's implicit inverse menu obeys the same contract as an
        # explicitly declared generate section.
        names = model.languageSpace._generate_unary_names
        assert 'interpret' not in names
        assert {'conjunction', 'disjunction'} <= set(model.languageSpace._generate_binary_names)
    finally:
        model.End()


def test_5_a_read_with_no_sentence_admits_no_word(tmp_path, monkeypatch, eager_reading):
    """Outside a sentence nothing is admitted, and so nothing is said.

    The grammar interprets only at a sentence's looks. A read with no
    sentence (a stem read, as the word evaluator makes) takes no look and
    admits no word. Alec, 2026-11-19 (1.8 §68.89): "we will just output nothing
    for non-WFF parses."
    """
    from test_packed_reconstruction_parity import build_model
    from reading_fixtures import read_in_source_order
    model = build_model(tmp_path, word_capacity=8)
    read_in_source_order(model, monkeypatch)
    calls = []
    chooser = model._stm_reducer()
    interpret = chooser.interpret_at_look
    def observed(*args, **kwargs):
        calls.append(None)
        return interpret(*args, **kwargs)
    monkeypatch.setattr(chooser, 'interpret_at_look', observed)
    raw = model.inputSpace.prepInput(['first second last'])
    with torch.no_grad():
        model._lex_embed_stem(raw)
    reading = model._attention_words
    spans = model._attention_spans  # separators have an empty extent
    live = spans[..., 1] > spans[..., 0]
    assert int(live.sum()) == 3
    assert calls == []
    assert not reading.accepted.any()
    assert not model.inputSpace._ar_grammar_leaf_mask.bool().any()
    assert int(reading.looks.sum()) == 0
