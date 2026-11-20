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
    attend, interpret = chooser.attend, chooser.choose_interpret
    events, scopes = [], []
    declined = False

    def first_focus(keys, legal, space, **kw):
        live = legal.clone()
        live[...,2:] = False
        if leave_first and declined:
            live[:,0] = False
        flat = live.flatten(1)
        kw['replay_action'] = torch.where(flat.any(-1),flat.long().argmax(-1),flat.shape[1])
        events.append(('attend',len(events)))
        return attend(keys,legal,space,**kw)

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

    perceive = SentenceLooks.perceive
    def look(current,interval,active):
        result = perceive(current,interval,active)
        torch.testing.assert_close(current._concept_owner()._passback_scope_where,
            interval/current._staged_concepts_in.shape[-1])
        scopes.append(interval.clone())
        return result

    monkeypatch.setattr(chooser,'attend',first_focus)
    monkeypatch.setattr(chooser,'choose_interpret',optional)
    monkeypatch.setattr(ModelAttention,'observe_percept',brick)
    monkeypatch.setattr(SentenceField,'remember',word_step)
    monkeypatch.setattr(SentenceLooks,'perceive',look)
    try:
        with torch.no_grad():
            model.understand(model.inputSpace.prepInput([text]))
        calls = [i for i,event in enumerate(events) if event[0]=='attend']
        words = [event[1] for event in events if event[0]=='word']
        bricks = [event[1] for event in events if event[0]=='brick']
        assert bricks == words and len(set(words)) == len(words)
        assert words == ([2,4] if leave_first else [0,2,4])
        if not leave_first:
            assert next(i for i,event in enumerate(events) if event[0]=='word') < calls[-1]
        readings = model._sentence_look_fields[0][0]
        assert len(readings) == len(calls)+1
        assert len(scopes) == len(calls)
        for reading,scope in zip(readings[1:],scopes):
            torch.testing.assert_close(reading['where'],scope)
        assert any(not entry['field'].evidence.any() or
                   int((model._attention_spans[0,:,0]>=entry['where'][0,0]).sum())>1
                   for entry in readings)
        placed = model._last_sentence_field['trials'][0]['admission'].bool()
        assert int(placed.sum()) == len(words)
    finally:
        model.End()


def test_4_word_level_focus_spends_one_look_and_leaves_interpretation_to_grammar():
    from Attention import open_narrowing, narrow_step, narrowing_view, interpretation_at, record_interpretation

    class Focus:
        attention_operations = tuple(range(6))

        def attend(self, keys, legal, space, **options):
            # This exact known word gives only focus at its current level.
            assert legal.nonzero().tolist() == [[0, 0, 1]]
            weight = keys.new_ones(1)
            return (legal.flatten(1).long().argmax(-1), weight, torch.tensor([False]),
                    dict(probability=weight, alternative_count=torch.zeros_like(weight)))

    spans = torch.tensor([[[0, 3]]])
    keys = torch.tensor([[[1., 0.]]])
    slot = torch.tensor([0])
    for pair in ((1., 0.), (0., 1.)):
        state = open_narrowing(Focus(), keys, spans, torch.tensor([[True]]),
                               budget=3, poles=torch.tensor([[pair]]))
        original = narrowing_view(state)[0]
        for turn in range(3):
            narrow_step(state)
            assert state.table.spent.item() == turn + 1
            for name in ('intervals', 'valid', 'done', 'level', 'space'):
                torch.testing.assert_close(getattr(state.table, name), getattr(original, name))
            assert not state.accepted.any() and not state.descended.any()
            index, allowed = interpretation_at(state, slot)
            assert index.item() == 0 and allowed.item()
            if turn < 2:
                record_interpretation(state, slot, torch.tensor([False]))
        assert not narrowing_view(state)[8].any()
        record_interpretation(state, slot, allowed)
        assert state.table.done[0, 0] and state.accepted.item() and state.table.spent.item() == 3
    for pair in ((0., 0.), (1., 1.)):
        state = open_narrowing(Focus(), keys, spans, torch.tensor([[True]]),
                               budget=3, poles=torch.tensor([[pair]]))
        assert not interpretation_at(state, slot)[1].item()


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
    attend, interpret, depart, run = chooser.attend, chooser.choose_interpret, SentenceCredit.departure, Looks.run
    paths, arrivals = [], []

    def ordered(keys, legal, space, **options):
        if options.get('masked_action') is None:
            focus = legal.clone()
            focus[..., 2:] = False
            first = focus.flatten(1).long().argmax(-1)
            options['replay_action'] = torch.where(focus.flatten(1).any(-1), first, -1)
        return attend(keys, legal, space, **options)

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

    monkeypatch.setattr(chooser, 'attend', ordered)
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
            torch.testing.assert_close(base.fields[i]['where'], other.fields[i]['where'])
            assert base.decisions[i]['index'].equal(other.decisions[i]['index'])
            if i < turn:
                assert base.decisions[i]['action'].equal(other.decisions[i]['action'])
        assert base.decisions[turn]['interpreted'].item()
        assert not other.decisions[turn]['interpreted'].item()
    finally:
        model.End()


def test_9_unfocus_restores_unread_parent_and_costs_one_act():
    from Attention import open_narrowing, narrow_step, narrowing_view, record_interpretation
    class Chooser:
        attention_operations = (0, 1, 2)
        action = 1
        def attend(self, keys, legal, space, **options):
            assert legal.shape[-1] == 6 and legal.flatten(1)[0, self.action]
            return torch.tensor([self.action]), torch.ones(1), torch.tensor([False]), {
                'probability': torch.ones(1), 'alternative_count': torch.zeros(1)}
    chooser = Chooser()
    state = open_narrowing(chooser, torch.eye(3)[None], torch.tensor([[[0,3],[4,7],[8,14]]]),
                           torch.ones(1,3,dtype=torch.bool), budget=8)
    parent = narrowing_view(state)[0]
    narrow_step(state)
    assert narrowing_view(state)[8][0,0,2]
    chooser.action = 2
    narrow_step(state)
    restored = narrowing_view(state)[0]
    for key in ('intervals','level','space','valid','done'):
        assert getattr(restored,key).equal(getattr(parent,key))
    assert restored.spent.item() == 2 and not narrowing_view(state)[8][...,2].any()
    chooser.action = 1
    narrow_step(state)
    record_interpretation(state,torch.tensor([0]),torch.tensor([True]))
    assert not narrowing_view(state)[8][...,2].any()


def test_8_fourth_packed_sentence_can_use_the_shared_allowance(monkeypatch):
    import ModelAttention
    from Attention import narrow_step, interpretation_at, record_interpretation
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
        state = model._attention_open(sentence=sid, spent_override=torch.tensor([meter.spent])).state
        assert state.table.valid.shape == (1, 2)
        for turn in (0, 1):
            narrow_step(state)
            word, allowed = interpretation_at(state, state.look_slot)
            assert allowed.item() and word.item() == 2 * sid + turn
            record_interpretation(state, state.look_slot, allowed)
        meter.require('bracket', int(state.table.spent[0]) - meter.spent)
        assert meter.remaining == 2 * (3 - sid)
    assert meter.spent == 8


@pytest.mark.parametrize('bias', [0., 1.])
@pytest.mark.parametrize('length', range(1, 7))
def test_11_lesson_teaches_shortest_walk_and_interpretation(length, bias, monkeypatch):
    from types import SimpleNamespace
    import SentenceLooks
    import ModelAttention
    from Attention import open_narrowing
    from AttentionLesson import reading_lesson
    from Language import OperationSelectionLayer
    word_rule = torch.nn.Identity()
    word_rule.rule_name = 'interpret'
    chooser = OperationSelectionLayer(d_model=4, unary_ops=[word_rule])
    stand_in(chooser)
    chooser.interpret_bias = bias
    chooser.attention_operations = tuple(range(6))
    spans = torch.arange(length)[:,None] * 2 + torch.tensor([[0,1]])
    keys = torch.ones(1,length,4)
    state = open_narrowing(chooser,keys,spans[None],torch.ones(1,length,dtype=torch.bool),budget=12)
    forms = [[str(i) for i in range(length)]]
    model = SimpleNamespace(training=True, _sentence_training=True, _attention_forms=(forms,),
                            _sentence_open_fields={0:None})
    walk = SimpleNamespace(state=state,width=length)
    monkeypatch.setattr(ModelAttention,'handoff',lambda *args: None)
    monkeypatch.setattr(SentenceLooks,'perceive',lambda *args: None)
    arrivals=[]
    with reading_lesson(model, forms):
        looks=SentenceLooks.Looks(model,walk,0,torch.tensor([True]))
        looks.run(lambda i, rows: arrivals.append(int(i)))
    assert arrivals == list(range(length))
    assert len(state.actions) == (0 if length==1 else length)
    taught = looks.lesson_cost()
    assert taught.shape == (1,)
    if bias == 1.:
        assert not any(d['eligible'].any() for d in looks.decisions)
        assert int(looks.teacher.count) == len(state.actions)
        assert all(not d['probability'].requires_grad for d in looks.decisions)
        if length == 1:
            assert taught.item() == 0
        return
    assert taught.item() > 0
    taught.sum().backward()
    assert chooser.stand_in.unary.grad is not None and chooser.stand_in.unary.grad.abs().sum() > 0


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

    No act of attention accepts a word (§60.4), and the grammar interprets
    only at a sentence's looks. A read with no sentence (a stem read, as the
    word evaluator makes) is attention's walk alone: it reaches the words and
    admits none. Alec, 2026-11-19 (1.8 §68.89): "we will just output nothing
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
    assert int(reading.table.done.sum()) == 0
