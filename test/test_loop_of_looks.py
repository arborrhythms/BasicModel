"""Certificates of item 1.8 §60: one causal loop of looks."""
from dataclasses import fields
import torch
import pytest


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
def test_1_2_3_words_join_before_the_walk_finishes(leave_first, monkeypatch):
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
        'doc/benchmarks/2026-11-11-whole-paths/configurations/explore-1.xml',data=TheData)[0]
    admit_training_forms(model,[text])
    model.eval()
    chooser = model._stm_reducer()
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
