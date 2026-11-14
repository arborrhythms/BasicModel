"""One sentence/thought clock, reversible epochs, and clock-independent reads."""
from types import SimpleNamespace
import copy

import pytest
import torch

from RowTime import encoder, HORIZON


def native(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8',
        input_width=32, concept_rows=64,
        training_overrides={'reconstructionPlacement': 'eager'},
        architecture_overrides={'ltmConsolidation': True, 'answerSynthesis': False})
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = False
    return model


def test_packed_sentences_tick_before_perception_on_one_ladder(tmp_path, monkeypatch):
    import SentenceTraining
    from PrependedSymbols import word_view
    model = native(tmp_path, monkeypatch)
    entries = []
    begin = SentenceTraining.begin_sentence
    def capture(owner):
        sid = max(owner._sentence_ticks)
        tick = owner._sentence_ticks[sid]
        assert tick == owner.present()
        ids = owner.inputSpace._packed_sentence_ids
        selected = (ids == sid) & owner.inputSpace._word_active_mask
        decoded = owner.when_encoding.decode_index(word_view(owner.inputSpace, 'symbol_when'))
        assert (decoded[selected] == tick).all()
        entries.append((sid, tick))
        return begin(owner)
    monkeypatch.setattr(SentenceTraining, 'begin_sentence', capture)
    try:
        source = model.inputSpace.prepPackedInput([['alpha beta', 'beta alpha'], ['beta alpha']])
        with torch.no_grad():
            model.runBatch(train=False, batchSize=2,
                batch_override=(source, torch.empty(2, 0)))
        assert [sid for sid, _ in entries] == [0, 1]
        assert entries[1][1] > entries[0][1] > 0
        for sid, tick in entries:
            field = model._sentence_open_fields[sid]
            assert (field.when[..., 0] == tick).all()
        store = model.symbolSpace.ltm_store
        stamped = store.time[:len(store)]
        for index in (store.episodic_concepts[:len(store)] >= 0).any((-1, -2)).nonzero().flatten():
            assert int(encoder().decode_index(stamped[index])) in dict(entries).values()
    finally:
        model.End()


def test_thought_deposits_take_unique_nonempty_model_ticks_and_updates_keep_them():
    from clock_fixtures import install
    from Memory import ShortTermMemory
    from Meaning import ConceptualMeaning
    from ThoughtChunks import put
    from ThoughtReferences import bindings
    model = install(SimpleNamespace())
    stm = ShortTermMemory(batch=1, capacity=4, concept_dim=8)
    object.__setattr__(stm, '_time_owner', lambda: model)
    value = ConceptualMeaning.from_description(torch.ones(8))
    first = put(stm, value)
    one = bindings(first)['_chunk_when']
    second = put(stm, value)
    two = bindings(second)['_chunk_when']
    assert one != two and any(one) and any(two)
    assert [int(encoder().decode_index(torch.tensor(band))) for band in (one, two)] == [1, 2]
    put(stm, first, when=one, existing=True)
    assert model.present() == 2 and not hasattr(stm, '_thought_clock')




def test_relative_reader_inputs_and_word_order_survive_the_full_clock_horizon():
    from clock_fixtures import install
    from RowTime import age_band, age_interval, subintervals
    model = install(SimpleNamespace())
    reference = None
    for tick in (0, 25, 50, 100, 200, 400, 777, 1500, 3000, HORIZON):
        model.when_time.fill_(tick)
        actual = age_band(model, model.when_encoding.encode(torch.tensor([tick, max(0, tick - 1)])))
        assert torch.equal(actual[0], model.when_encoding.encode(0))
        # Float64 intervals preserve the subdivisions even at the ladder's
        # largest tick, a stronger bound than any permitted default epoch.
        count = 256
        rank = torch.arange(count, dtype=torch.float64)
        intervals = subintervals(torch.full((count,), tick, dtype=torch.long), rank, torch.tensor(count))
        assert torch.equal(((intervals[:, 0] - tick) * count).long(), rank.long())
        relative = age_interval(model, intervals)
        if reference is None:
            reference = relative
        assert torch.equal(relative, reference)


@pytest.mark.parametrize('device', ('cpu', 'mps'))
def test_word_subintervals_preserve_order_on_the_native_device(device):
    from RowTime import subintervals, age_interval
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Metal is unavailable')
    rank = torch.arange(256, device=device)
    intervals = subintervals(torch.full_like(rank, HORIZON), rank, torch.tensor(256, device=device))
    ages = age_interval(SimpleNamespace(when_time=torch.tensor(HORIZON, device=device)), intervals)
    torch.testing.assert_close(ages[:, 1], -torch.arange(256, dtype=torch.float64, device=ages.device)/256,
                               rtol=0, atol=0)


@pytest.mark.learning_certificate
def test_trained_numeric_answers_do_not_move_at_nine_clock_offsets(tmp_path):
    """One ordinary fit; replay its weights without selecting a random draw."""
    from test_mm_xor import TestMMXorConvergence, _CONFIG
    rows = []
    def inspect(model, source, target):
        reference = None
        parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        with torch.no_grad():
            for tick in (0, 25, 50, 100, 200, 400, 777, 1500, 3000):
                model.begin_presentation()
                model.when_time.fill_(tick)
                model.when_encoding.t = tick
                result, _ = model.runBatch(train=False, batchSize=len(source), split='runtime',
                    batch_override=(source, target))
                answers = result.outputPred.detach().clone()
                if reference is None:
                    reference = answers
                torch.testing.assert_close(answers, reference, rtol=0, atol=0)
                rows.append(dict(clock=tick, answers=answers.cpu().tolist(),
                                 maximum_change=float((answers-reference).abs().max())))
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
    TestMMXorConvergence()._fit_xor(_CONFIG, epochs=200, threshold=.20, after_fit=inspect)
    import json
    (tmp_path/'trained-clock-answers.json').write_text(json.dumps(rows, indent=2)+'\n')


def test_same_weights_and_sentences_choose_the_same_derivations_at_nine_clock_offsets(tmp_path, monkeypatch):
    """Replay one ordinary initialization; no random draw is selected to pass."""
    initial, reference = None, None
    rows = []
    for tick in (0, 25, 50, 100, 200, 400, 777, 1500, 3000):
        model = native(tmp_path, monkeypatch)
        try:
            if initial is None:
                initial = copy.deepcopy(model.state_dict())
            else:
                model.load_state_dict(initial)
            model.when_time.fill_(tick)
            model.when_encoding.t = tick
            source = model.inputSpace.prepInput(['alpha beta', 'beta alpha'])
            with torch.no_grad():
                model.runBatch(train=False, batchSize=2,
                    batch_override=(source, torch.empty(2, 0)))
            choices = model._reconstruction_stack()._choice_actions.detach().clone()
            if reference is None:
                reference = choices
            assert torch.equal(choices, reference), (tick, choices, reference)
            rows.append(dict(clock=tick, choices=choices.tolist()))
        finally:
            model.End()
            del model
            import gc
            gc.collect()
    import json
    (tmp_path/'clock-derivations.json').write_text(json.dumps(rows, indent=2)+'\n')



def test_compiled_clock_difference_reuses_one_graph_and_keeps_exact_arithmetic():
    from RowTime import difference
    clock = encoder()
    pointer = clock.t.data_ptr()
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    compiled = torch.compile(lambda band: difference(clock, band),
                             backend=backend, fullgraph=True)
    for tick in (1, 25, 777, 3000):
        clock.t = tick
        band = clock.encode(torch.tensor([tick - 1, tick])).detach().requires_grad_()
        actual = compiled(band)
        # This is the preceding integer-clock calculation, unchanged.
        age = tick - clock.decode_index(band)
        expected = torch.where(band.ne(0).any(-1, keepdim=True),
            clock.encode(age).to(band) - clock.encode(0).to(band), 0.)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert clock.t.data_ptr() == pointer
    assert len(graphs) == 1
    assert '_model_tick' not in clock.state_dict()
