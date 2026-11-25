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
        # MTM nominates a form on recurrence (sigma-pi §21, rule 20).
        from nomination_fixtures import nominated_forms
        nominated_forms(model, ['alpha beta'])
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
    from RowTime import how_far_back, subintervals
    model = install(SimpleNamespace())
    reference = None
    for tick in (0, 25, 50, 100, 200, 400, 777, 1500, 3000, HORIZON):
        model.set_present(tick)
        actual = how_far_back(model.present_code(), model.when_encoding.encode(torch.tensor([tick, max(0, tick - 1)])))
        torch.testing.assert_close(actual[0], model.when_encoding.encode(0).to(actual))
        # Float64 intervals preserve the subdivisions even at the ladder's
        # largest tick, a stronger bound than any permitted default epoch.
        count = 256
        rank = torch.arange(count, dtype=torch.float64)
        intervals = subintervals(torch.full((count,), tick, dtype=torch.long), rank, torch.tensor(count))
        assert torch.equal(((intervals[:, 0] - tick) * count).long(), rank.long())
        relative = how_far_back(model.present_code(), model.when_encoding.encode(intervals).float())
        if reference is None:
            reference = relative
        torch.testing.assert_close(relative, reference, rtol=0, atol=1e-5)


@pytest.mark.parametrize('device', ('cpu', 'mps'))
def test_word_subintervals_preserve_order_on_the_native_device(device):
    from clock_fixtures import install
    from RowTime import subintervals, how_far_back
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Metal is unavailable')
    rank = torch.arange(256, device=device)
    intervals = subintervals(torch.full_like(rank, HORIZON), rank, torch.tensor(256, device=device))
    model = install(SimpleNamespace(), HORIZON)
    # Each word's end lies a distinct distance back: no two collapse.
    ends = how_far_back(model.present_code(), encoder().encode(intervals[:, 1]).float())
    assert torch.unique(ends, dim=0).shape[0] == 256


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
                model.set_present(tick)
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
            model.set_present(tick)
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



def test_compiled_comparator_reuses_one_graph_and_keeps_exact_arithmetic():
    from RowTime import how_far_back
    clock = encoder()
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    compiled = torch.compile(how_far_back, backend=backend, fullgraph=True)
    for tick in (1, 25, 777, 3000):
        present = clock.encode(tick).float()
        band = clock.encode(torch.tensor([tick - 1, tick])).detach().requires_grad_()
        actual = compiled(present, band)
        torch.testing.assert_close(actual, clock.encode(torch.tensor([1, 0])).to(actual), rtol=0, atol=1e-6)
        assert clock.decode_index(actual).tolist() == [1, 0]
    assert len(graphs) == 1
    assert not clock.state_dict()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires Metal')
def test_native_clock_rotation_and_stamp_gradient_match_double_precision():
    from RowTime import how_far_back
    from torch._dynamo.testing import CompileCounterWithBackend

    generator = torch.Generator(device='cpu').manual_seed(1505)
    latest = torch.randint(0, HORIZON, (96,), generator=generator, device='cpu')
    back = (torch.rand(96, generator=generator, device='cpu') * latest).long()
    now = encoder().encode(latest).float().cpu()
    past = encoder().encode(latest - back).float().cpu()
    stamps = torch.stack((past, torch.rand(past.shape, generator=generator, device='cpu')), 1)
    stamps[:3, 0] = torch.tensor(((0., 0., 0., 0.), (.5, .5, .6, .1), (.5, .5, .5, .5)), device='cpu')
    grad = torch.randn(stamps.shape, generator=generator, device='cpu')
    counter = CompileCounterWithBackend('inductor')
    compiled = torch.compile(how_far_back, backend=counter, fullgraph=True)
    for source, weights in ((stamps, grad), (stamps.flip(0), grad.flip(0))):
        cpu = source.detach().requires_grad_()
        native = source.detach().to('mps').requires_grad_()
        expected = how_far_back(now, cpu)
        clock = now.to('mps')
        actual = compiled(clock, native)
        # Both source and clock can be reused by a later look before backward.
        with torch.no_grad():
            native.zero_()
            clock.zero_()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        reference, = torch.autograd.grad(expected, cpu, weights)
        gradient, = torch.autograd.grad(actual, native, weights.to('mps'))
        torch.testing.assert_close(gradient.cpu(), reference, rtol=0, atol=0, equal_nan=True)
    assert counter.frame_count == 1
