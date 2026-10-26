"""Regressions for the rejected shared-chunk source's review findings."""
import json

import pytest
import torch


@pytest.mark.slow
def test_recorded_two_stream_readings(tmp_path, monkeypatch, eager_reading):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    import util
    from Models import BasicModel
    from test_math_chain_decomposition import test_forced_native_decomposition_through_training_driver
    root = Path(__file__).resolve().parent
    fixture = root/'fixtures/2026-10-25-stm-repair'
    # Reuse the original zero-arm length-1 seed and immutable parameter
    # snapshot. The adapter retains every surviving parameter/column.
    monkeypatch.setenv('IDENTITY_GATE_DIR', str(fixture))
    monkeypatch.setenv('LAYOUT_GATE_DIR', str(tmp_path/'initialization/rng'))
    monkeypatch.setenv('BIND_GAIN_ARM', 'zero')
    monkeypatch.setattr(util.XMLConfig, '_parse_xml', staticmethod(util.XMLConfig._parse_xml))
    monkeypatch.setattr(BasicModel, '_create_per_stage', BasicModel._create_per_stage)
    spec = importlib.util.spec_from_file_location('recorded_chunk_initialization', fixture/'recorded_initialization.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.pytest_runtest_setup(SimpleNamespace(nodeid=
        'test/test_math_chain_decomposition.py::test_forced_native_decomposition_through_training_driver[1]'))
    # The inherited proof still asserts its original existence criteria.
    test_forced_native_decomposition_through_training_driver(tmp_path, 1, eager_reading)
    progress = json.loads((tmp_path/'decomposition-1/progress.json').read_text())
    for stream in (0,1):
        for sentence in (0,1,2):
            refs = progress['source_readings'][str((stream,sentence))]
            assert all(ref is not None for ref in refs), (stream,sentence,refs)
            assert refs[0][0]=='ltm' and refs[1][0]=='sym', (stream,sentence,refs)


def test_same_chunks_supply_current_time_and_untimed_estimate(tmp_path):
    from Layers import SymbolExpectation, LTM0
    from RowTime import encoder
    import STMChunks as chunks
    width = 16
    predictor = SymbolExpectation(n_symbols=4, max_depth=8, n_dim=width,
                                  concept_dim=width, batch=1)
    store = LTM0(width, capacity=8, percept_width=12)
    predictor._ltm_store = store
    predictor.set_symbol_bank(torch.eye(width)[:4])
    state = chunks.empty(1, 8, width)
    value = torch.arange(1, width+1).float().div(width)
    value[8:12] = encoder().encode(2)
    state, _ = chunks.deposit(state, value[None], torch.tensor([True]))
    before = tuple(v.clone() for v in state)
    roles = torch.cat((state[chunks.BUFFER][0,:1], torch.zeros(2,width)))
    predictor._inter_context[0].append((1, roles, torch.tensor([True,False,False])))
    predictor._inter_context_lanes[0].append((1.,0.))
    predictor._inter_context_ids[0].append(torch.tensor([0,-1,-1]))
    predictor._inter_context_occurrences[0].append(None)
    contexts, estimates = [], []
    handle = predictor._inter_predictor.context.register_forward_pre_hook(
        lambda module, args: contexts.append(args[0].detach().clone()))
    try:
        for tick in (7, 19):
            store.model_clock.fill_(tick)
            estimate = predictor.expect_next_meaning()
            estimates.append(estimate.roles.clone())
            assert int(store.model_clock) == tick
            assert torch.count_nonzero(estimate.roles[0,8:12]) == 0
    finally:
        handle.remove()
    assert len(contexts)==2 and contexts[0].shape==(1,4*width+4)
    torch.testing.assert_close(contexts[0][:,:4*width],contexts[1][:,:4*width],rtol=0,atol=0)
    for context,tick in zip(contexts,(7,19)):
        torch.testing.assert_close(context[0,4*width:],encoder().encode(tick),rtol=0,atol=0)
    assert not torch.equal(contexts[0],contexts[1])
    assert all(torch.equal(old,new) for old,new in zip(before,state))
    # Persistence happens at a later close. The forecast consumes no tick;
    # the source and arriving observation keep their exact absolute times.
    from Meaning import ConceptualMeaning
    from meaning_fixtures import estimate_meaning
    from RowTime import read_stamp
    source = store.append_meaning(ConceptualMeaning.from_description(value),
                                  when=encoder().encode(2), stream=0)
    early = store.append_estimate(estimate_meaning(estimate),
        presence_logits=estimate.presence_logits,
        source_occurrences=(store.occurrence_of(source),), document='early')
    cursor_after_estimate = int(store.model_clock)
    assert cursor_after_estimate == 19
    assert torch.count_nonzero(read_stamp(store,early)) == 0
    store.model_clock.fill_(31)
    predicted, observed = store.append_expectation_pair(estimate_meaning(estimate),
        ConceptualMeaning.from_description(value*.5),
        presence_logits=estimate.presence_logits,
        source_occurrences=(store.occurrence_of(source),),
        stream=('external', 'clock'), document='clock', index_stream=0)
    assert torch.count_nonzero(read_stamp(store,predicted)) == 0
    torch.testing.assert_close(read_stamp(store,source),encoder().encode(2),rtol=0,atol=0)
    torch.testing.assert_close(read_stamp(store,observed),encoder().encode(31),rtol=0,atol=0)
    assert int(store.model_clock) == 32
    (tmp_path/'certificate.json').write_text(json.dumps(dict(ticks=[7,19],
        contexts=[v.tolist() for v in contexts], estimates=[v.tolist() for v in estimates],
        fixed_now_start=4*width, chunks_unchanged=True, cursor_after_estimate=cursor_after_estimate,
        stored_stamps={name:read_stamp(store,row).tolist() for name,row in
                       (('source',source),('estimate',predicted),('observation',observed))}),indent=2)+'\n')
