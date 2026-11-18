"""Every serial binding deposits word codes before comprehension."""
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize('configuration', ['XOR_grammar.xml', 'MM_grammar.xml', 'aligned'])
def test_grammar_deposits_each_word_with_its_own_code(tmp_path, monkeypatch, configuration):
    import util
    from Memory import ShortTermMemory
    from PrependedSymbols import word_view
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    if configuration == 'aligned':
        from test_compiled_word_chunk import _tiny_canonical_model
        model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', batch_size=4)
        model.reconstruct_in_loop = False
        model._tensor_peer_while_eager = True
        model._chart_compose_per_word = lambda: None
    else:
        from test_mm_xor import _fresh_model
        import Models
        model, _, _ = _fresh_model(str(Path(Models.__file__).resolve().parents[1] / 'data' / configuration))
        if configuration == 'MM_grammar.xml':
            # A sign/Part can define an unknown label after the read.
            # Keep an absolute composition for this lexical read-back.
            from reading_fixtures import force_absolute_reading
            force_absolute_reading(model)
    model.eval()
    seen = []
    push = ShortTermMemory.functional_push_step_masked
    def observed(state, idea, gate, order, grammar_order, row, activation):
        import inspect
        caller = inspect.currentframe().f_back
        if caller.f_code.co_name == 'stage_cs_lang' and bool(gate.any()):
            local = caller.f_locals
            index = int(local['index'])
            own_rows = word_view(model.inputSpace, 'concept_rows')[:, index]
            own_orders = word_view(model.inputSpace, 'concept_orders')[:, index]
            atoms = model.inputSpace._ar_word_concept_atoms[:, index]
            expected = model._concept_owner().interpret.activate(atoms, activation.reshape(-1),
                evidence=local['leaf_evidence'])
            start = model.symbolSpace.ltm_store.when_start
            if start is not None:
                expected[:, start-4:start+4] = local['symbolic_event'][:, 0, start-4:start+4]
            live = gate.reshape(-1)
            torch.testing.assert_close(idea[live], expected[live], rtol=0, atol=0)
            torch.testing.assert_close(row.reshape(-1)[live], own_rows[live], rtol=0, atol=0)
            torch.testing.assert_close(order.reshape(-1)[live], own_orders[live], rtol=0, atol=0)
            seen.extend(own_rows[live].tolist())
        return push(state, idea, gate, order, grammar_order, row, activation)
    monkeypatch.setattr(ShortTermMemory, 'functional_push_step_masked', staticmethod(observed))
    try:
        inputs = model.inputSpace.prepInput(['hello world', 'hello there', 'loving world', 'loving there'])
        with torch.no_grad():
            model(inputs)
        assert len(seen) == 8, f'every real word must be deposited once; observed {len(seen)} word leaves'
        owner = model._concept_owner()
        expected_rows = {owner._csw_row_of(owner.interpret.word_for_form(word))
                         for word in ('hello', 'loving', 'world', 'there')}
        assert len(expected_rows) == 4
        assert set(seen) == expected_rows
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
