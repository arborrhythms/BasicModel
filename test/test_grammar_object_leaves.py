"""Every serial binding presents full object codes at the grammar boundary."""
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize('configuration', ['XOR_grammar.xml', 'MM_grammar.xml', 'aligned'])
def test_grammar_resolves_each_word_to_its_full_object_code(tmp_path, monkeypatch, configuration):
    import util
    from Models import FunctionalPeerSTM
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    if configuration == 'aligned':
        from test_compiled_word_chunk import _tiny_canonical_model
        model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8')
        model.reconstruct_in_loop = False
        model._tensor_peer_while_eager = True
        model._chart_compose_per_word = lambda: None
    else:
        from test_mm_xor import _fresh_model
        import Models
        model, _, _ = _fresh_model(str(Path(Models.__file__).resolve().parents[1] / 'data' / configuration))
    model.eval()
    seen = []
    resolve = FunctionalPeerSTM.resolve_top_reference
    def observed(state, idea, row, order, activation, gate):
        if bool(gate.any()):
            owner = model._concept_owner()
            atoms = owner.similarity_codebook.lookup_rows(row[gate])
            derived = getattr(owner.similarity_codebook, 'mereology', None)
            width = atoms.shape[-1] if derived is None else derived.percept_event_width
            expected = torch.cat((atoms[..., :width] * activation[gate, None].abs(),
                                  atoms[..., width:] * activation[gate, None]), -1)
            torch.testing.assert_close(idea[gate], expected, rtol=0, atol=0)
            seen.extend(row[gate].tolist())
        return resolve(state, idea, row, order, activation, gate)
    monkeypatch.setattr(FunctionalPeerSTM, 'resolve_top_reference', staticmethod(observed))
    try:
        inputs = model.inputSpace.prepInput(['hello world', 'hello there', 'loving world', 'loving there'])
        with torch.no_grad():
            model(inputs)
        assert len(seen) == 8, f'every real word must resolve once; observed {len(seen)} object leaves'
        owner = model._concept_owner()
        expected_rows = {owner._csw_row_of(owner.definitions.deref(owner.definitions.word(form=word)))
                         for word in ('hello', 'loving', 'world', 'there')}
        assert set(seen) == expected_rows
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
