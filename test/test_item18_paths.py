"""Real phase boundaries for field and examination configurations."""
from pathlib import Path
import pytest
import torch


@pytest.mark.parametrize('train', (False, True))
def test_examination_has_no_vector_walk_and_owns_its_reverse(tmp_path, monkeypatch, train):
    from test_compiled_word_chunk import _tiny_canonical_model
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1,
        grammar=str(Path('data/examination.grammar').resolve()))
    model._tensor_peer_while_eager = True
    def forbidden(*args, **kwargs):
        raise AssertionError('an examination path entered a vector walk')
    monkeypatch.setattr(model, 'generate', forbidden)
    monkeypatch.setattr(model.languageSpace.language_layer.operation_layer, 'forward', forbidden)
    monkeypatch.setattr(model.inputSpace.data, 'has_supervised_outputs', False)
    source = model.inputSpace.prepInput(['one two'])
    try:
        model.runBatch(train=train, optimizer=model.getOptimizer(lr=.003) if train else None,
            batchSize=1,batch_override=(source,torch.empty(1,0)))
        audit = model._last_examination
        assert len(audit['readings']) == 1 + (model.explore_paths if train else 0)
        assert all(departure.ndim == 1 for departure in audit['departures'])
        assert model._input_reconstructions
        assert len(model.symbolSpace.ltm_store) > 0
    finally:
        model.End()


def test_numeric_understanding_owns_evidence_across_next_forward():
    from recon_bench import _build_model
    model, *_ = _build_model('data/MM_xor.xml')
    values = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]).unsqueeze(-1)
    first = model.understand(values)
    model.forward(values * 2.)
    restored, _ = model.reverseReconstruct(first)
    torch.testing.assert_close(restored, values)
