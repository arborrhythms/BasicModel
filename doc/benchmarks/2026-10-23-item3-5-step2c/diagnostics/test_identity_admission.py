"""Identity admission and Language consume the same resolved reading."""
import torch
from conftest import eager_reading


def test_identity_admission_uses_the_resolved_stm_order(tmp_path, monkeypatch, eager_reading):
    from Models import FunctionalPeerSTM
    from MTMOverlay import MTMOverlay
    from test_compiled_word_chunk import _tiny_canonical_model
    from What import What
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8',
        concept_rows=128, input_width=16, batch_size=1, chooser_depth=1,
        training_overrides={'reconstructionPlacement': 'eager'},
        architecture_overrides={'answerSynthesis': False})
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    resolve, push = FunctionalPeerSTM.resolve_top_reference, MTMOverlay.push
    expected, observed = [], []
    def resolved(*args, **kwargs):
        result = resolve(*args, **kwargs)
        expected.append(result[2][:, 0].clone())
        return result
    def opening(state, active, value, **kwargs):
        if active.any():
            torch.testing.assert_close(kwargs['order'][active], expected[-1][active], rtol=0, atol=0)
            observed.append(True)
        return push(state, active, value, **kwargs)
    monkeypatch.setattr(FunctionalPeerSTM, 'resolve_top_reference', staticmethod(resolved))
    monkeypatch.setattr(MTMOverlay, 'push', staticmethod(opening))
    try:
        raw = model.inputSpace.prepInput(['a b'])
        model.runBatch(train=False, batchSize=1, split='validation',
            batch_override=(raw, torch.empty(1, 0)), questions=(What.present(0, split='validation'),))
        assert observed
    finally:
        model.End()
