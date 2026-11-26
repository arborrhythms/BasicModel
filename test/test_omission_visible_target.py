"""Omission's hidden-word target is the word's visible reading."""
import torch


def _model(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32,
        concept_rows=32, training_overrides={'reconstructionPlacement': 'eager'},
        architecture_overrides={'explorePaths': 0})
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = False
    return model


def _present(model, monkeypatch, text, hide, *, train=False):
    from Omission import Draw
    def draw(valid, *, training):
        forms = torch.zeros_like(valid)
        if hide is not None:
            forms[:, hide] = valid[:, hide]
        return Draw(forms.any(-1), forms, torch.zeros(len(valid), dtype=torch.long))
    monkeypatch.setattr(model.omission, 'draw', draw)
    # A measuring presentation leaves the forms and their codes unchanged.
    for space in (model.inputSpace, model.perceptualSpace, model.conceptualSpace, model.wholeSpace):
        object.__setattr__(space, '_online_learning_frozen', not train)
    source = model.inputSpace.prepInput([text])
    model.runBatch(train=train, optimizer=model.getOptimizer(lr=.003) if train else None, batchSize=1,
                   batch_override=(source, torch.empty(1, 0)))
    return model.inputSpace._ar_embedded_N[0].detach().clone(), model._omission_targets


def test_hidden_target_is_the_visible_reading(tmp_path, monkeypatch):
    model = _model(tmp_path, monkeypatch)
    what = model.perceptualSpace.nWhat
    try:
        # Not yet a promoted form: read by its existing parts (its bytes).
        visible, _ = _present(model, monkeypatch, 'alpha gamma', None)
        tiles = int(model.inputSpace._ar_word_part_mask[0, 1].sum())
        assert tiles == len(b'gamma')
        hidden, targets = _present(model, monkeypatch, 'alpha gamma', 1)
        assert not torch.equal(hidden[1], visible[1])
        # The content is the visible reading; the place and time are this presentation's.
        torch.testing.assert_close(targets['word_values'][0, 1, :what], visible[1, :what], rtol=0, atol=0)
        torch.testing.assert_close(targets['word_values'][0, 1, what:], hidden[1, what:], rtol=0, atol=0)
        # A promoted form: the target is the one form the visible word reads.
        for _ in range(4):
            _present(model, monkeypatch, 'alpha gamma', None, train=True)
        visible, _ = _present(model, monkeypatch, 'alpha gamma', None)
        assert int(model.inputSpace._ar_word_part_mask[0, 1].sum()) == 1
        _, targets = _present(model, monkeypatch, 'alpha gamma', 1)
        torch.testing.assert_close(targets['word_values'][0, 1, :what], visible[1, :what], rtol=0, atol=0)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
