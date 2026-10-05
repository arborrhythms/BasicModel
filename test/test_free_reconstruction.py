"""Witness-free reconstruction competes over live, unnormalized concept codes."""
import torch


def test_readback_uses_signed_activation_cosine_and_priming():
    from SentenceUnderstanding import readback_scores
    leaf = torch.tensor([[2., 0.]])
    codes = torch.tensor([[[2., 0.], [0., 4.], [-1., 1.]]])
    weights = torch.tensor([[3., 9., 2.]])
    # The least-squares signed activation is 1, 0, -1 respectively.
    expected = torch.tensor([[3., 0., 2.**.5]])
    torch.testing.assert_close(readback_scores(leaf, codes, weights), expected)
    torch.testing.assert_close(readback_scores(-leaf, codes, weights), expected)


def test_readback_error_reaches_true_and_competing_codes_without_unit_constraint():
    from SentenceUnderstanding import readback_scores
    codes = torch.nn.Parameter(torch.tensor([[[2., .3], [.4, 3.]]]))
    leaf = torch.tensor([[1., .2]], requires_grad=True)
    scores = readback_scores(leaf, codes, torch.ones(1, 2))
    torch.nn.functional.cross_entropy(scores, torch.tensor([0])).backward()
    assert codes.grad[0, 0].norm() > 0
    assert codes.grad[0, 1].norm() > 0
    assert leaf.grad.norm() > 0
    torch.testing.assert_close(codes.detach(), torch.tensor([[[2., .3], [.4, 3.]]]))


def test_free_trial_uses_no_reference_or_offsets_and_only_byte_cost(monkeypatch):
    from pathlib import Path
    import Models, util
    from test_mm_xor import _fresh_model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _, data = _fresh_model(str(Path(Models.__file__).resolve().parents[1]/'data/XOR_grammar.xml'))
    original = model._reconstruct_sentences
    calls=[]
    def observe(*args, **kw):
        record = kw['understanding']
        assert not record.primed.codes.requires_grad
        from dataclasses import replace, fields
        assert 'witness_offsets' not in {f.name for f in fields(record)}
        # The original leaves are targets, never inverse operands. Changing
        # them cannot change the recovered leaves or the byte objective.
        returned = original(*args, **kw)
        result = returned[0] if kw.get('return_decoded') else returned
        changed = replace(record, word_values=record.word_values + 19)
        altered = original(args[0], changed.word_values, *args[2:],
                           **(kw | {'understanding': changed}))
        if kw.get('return_decoded'):
            altered = altered[0]
        torch.testing.assert_close(result[0], altered[0])
        torch.testing.assert_close(result[2], altered[2])
        calls.append('explore' if kw.get('decoder_exploit') is not None else 'greedy')
        return returned
    monkeypatch.setattr(model, '_reconstruct_sentences', observe)
    optimizer=model.getOptimizer(lr=.01)
    try:
        raw,target=next(iter(data.data_loader(split='train',num_streams=4)))
        model.runBatch(train=True,optimizer=optimizer,batchSize=4,split='train',
            batch_override=(model.inputSpace.prepInput(raw),model.outputSpace.prepOutput(target)))
        assert calls == ['greedy', 'explore', 'greedy', 'explore']
        assert {name for name in model._sentence_cost_registry._terms if name.startswith('reconstruction.')} == {'reconstruction.free_bytes', 'reconstruction.decomposition'}
        audit = model.ownership_gradient_diagnostics(optimizer)
        assert audit['conflicts'] == 0
        decoder = [row for row in audit['parameters'] if 'generate_policy' in row['parameter']]
        assert decoder and all(row['owner'] == 'reconstruction' and row['writers'] == ['reconstruction'] for row in decoder)
    finally:
        model.End()
