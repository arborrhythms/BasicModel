"""An identical second derivation cannot win an equal-parameter comparison."""
import torch




def test_whole_paths_share_the_perception_graph_until_the_one_owned_update(tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    import WholePaths, ObjectiveOwnership, util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8',
        architecture_overrides={'separateChooser': True})
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = False
    versions, calls = [], []
    finish, backward = WholePaths.finish, ObjectiveOwnership.backward_owned
    def observed(*args, **kwargs):
        versions.append(tuple(p._version for p in model.parameters()))
        return finish(*args, **kwargs)
    def owned(costs, owners, **kwargs):
        assert kwargs.get('pullback') is None
        calls.append(tuple(costs))
        return backward(costs, owners, **kwargs)
    monkeypatch.setattr(WholePaths, 'finish', observed)
    monkeypatch.setattr(ObjectiveOwnership, 'backward_owned', owned)
    try:
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=1e-4), batchSize=2, split='runtime',
            batch_override=(model.inputSpace.prepInput(['a b','c d']), torch.empty(2,0)))
        assert len(versions) == 2 and versions[0] == versions[1]
        assert len(calls) == 1 and 'reconstruction' in calls[0]
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_no_departure_reuses_the_real_greedy_path_and_ties(monkeypatch):
    import SentenceCredit, util
    from test_mm_xor import _fresh_model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _, data = _fresh_model('data/XOR_grammar.xml')
    native = SentenceCredit.departure
    def none(*args, **kwargs):
        result = native(*args, **kwargs)
        for name, value in result.items():
            if name == 'walk' or name.endswith('_round'): result[name] = torch.full_like(value,-1)
        return result
    monkeypatch.setattr(SentenceCredit, 'departure', none)
    try:
        raw, target = next(iter(data.data_loader(split='train',num_streams=4)))
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.01), batchSize=4,
            batch_override=(model.inputSpace.prepInput(raw),model.outputSpace.prepOutput(target)))
        audit = model._last_sentence_credit
        torch.testing.assert_close(audit['costs'][:,0], audit['costs'][:,1], rtol=0,atol=0)
        assert not audit['winner'].any()
    finally:
        model.End(); model.symbolSpace.soft_reset()
