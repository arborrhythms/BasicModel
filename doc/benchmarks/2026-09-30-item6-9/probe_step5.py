"""An identical second derivation cannot win an equal-parameter comparison."""
import torch


def test_identical_second_trial_never_wins_and_both_use_original_parameters():
    from SentenceCompose import saved_sentence_values, sentence_pair
    parameter = torch.nn.Parameter(torch.tensor(2.))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    observations, gradients = [], []
    active = torch.tensor([True, True, False])
    def compose(cache, prior):
        observations.append(float(parameter.detach()))
        return parameter * cache
    def score(path, alternative):
        return path.square(), path
    def step(loss):
        optimizer.zero_grad()
        loss.backward()
        gradients.append(float(parameter.grad))
        optimizer.step()
    with saved_sentence_values((parameter,)):
        chosen, costs, wins = sentence_pair(torch.ones(3), compose, score, step, active=active)
    assert observations == [2., 2.], observations
    torch.testing.assert_close(costs[:, 0], costs[:, 1], rtol=0, atol=0)
    assert not wins.any(), wins
    assert gradients == [4., 4.], gradients
    torch.testing.assert_close(parameter, torch.tensor(1.2))
    torch.testing.assert_close(chosen, torch.full((3,), 2.))


def test_sentence_trials_keep_their_own_perception_pullbacks(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model
    import SentenceCompose
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8',
        training_overrides={'reconstructInLoop': False})
    model._tensor_peer_while_eager = True
    optimizer = model.getOptimizer(lr=1e-4)
    created, called, versions = [], [], []
    original = SentenceCompose.fork_perception
    def observed(cache):
        trial, pullback = original(cache)
        index = len(created)
        created.append(index)
        versions.append(tuple(parameter._version for parameter in model.parameters()))
        def owned():
            called.append(index)
            pullback()
        owned.gradients = pullback.gradients
        return trial, owned
    monkeypatch.setattr(SentenceCompose, 'fork_perception', observed)
    try:
        raw = model.inputSpace.prepInput(['a b', 'c d'])
        model.runBatch(train=True, optimizer=optimizer, batchSize=2, split='runtime',
            batch_override=(raw, torch.empty(2, 0)))
        assert created == called == [0, 1], (created, called)
        assert versions[0] == versions[1], 'an optimizer update separated the sentence trials'
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_replaying_real_greedy_derivation_ties_in_every_row(monkeypatch):
    from pathlib import Path
    import Models
    import SentenceCompose
    import util
    from test_mm_xor import _fresh_model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _, data = _fresh_model(str(Path(Models.__file__).resolve().parents[1] / 'data/XOR_grammar.xml'))
    optimizer = model.getOptimizer(lr=.01)
    pair = SentenceCompose.sentence_pair
    comparisons = []
    def repeat(cache, compose, score, step, **kwargs):
        # This diagnostic deliberately replays greedy. The production
        # exploration-integrity assertion is about different actions; the
        # ordinary score's reconstruction/expectation arithmetic is identical.
        result = pair(cache, lambda value, prior: compose(value, None),
                      lambda path, alternative: score(path, False), step, **kwargs)
        comparisons.append((result[1], result[2]))
        return result
    monkeypatch.setattr(SentenceCompose, 'sentence_pair', repeat)
    try:
        batch = next(iter(data.data_loader(split='train', num_streams=4)))
        raw = model.inputSpace.prepInput(batch[0])
        target = model.outputSpace.prepOutput(batch[1])
        model.runBatch(train=True, optimizer=optimizer, batchSize=4,
                       split='runtime', batch_override=(raw, target))
        assert comparisons
        for costs, wins in comparisons:
            torch.testing.assert_close(costs[:, 0], costs[:, 1], rtol=0, atol=0)
            assert not wins.any(), wins
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
