"""Preserve the evaluator's accidental autograd boundary before repairing it."""
import torch
import eval_nanochat_grammar as gate


def test_frozen_control_scoring_disables_autograd(monkeypatch):
    from test_mm_xor import _fresh_model
    model, _, data = _fresh_model('data/XOR_grammar.xml')
    model.eval()
    owner = model.symbolSpace.expectation
    expect = owner.expect
    calls = []
    def inference(*args, **kwargs):
        calls.append(torch.is_grad_enabled())
        assert not torch.is_grad_enabled(), 'held-out candidate scoring retained a backward graph'
        return expect(*args, **kwargs)
    monkeypatch.setattr(owner, 'expect', inference)
    item = dict(prefix='hello ', candidates=['world', 'there'])
    with gate.frozen_online_learning(model):
        gate._score_control_batch(model, data, [item], 'intact', 2)
    assert calls == [False, False]
