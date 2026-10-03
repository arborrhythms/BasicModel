"""Supplied answers train each sentence trial; absent answers retain the cut."""
from types import MethodType, SimpleNamespace

import pytest
import torch


def _sentence(supplied):
    from Models import BasicModel
    codes = torch.nn.Parameter(torch.tensor([[1., 2.], [2., 3.]]))
    chooser = torch.nn.Parameter(torch.tensor(2.))
    root = chooser * codes
    class Head:
        concept_ids = ()
        inputShape = (3, 2)
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.ones(6))
        def __call__(self, sub):
            value = sub.materialize().flatten(1) @ self.weight
            return SimpleNamespace(materialize=lambda: value[:, None])
    head = Head()
    slots = torch.cat((root[:, None], root.new_zeros(2, 2, 2)), 1)
    lang = [None] * 15
    lang[9], lang[13], lang[14] = root[:, None], slots.flatten(1)[:, None], torch.ones(2, 1, dtype=torch.long)
    model = SimpleNamespace(serial=True, outputSpace=head,
        normalizer=SimpleNamespace(denormalize=lambda value, **kw: value),
        inputSpace=SimpleNamespace(data=SimpleNamespace(has_supervised_outputs=supplied)),
        _sentence_supplied_answers=torch.zeros(2, 1) if supplied else None,
        _sentence_training=True, _align_output_pred=lambda pred, target: pred,
        _publish_sentence_scratch=lambda state: None,
        _tensor_pushed_ideas=root[:, None], reconstruct_in_loop=False,
        _sentence_observation=lambda *a: {}, _reading_lesson_enabled=False,
        grammar_lesson_weight=0., symbolSpace=SimpleNamespace())
    model._forward_head = MethodType(BasicModel._forward_head, model)
    return model, ((), lang, ()), (codes, chooser, head.weight)


def test_supplied_answer_is_in_trial_cost_with_code_chooser_and_readout_gradient():
    from Models import BasicModel
    model, state, parameters = _sentence(True)
    cost, *_ = BasicModel._sentence_path_cost(model, state, 0, torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(cost, torch.tensor([36., 100.]))
    gradients = torch.autograd.grad(cost.sum(), parameters, allow_unused=True)
    assert all(g is not None and bool(g.abs().sum() > 0) for g in gradients)


def test_sentence_without_answer_keeps_code_and_chooser_out_of_answer_gradient():
    from Models import BasicModel
    model, state, parameters = _sentence(False)
    cost, *_ = BasicModel._sentence_path_cost(model, state, 0, torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(cost, torch.zeros(2))
    root, slots, depth = state[1][9][:, 0], state[1][13][:, 0].reshape(2, 3, 2), state[1][14][:, 0]
    answer = model._forward_head(None, sentence_state=(root, slots, depth)).materialize()
    gradients = torch.autograd.grad(answer.square().sum(), parameters, allow_unused=True)
    assert gradients[0] is None and gradients[1] is None
    assert gradients[2] is not None and bool(gradients[2].abs().sum() > 0)


def test_real_supplied_trials_cost_before_updates_and_reach_all_three_owners(monkeypatch):
    from pathlib import Path
    import Models
    import SentenceCompose
    import util
    from test_mm_xor import _fresh_model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _, data = _fresh_model(str(Path(Models.__file__).resolve().parents[1] / 'data/XOR_grammar.xml'))
    optimizer = model.getOptimizer(lr=.01)
    score = model._sentence_path_cost
    measured, versions = [], []
    def observed(state, sid, active):
        value = score(state, sid, active)
        error = getattr(model, '_sentence_answer_cost', None)
        assert torch.is_tensor(error) and error.requires_grad
        params = tuple(model.parameters())
        grads = model._sentence_pullback.gradients(error.mean(), params)
        names = {id(p): name for name, p in model.named_parameters()}
        live = [names[id(p)] for p, g in zip(params, grads) if g is not None and bool(g.abs().any())]
        assert any(name.startswith('conceptualSpaces.') and name.endswith('.W') for name in live), live
        assert any('operation_layer.' in name for name in live), live
        readout = {id(p) for p in model.outputSpace.parameters()}
        assert any(id(p) in readout and g is not None and bool(g.abs().any())
                   for p, g in zip(params, grads)), live
        measured.append(live)
        versions.append(tuple(p._version for p in params))
        return value
    monkeypatch.setattr(model, '_sentence_path_cost', observed)
    try:
        batch = next(iter(data.data_loader(split='train', num_streams=4)))
        raw, target = model.inputSpace.prepInput(batch[0]), model.outputSpace.prepOutput(batch[1])
        model.runBatch(train=True, optimizer=optimizer, batchSize=4,
            split='train', batch_override=(raw, target))
        assert len(measured) == 2
        assert versions[0] == versions[1]
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
