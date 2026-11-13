"""One supplied-answer reader trains the exposed whole paths."""
from types import MethodType, SimpleNamespace

import pytest
import torch


def _sentence(supplied):
    from Models import BasicModel
    from Layers import ModelLoss
    codes = torch.nn.Parameter(torch.tensor([[1., 2.], [2., 3.]]))
    chooser = torch.nn.Parameter(torch.tensor(2.))
    root = chooser * codes
    class Head:
        field_readout = True
        inputShape = (3, 2)
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.ones(6))
            self.subspace = SimpleNamespace(set_event=lambda value: setattr(self, 'value', value),
                materialize=lambda: self.value)
        def read_field(self, field, owner):
            self.seen = field
            value = field.evidence.amax(2).permute(1, 0, 2).flatten(1)
            return (value @ self.weight)[:, None]
    head = Head()
    slots = torch.cat((root[:, None], root.new_zeros(2, 2, 2)), 1)
    lang = [None] * 12
    lang[6], lang[10], lang[11] = root[:, None], slots.flatten(1)[:, None], torch.ones(2, 1, dtype=torch.long)
    model = SimpleNamespace(word_brackets=True, outputSpace=head,
        loss=ModelLoss(reconstruction_scale=1., what_scale=1., where_scale=1., when_scale=1.),
        normalizer=SimpleNamespace(denormalize=lambda value, **kw: value),
        inputSpace=SimpleNamespace(data=SimpleNamespace(has_supervised_outputs=supplied)),
        _sentence_supplied_answers=torch.zeros(2, 1) if supplied else None,
        _sentence_training=True, _sentence_answer_questions=(),
        _align_output_pred=lambda pred, target: pred,
        _publish_sentence_scratch=lambda state: None,
        _tensor_pushed_ideas=root[:, None], reconstruct_in_loop=False,
        _sentence_observation=lambda *a: {}, _reading_lesson_enabled=False,
        grammar_lesson_weight=0., symbolSpace=SimpleNamespace())
    from Understanding import ConceptualField
    model._numeric_field = ConceptualField(torch.arange(3), slots.permute(1, 0, 2)[:, :, None], None, None, None)
    model._concept_owner = lambda: None
    model._forward_head = MethodType(BasicModel._forward_head, model)
    return model, ((), lang, ()), (codes, chooser, head.weight)




def test_answer_cost_does_not_include_a_generate_lesson():
    from Models import BasicModel
    from Layers import Error
    model, state, _ = _sentence(True)
    active = torch.ones(2, dtype=torch.bool)
    registry = Error(row_mask=active)
    registry.add('grammar.generate', torch.tensor([7., 9.]), objective='output')
    answer = BasicModel._sentence_numeric_error(model, state, 0, active, {"record": None}, registry=registry)
    torch.testing.assert_close(answer, torch.tensor([36., 100.]))
    torch.testing.assert_close(registry.total(), torch.tensor([43., 109.]))




def test_real_supplied_paths_cost_before_updates_and_reach_understanding(monkeypatch):
    from test_mm_xor import _fresh_model
    import WholePaths
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _, data = _fresh_model('data/XOR_grammar.xml')
    # The new linear head starts at zero. A nonzero reader exposes its
    # cotangent into the field independently of a preceding optimizer step.
    with torch.no_grad(): model.outputSpace.field_readout.weight.fill_(.1)
    score = WholePaths.output
    measured, versions = [], []
    def observed(owner, state, observation, sid, active, registry, **kwargs):
        value = score(owner, state, observation, sid, active, registry, **kwargs)
        error = value[0]
        params = tuple(model.parameters())
        grads = torch.autograd.grad(error.mean(), params, retain_graph=True, allow_unused=True)
        live = {id(p) for p, g in zip(params, grads) if g is not None and bool(g.abs().any())}
        assert any(id(p) in live for p in model.outputSpace.parameters())
        field = observation['answer_field'].evidence
        gradient, = torch.autograd.grad(error.mean(), field, retain_graph=True)
        assert gradient.abs().sum() > 0
        assert not hasattr(model, 'comparison_reader')
        measured.append(live)
        versions.append(tuple(p._version for p in params))
        return value
    monkeypatch.setattr(WholePaths, 'output', observed)
    try:
        batch = next(iter(data.data_loader(split='train', num_streams=4)))
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.01), batchSize=4,
            batch_override=(model.inputSpace.prepInput(batch[0]), model.outputSpace.prepOutput(batch[1])))
        assert len(measured) == 2 and versions[0] == versions[1]
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_numeric_path_reads_its_composed_state_with_live_gradient():
    from Models import BasicModel
    model, state, parameters = _sentence(True)
    error = BasicModel._sentence_numeric_error(model, state, 0, torch.ones(2, dtype=torch.bool), {'record': None})
    gradients = torch.autograd.grad(error.sum(), parameters, allow_unused=True)
    assert all(gradient is not None and bool(gradient.abs().any()) for gradient in gradients)


def test_unsupervised_data_does_not_start_an_output_path():
    from WholePaths import output
    from Layers import Error
    model, state, _ = _sentence(False)
    assert output(model, state, {}, 0, torch.ones(2, dtype=torch.bool), Error()) == (None, None, None)
