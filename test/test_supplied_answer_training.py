"""Supplied answers train each sentence trial; absent answers retain the cut."""
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
        loss=ModelLoss(reconstruction_scale=1., what_scale=1., where_scale=1., when_scale=1.),
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




def test_answer_cost_does_not_include_a_generate_lesson():
    from Models import BasicModel
    from Layers import Error
    model, state, _ = _sentence(True)
    active = torch.ones(2, dtype=torch.bool)
    registry = Error(row_mask=active)
    registry.add('grammar.generate', torch.tensor([7., 9.]), objective='output')
    answer = BasicModel._sentence_answer_error(model, state, 0, active, {"record": None}, registry=registry)
    torch.testing.assert_close(answer, torch.tensor([36., 100.]))
    torch.testing.assert_close(registry.total(), torch.tensor([43., 109.]))




def test_real_supplied_trials_cost_before_updates_and_reach_only_reader(monkeypatch):
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
        assert not any(name.startswith('conceptualSpaces.') for name in live), live
        assert not any('operation_layer.' in name for name in live), live
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


def test_native_trial_reads_the_completed_point_without_state_credit():
    from Meaning import ConceptualMeaning
    from Models import BasicModel
    from What import What
    model, state, parameters = _sentence(True)
    root = state[1][9][:, 0]
    meanings = [ConceptualMeaning(torch.stack((row, row, torch.zeros_like(row))),
                                  torch.tensor([True, True, False])) for row in root]
    clauses = [SimpleNamespace(point=row, relation=None, meaning=meaning)
               for row, meaning in zip(root, meanings)]
    model.answer_synthesis = True
    model._sentence_answer_questions = tuple(What.supervised(i) for i in range(2))
    model.inputSpace.data.what = lambda question: SimpleNamespace(available=True, what=0.)
    model._what_grammar_context = lambda *args, **kwargs: (None, None)
    model.outputSpace.prepOutput = lambda values: torch.tensor(values)[:, None]
    seen = []
    def realize(understanding, derivation):
        assert [int(field.meaning.role_mask.sum()) for field in derivation.sentence_states] == [1, 1]
        seen.append(derivation.conceptual_answer)
        return SimpleNamespace(actual=(derivation.conceptual_answer.detach().flatten(1) @ parameters[-1])[:, None])
    model.reverseOutput = realize
    error = BasicModel._sentence_answer_error(model, state, 0, torch.ones(2, dtype=torch.bool),
                                              dict(meanings=meanings, clauses=clauses, record=None))
    assert len(seen) == 1
    torch.testing.assert_close(seen[0][:, 0], root)
    gradients = torch.autograd.grad(error.sum(), parameters, allow_unused=True)
    assert gradients[0] is None and gradients[1] is None
    assert gradients[2] is not None and bool(gradients[2].abs().any())


def test_automatic_answers_do_not_start_a_supplied_answer_trial():
    from Models import BasicModel
    from What import What
    model, state, _ = _sentence(True)
    model.answer_synthesis = True
    model._sentence_answer_questions = tuple(What.future(i) for i in range(2))
    model.inputSpace.data.what = lambda question: SimpleNamespace(available=True, what='automatic')
    def forbidden(*args, **kwargs):
        raise AssertionError('an automatic answer must not start a supervised generation trial')
    model.reverseOutput = forbidden
    model._what_grammar_context = lambda *args, **kwargs: (None, None)
    assert BasicModel._sentence_answer_error(model, state, 0, torch.ones(2, dtype=torch.bool),
                                              dict(meanings=[None, None])) is None
