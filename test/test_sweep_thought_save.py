"""Thought-reader saved values retain the exact pre-update cotangent once."""
from types import SimpleNamespace

import torch


def test_repeated_thought_reader_saves_one_frozen_parameter_view(monkeypatch):
    import ThoughtAnswer
    from Meaning import ConceptualMeaning

    class Reader(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Six coordinates and sixteen repeated reads expose copying per
            # use without allocating a native dictionary in this regression.
            self.weight = torch.nn.Parameter(torch.arange(36, dtype=torch.float64).reshape(6, 6) / 36)
            self.register_buffer('scale', torch.linspace(.5, 1., 6, dtype=torch.float64))
            self.input = torch.linspace(.1, .6, 6, dtype=torch.float64)[None].requires_grad_()
            self.inputSpace = SimpleNamespace(data=SimpleNamespace(
                what=lambda question: SimpleNamespace(available=False)))
            self._sentence_answer_questions = (object(),)
            self.comparison_reader = None

        def costs(self):
            return sum((self.input @ self.weight * self.scale).square().sum(-1)
                       for _ in range(16))

        def _sentence_reader_error(self, state, sentence, active, observation, *, registry):
            value = self.costs()
            registry.add('answer.what', value, category='prediction', objective='output')
            return value

    model = Reader()
    reference = model.costs()
    expected = torch.autograd.grad(reference.sum(), (model.weight, model.input))
    meaning = ConceptualMeaning.from_description(torch.ones(6, dtype=torch.float64))
    observation = {'meanings': [meaning]}
    # This regression isolates saved values, not the independently certified
    # substitution of a concluded meaning into its row.
    monkeypatch.setattr(ThoughtAnswer, 'bound_view',
                        lambda state, view, sentence, row, meaning: (state, view))
    score, trials = ThoughtAnswer.scorer(model, (), observation, 0)
    watched = {model.weight.untyped_storage()._cdata: 'weight',
               model.scale.untyped_storage()._cdata: 'scale'}
    copies = {'weight': 0, 'scale': 0}
    clone = torch.Tensor.clone

    def counted(value, *args, **kwargs):
        key = watched.get(value.untyped_storage()._cdata)
        if key is not None:
            copies[key] += 1
        return clone(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'clone', counted)
        value = score(0, SimpleNamespace(meaning=meaning))
    assert copies == {'weight': 1, 'scale': 1}
    torch.testing.assert_close(value, reference[0], rtol=0, atol=0)
    assert len(trials) == 1 and trials[0][0]._terms['answer.what']['trained']
    with torch.no_grad():
        model.weight.add_(3)
        model.scale.mul_(2)
    actual = torch.autograd.grad(value, (model.weight, model.input))
    for gradient, old_gradient in zip(actual, expected):
        torch.testing.assert_close(gradient, old_gradient, rtol=0, atol=0)
