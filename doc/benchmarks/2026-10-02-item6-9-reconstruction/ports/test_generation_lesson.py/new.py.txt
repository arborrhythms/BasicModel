"""The sentence's trained generation lesson retains its diagnostic credit."""
from types import SimpleNamespace

import torch


def test_sentence_generation_lesson_keeps_its_weighted_output_gradient():
    from Models import BasicModel
    from Layers import Error

    generate = torch.tensor(2., requires_grad=True)
    compose = torch.tensor(3., requires_grad=True)
    lesson_errors = {'compose': Error(), 'generate': Error()}
    lesson_errors['compose'].squared('compose', compose + 1., torch.ones_like(compose), category='expectation')
    lesson_errors['generate'].squared('generate', generate + 1., torch.ones_like(generate), category='output')
    batch, words, width = 2, 1, 3
    lang = [None] * 15
    lang[9] = torch.zeros(batch, 1, width, requires_grad=True)
    lang[13] = torch.zeros(batch, 1, 3 * width)
    lang[14] = torch.ones(batch, 1, dtype=torch.long)
    model = SimpleNamespace(
        _publish_sentence_scratch=lambda state: None,
        _trial_understanding=lambda *args: object(),
        _tensor_pushed_ideas=torch.zeros(batch, words, width),
        reconstruct_in_loop=False,
        _sentence_observation=lambda *args: {'entries': (object(), object())},
        _reading_lesson_enabled=True, grammar_lesson_weight=.7,
        _reading_lesson_sources=[0, 1], _reading_lesson_split='train',
        _grammar_lesson_objectives=lambda *args, **kwargs: {
            'compose': compose.square(), 'generate': generate.square()},
        _grammar_lesson_errors=lesson_errors,
        _reading_lesson_reports=[], symbolSpace=SimpleNamespace(),
        _sentence_operator_gradients={})
    cost, *_ = BasicModel._sentence_path_cost(
        model, ((), tuple(lang), ()), 0, torch.ones(batch, dtype=torch.bool))
    output = model._sentence_gradient_objectives['output']
    gradient, = torch.autograd.grad(output.mean(), generate, retain_graph=True)
    trained_gradient, = torch.autograd.grad(cost.mean(), generate)
    torch.testing.assert_close(gradient, .7 * 2 * generate.detach())
    torch.testing.assert_close(gradient, trained_gradient)
    assert not model._reading_lesson_reports[0].requires_grad
