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
    lang = [None] * 12
    lang[6] = torch.zeros(batch, 1, width, requires_grad=True)
    lang[10] = torch.zeros(batch, 1, 3 * width)
    lang[11] = torch.ones(batch, 1, dtype=torch.long)
    from Understanding import ConceptualField
    from SentenceUnderstanding import SentenceUnderstanding
    field = ConceptualField(torch.empty(0, dtype=torch.long), torch.empty(0, batch, words, 2),
        torch.tensor([[[0, 1]]]).expand(batch, -1, -1),
        torch.tensor([[[1, 2]]]).expand(batch, -1, -1), None)
    record = SentenceUnderstanding(lang[6][:, 0], lang[10][:, 0].reshape(batch, 3, width),
        lang[11][:, 0], lang[6], lang[11], torch.zeros(batch, words, width),
        torch.zeros(batch, words, dtype=torch.long), torch.ones(batch, words, dtype=torch.bool),
        None, torch.zeros(batch, dtype=torch.long), open_field=field)
    program = SimpleNamespace(actions=torch.tensor([[0, 0, 0]]),
        concept_ids=torch.tensor([-1]), interpret_forced=False, tree=None)
    model = SimpleNamespace(
        _publish_sentence_scratch=lambda state: None,
        _trial_understanding=lambda *args: record,
        _concept_owner=lambda: None,
        languageSpace=SimpleNamespace(_tree_layer=lambda arity: SimpleNamespace()),
        _sentence_leaf_positions=lambda record, row: torch.arange(words),
        _student_compose_states=lambda record, row: None,
        _tensor_pushed_ideas=torch.zeros(batch, words, width),
        reconstruct_in_loop=False,
        _sentence_observation=lambda *args: {'entries': (program, program),
            'clauses': (None, None), 'reference_plans': ()},
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
