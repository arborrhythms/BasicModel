"""Supplied syntax lessons have no inherited batch question context."""
from types import SimpleNamespace

import torch

from GrammarLessons import compose_loss
from Language import OperationSelectionLayer


def test_teacher_state_uses_neutral_context_without_mutating_the_reading():
    class Sum(torch.nn.Module):
        def forward(self, left, right):
            return left + right
    class Difference(torch.nn.Module):
        def forward(self, left, right):
            return left - right
    layer = OperationSelectionLayer(d_model=4, ops=(Sum(), Difference()), chooser='mlp')
    context = torch.ones(3, layer.chooser.WHAT_CONTEXT_DIM)
    layer._what_context = context
    language = SimpleNamespace(_tree_layer=lambda _: layer,
        _compose_binary_rules=(SimpleNamespace(method_name='sum'), SimpleNamespace(method_name='difference')),
        _compose_unary_rules=())
    program = SimpleNamespace(leaves=torch.eye(4)[:2])
    lesson = {'tree': ['sum', 0, 1]}
    first = compose_loss(language, [program], [lesson])
    assert first is not None and first.requires_grad and torch.isfinite(first)
    assert layer._what_context is context
    layer._what_context = context * -5
    second = compose_loss(language, [program], [lesson])
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    first.backward()
    assert any(parameter.grad is not None and bool(parameter.grad.any())
               for parameter in layer.chooser.parameters())


def test_unary_lesson_credits_the_unary_action_in_the_joint_menu():
    class Sum(torch.nn.Module):
        def forward(self, left, right):
            return left + right
    layer = OperationSelectionLayer(d_model=4, ops=(Sum(),),
        unary_ops=(torch.nn.Identity(),), chooser='mlp')
    language = SimpleNamespace(_tree_layer=lambda _: layer,
        _compose_binary_rules=(SimpleNamespace(method_name='sum'),),
        _compose_unary_rules=(SimpleNamespace(method_name='identity'),))
    program = SimpleNamespace(leaves=torch.eye(4)[:1])
    routes = []
    hook = layer.register_forward_hook(lambda module, args, result: routes.append(result[2]))
    try:
        loss = compose_loss(language, [program], [dict(tree=['identity', 0])])
    finally:
        hook.remove()
    probability = routes[-1]['probabilities']
    probability.retain_grad()
    loss.backward()
    assert probability.shape[-1] == layer.r_apply + 1  # No binary position at depth one.
    assert probability.grad[0, 0] < 0
    assert probability.grad[0, -1] == 0


def test_lesson_validates_an_unnamed_leaf_from_its_captured_bytes(monkeypatch):
    from Models import BasicModel
    import GrammarLessons
    lesson = dict(text='word ?', tree=['sum', 0, 1])
    model = SimpleNamespace(grammar_lesson_weight=1.,
        inputSpace=SimpleNamespace(data=SimpleNamespace(grammar_lessons={'train': [lesson]}),
            _ar_target_word_bytes=torch.tensor([[[119, 111, 114, 100], [63, 0, 0, 0]]]),
            _ar_target_word_mask=torch.tensor([[[True]*4, [True, False, False, False]]])),
        _concept_owner=lambda: SimpleNamespace(word_surface_for_row=lambda row: b'word' if row == 0 else None),
        languageSpace=None, grammatical_thoughts=None, _grammar_target_word=lambda *args: None)
    program = SimpleNamespace(leaves=torch.eye(2), word_rows=torch.tensor([0, -1]),
                              concept_ids=torch.tensor([1, -1]))
    monkeypatch.setattr(GrammarLessons, 'compose_loss', lambda *args, **kwargs: None)
    monkeypatch.setattr(GrammarLessons, 'generate_loss', lambda *args, **kwargs: None)
    assert BasicModel._grammar_lesson_objectives(model, [program], split='train',
        source_rows=[0], source_positions=[torch.tensor([0, 1])], streams=[0]) is None
    import pytest
    lesson['text'] = 'word !'
    with pytest.raises(ValueError, match='does not match'):
        BasicModel._grammar_lesson_objectives(model, [program], split='train',
            source_rows=[0], source_positions=[torch.tensor([0, 1])], streams=[0])


def test_open_binding_annotation_labels_only_the_native_dummy_without_replacing_states():
    from GrammarLessons import constituent_examples, student_target_actions
    language = SimpleNamespace(
        _compose_binary_rules=tuple(SimpleNamespace(method_name=name) for name in ('sum', 'bind')),
        _compose_unary_rules=())
    references = dict(binary_ops=(0, 1, 1), binary_bind_dummy=(False, True, False))
    before = dict(spans=((0,), (1,)), operations={},
                  kwargs=dict(reference_data=references))
    after = dict(before, operations={(1,): (('open', 1),)})
    program = SimpleNamespace(leaves=torch.eye(2), student_states=[before, after])
    examples = constituent_examples(language, program, ['sum', 0, ['bind.open', 1]])
    layer = SimpleNamespace(r_reduce=2)
    assert student_target_actions(layer, examples[0]) == [1]
    assert student_target_actions(layer, examples[1]) == [0]
    assert 'target' not in before and 'target' not in after
    # When its earlier operands have been folded incorrectly, the teacher
    # cannot install or recover the missing open site.
    program.student_states = [dict(before, spans=((0, 1),))]
    assert constituent_examples(language, program, ['sum', 0, ['bind.open', 1]])[0]['target'] == -1


def test_one_concept_generation_annotation_trains_stop_on_the_existing_code():
    from GrammarLessons import generate_loss
    from Language import LanguageSpace
    from types import MethodType
    logits = torch.nn.Linear(3, 2, bias=False)
    language = SimpleNamespace(_generate_binary_names=(), _generate_binary_ops=(),
        _generate_unary_names=('identity',), _generate_unary_ops=(torch.nn.Identity(),),
        generate_policy_logits=logits, operator_cost=1.)
    language.choose_generate = MethodType(LanguageSpace.choose_generate, language)
    point = torch.tensor([.2, .7, 1.])
    program = SimpleNamespace(leaves=torch.zeros(1, 3))
    loss = generate_loss(language, [program], [dict(generation=dict(
        form='concept', text='label', roots=[0]))], None, lambda word: point)
    loss.backward()
    assert (logits.weight.grad[0] > 0).all()
    assert (logits.weight.grad[1] < 0).all()


def test_student_unary_target_precedes_its_open_binding_and_preserves_native_positions():
    from GrammarLessons import constituent_examples, student_target_actions
    language = SimpleNamespace(
        _compose_binary_rules=tuple(SimpleNamespace(method_name=name) for name in ('sum', 'bind')),
        _compose_unary_rules=(SimpleNamespace(method_name='lower'),))
    refs = dict(binary_ops=(0, 1, 1), binary_bind_dummy=(False, True, False), unary_ops=(0, 0))
    state = dict(spans=((0,), (1,)), operations={}, window=torch.zeros(1, 2, 3),
                 kwargs=dict(reference_data=refs))
    states = [state, dict(state, operations={(1,): ((1, 0),)}),
              dict(state, operations={(1,): ((1, 0), ('open', 1))})]
    program = SimpleNamespace(leaves=torch.eye(2), student_states=states)
    examples = constituent_examples(language, program, ['sum', 0, ['bind.open', ['lower', 1]]])
    layer = SimpleNamespace(r_reduce=2, r_apply=1)
    assert [student_target_actions(layer, ex) for ex in examples] == [[5, 6], [1], [0]]
