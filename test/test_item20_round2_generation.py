"""Candidate-owned action bases, independent of the input derivation."""
from types import MethodType, SimpleNamespace

import pytest
import torch

from test_meaning_kernels import paired


def proposal(kind):
    from DecompositionChooser import DecompositionChooser
    from Language import LanguageSpace, LiftLayer, VerbLayer, AdverbLayer
    from Generative import propose_actions, verb_groups
    lift = (AdverbLayer if kind == 'adverb' else LiftLayer)(12, 12).double()
    vp = VerbLayer(12, 12).double()
    with torch.no_grad():
        for index, parameter in enumerate(lift.meaning_action.parameters()):
            parameter.add_((torch.arange(parameter.numel(), dtype=parameter.dtype)
                            .reshape_as(parameter) + index + 1) / 300)
    words = torch.tensor([[[.8,.1,.1,.1,.15,.25,.35,.45,.2,.3,.4,.5],
                           [.1,.8,.1,.1,.7,.6,.5,.4,.6,.5,.4,.3],
                           [.1,.1,.8,.1,.2,.6,.3,.5,.3,.5,.2,.6]]], dtype=torch.float64)
    subject, verb, obj = words.unbind(1)
    right = vp.compose(verb, obj) if kind == 'object' else verb
    language = SimpleNamespace(_generate_binary_ops=(lift, vp), _generate_unary_ops=(),
                               decomposition_chooser=DecompositionChooser())
    for name in ('reverse_binary_step', '_reverse_of_binary_op', '_finish_binary_inverse',
                 'generate_unary_step', 'choose_generate', 'verb_group_table'):
        setattr(language, name, MethodType(getattr(LanguageSpace, name), language))
    language._bounded_binary_reconstruction = LanguageSpace._bounded_binary_reconstruction
    language.verb_action = lambda: lift.meaning_action
    lift.meaning_action.groups.assignments[21] = 0
    lift.meaning_action.groups.used = 1
    ids = torch.tensor([[20, 21, 22]])
    table = language.verb_group_table(SimpleNamespace(_event_symbol=lambda value: value), ids, words)
    groups = verb_groups(ids, table)
    valid = torch.ones(1, 3, dtype=torch.bool)
    structure = right[:, None] if kind == 'object' else None
    bank = propose_actions(language, words, valid, groups=groups,
        constituents=structure, constituent_valid=None if structure is None else valid[:, :1], limit=3)
    context = dict(object_value=obj, object_present=torch.tensor([kind == 'object']),
                   group=torch.tensor([0]))
    parent = lift.compose_blocks(lift.compose, subject, right, action_context=context)
    return language, words, right, parent, bank, groups, structure


@pytest.mark.parametrize('kind', ['object', 'group', 'adverb'])
def test_free_candidate_checks_its_own_action_basis(paired, kind):
    from Generative import native_recompose, candidate_action_context, absolute_residual, EXACT_RESIDUAL_TOLERANCE
    language, words, right, parent, bank, groups, structure = proposal(kind)
    op = language._generate_binary_ops[0]
    before = native_recompose(op, parent, words[:, 0], right, form_width=4)
    context = candidate_action_context(op, right, bank)
    after = native_recompose(op, parent, words[:, 0], right, form_width=4, action_context=context)
    def residual(value):
        form, lanes = absolute_residual(parent, value, 4, numeric=False)
        return float(torch.cat((form[:, None], lanes), -1).max().detach())
    assert residual(after) <= EXACT_RESIDUAL_TOLERANCE
    if kind == 'adverb':
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    else:
        assert residual(before) > EXACT_RESIDUAL_TOLERANCE
        assert context['object_present'].item() == (kind == 'object')
        assert context['group'].item() == 0
        if kind == 'object':
            torch.testing.assert_close(context['object_value'], words[:, 2], rtol=0, atol=1e-15)


@pytest.mark.parametrize('kind', ['object', 'group'])
@pytest.mark.parametrize('execution', ['value', 'gradient', 'compiled_gradient'])
def test_free_walk_has_no_action_record_or_live_stm(paired, monkeypatch, kind, execution):
    import ChunkJournal, util
    from Models import BasicModel, _ensure_grad_anchors
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    gradient = execution != 'value'
    def forbidden(*args, **kwargs):
        raise AssertionError('a free walk read an input record or live STM')
    monkeypatch.setattr(ChunkJournal, 'action_context', forbidden)
    monkeypatch.setattr(ChunkJournal, 'read_identity', forbidden)
    language, words, right, parent, bank, groups, structure = proposal(kind)
    class FreeOwner:
        languageSpace = language
        stm = property(forbidden)
        desiredOutput = property(forbidden)
    def policy(value, features=None):
        at_parent = (value[:, :4] == parent[:, :4]).all(-1)
        at_phrase = (value[:, :4] == right[:, :4]).all(-1) & (kind == 'object')
        choice = torch.where(at_parent, 0, torch.where(at_phrase, 1, 2))
        return torch.nn.functional.one_hot(choice, 3).to(value) * 10
    language.generate_policy_logits = policy
    event = torch.cat((parent.detach()[:, None], torch.zeros_like(words)), 1).requires_grad_(gradient)
    _ensure_grad_anchors(event.device, (event.dtype,))
    def walk(value):
        return BasicModel._output_generate_walk(FreeOwner(), value, 6, basis=words,
            basis_valid=torch.ones(1, 3, dtype=torch.bool),verb_groups=groups,
            constituents=structure,constituent_valid=None if structure is None else torch.ones(1, 1, dtype=torch.bool),
            candidate_limit=3,return_trace=True)
    if execution == 'compiled_gradient':
        walk = torch.compile(walk, backend='eager', fullgraph=True)
    with torch.set_grad_enabled(gradient):
        result = walk(event)
        assert result[1].item() == (3 if kind == 'object' else 2)
        assert not result[2].any()
        assert result[5].all(), 'every selected candidate must recompose exactly'
        if gradient:
            result[0].sum().backward()
            assert event.grad is not None and torch.isfinite(event.grad).all()


def test_unknown_candidate_basis_is_identity(paired):
    from Generative import candidate_action_context, native_recompose
    language, words, right, parent, bank, groups, structure = proposal('group')
    op = language._generate_binary_ops[0]
    unknown = right.clone(); unknown[:, :4] *= .5
    context = candidate_action_context(op, unknown, bank)
    assert not context['object_present'].any() and context['group'].eq(-1).all()
    torch.testing.assert_close(native_recompose(op, parent, words[:, 0], unknown, form_width=4,
        action_context=context), native_recompose(op, parent, words[:, 0], unknown, form_width=4), rtol=0, atol=0)


def test_nonfinite_candidate_action_is_not_an_absent_record(paired):
    from Generative import native_recompose, candidate_action_context
    language, words, right, parent, bank, groups, structure = proposal('object')
    op = language._generate_binary_ops[0]
    with torch.no_grad():
        op.meaning_action.readout[0, 0] = float('nan')
    with pytest.raises(RuntimeError):
        native_recompose(op, parent, words[:, 0], right, form_width=4,
                         action_context=candidate_action_context(op, right, bank))


def test_candidate_object_and_group_remain_stream_local(paired):
    from Generative import propose_actions, candidate_action_context, selected_inverse, selected_residual
    language, words, phrase, transitive, bank, groups, structure = proposal('object')
    op = language._generate_binary_ops[0]
    intransitive = op.compose_blocks(op.compose, words[:, 0], words[:, 1],
                                    action_context=dict(group=torch.tensor([0])))
    parent = torch.cat((transitive, intransitive))
    words, groups = words.repeat(2, 1, 1), groups.repeat(2, 1)
    structural = torch.cat((structure, words[1:2, 1:2]))
    structural_valid = torch.tensor([[True], [False]])
    valid = torch.ones(2, 3, dtype=torch.bool)
    bank = propose_actions(language, words, valid, groups=groups,
        constituents=structural, constituent_valid=structural_valid, limit=3)
    right = torch.cat((phrase, words[1:2, 1]))
    context = candidate_action_context(op, right, bank)
    assert context['object_present'].tolist() == [True, False]
    assert context['group'].tolist() == [0, 0]
    choice, live = torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool)
    left, right = selected_inverse(language, parent, choice, live, basis=words, basis_valid=valid,
        constituents=structural, constituent_valid=structural_valid,limit=3,action_candidates=bank)
    assert selected_residual(language, parent, left, right, choice, live,
        form_width=4,action_candidates=bank).max().item() <= 1e-8
