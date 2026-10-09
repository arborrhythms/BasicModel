"""Review §11: an affine numeric answer and support-governed decoding."""
from DecompositionChooser import DecompositionChooser
from types import MethodType, SimpleNamespace

import pytest
import torch


def test_numeric_head_does_not_call_learned_retrieval():
    from Models import BasicModel
    from Spaces import SubSpace
    shape = (3, 2)
    value = torch.tensor([[[1., 2.], [0., 0.], [0., 0.]]])
    sub = SubSpace.from_tensor(value, inputShape=shape, outputShape=shape,
                              nInputDim=2, nOutputDim=2)
    class Head:
        concept_ids = ()
        inputShape = shape
        def __call__(self, carrier):
            return carrier
    def forbidden(*args, **kwargs):
        raise AssertionError('the numeric head called nonlinear retrieval')
    model = SimpleNamespace(outputSpace=Head(), word_brackets=True,
        answer_synthesis=False, answer_attention=object(), answer_record_reader=None,
        _answer_attention_step=forbidden, perceptualSpace=SimpleNamespace(subspace=sub))
    state = (value[:, 0], value, torch.ones(1, dtype=torch.long))
    result = BasicModel._forward_head(model, sub, sentence_state=state)
    torch.testing.assert_close(result.materialize(), value)


def decoder():
    from Language import LanguageSpace
    from Models import BasicModel
    class Sum:
        inverse_kind = 'search'
        @staticmethod
        def compose(left, right):
            return left + right
    policy = torch.nn.Linear(2, 2)
    with torch.no_grad():
        policy.weight.zero_()
        policy.bias.copy_(torch.tensor([0., 20.]))
    language = SimpleNamespace(_generate_binary_ops=(Sum(),), _generate_unary_ops=(),
        generate_policy=policy, _generate_policy_width=2,
        decomposition_chooser=DecompositionChooser())
    for name in ('generate_policy_logits', 'reverse_binary_step', '_reverse_of_binary_op'):
        setattr(language, name, MethodType(getattr(LanguageSpace, name), language))
    language._bounded_binary_reconstruction = LanguageSpace._bounded_binary_reconstruction
    language.choose_generate = lambda logits: logits.argmax(-1)
    model = SimpleNamespace(languageSpace=language)
    model._output_generate_walk = MethodType(BasicModel._output_generate_walk, model)
    return model, policy


@pytest.mark.parametrize('chunk_in_bank', [False, True])
@pytest.mark.parametrize('compiled', [False, True])
def test_supported_compound_cannot_stop_even_with_a_chunk_code(monkeypatch, chunk_in_bank, compiled):
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _ = decoder()
    event = torch.tensor([[[1., 1.], [0., 0.], [0., 0.]]])
    bank = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]])
    valid = torch.tensor([[True, True, chunk_in_bank]])
    fn = (torch.compile(model._output_generate_walk, backend='inductor', fullgraph=True)
          if compiled else model._output_generate_walk)
    out, count, truncated, _, actions = fn(
        event, 6, basis=bank, basis_valid=valid, return_trace=True)
    assert count.tolist() == [2]
    assert not truncated.any()
    assert actions.tolist() == [[0, 1, 1, -1, -1, -1]]
    torch.testing.assert_close(out[:, :2], bank[:, :2])


def test_single_symbol_cannot_be_split_by_a_large_undo_logit(monkeypatch):
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, policy = decoder()
    with torch.no_grad():
        policy.bias.copy_(torch.tensor([20., 0.]))
    event = torch.tensor([[[1., 0.], [0., 0.], [0., 0.]]])
    bank = torch.tensor([[[1., 0.], [0., 1.]]])
    out, count, truncated, _, actions = model._output_generate_walk(
        event, 6, basis=bank, basis_valid=torch.ones(1, 2, dtype=torch.bool), return_trace=True)
    assert count.tolist() == [1]
    assert not truncated.any()
    assert actions.tolist() == [[1, -1, -1, -1, -1, -1]]
    torch.testing.assert_close(out[:, 0], event[:, 0])


def test_no_shortlist_support_is_unavailable_not_a_pseudo_terminal(monkeypatch):
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _ = decoder()
    event = torch.tensor([[[1., 1.], [0., 0.]]], requires_grad=True)
    out, count, truncated, _, actions = model._output_generate_walk(
        event, 4, basis=event, basis_valid=torch.zeros(1, 2, dtype=torch.bool), return_trace=True)
    assert count.tolist() == [0]
    assert truncated.all()
    assert actions.tolist() == [[-1, -1, -1, -1]]
    assert torch.isfinite(out).all()


def test_capacity_does_not_make_stop_eligible_for_a_compound(monkeypatch):
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _ = decoder()
    event = torch.tensor([[[1., 1.]]])
    bank = torch.tensor([[[1., 0.], [0., 1.]]])
    out, count, truncated, _, actions = model._output_generate_walk(
        event, 4, basis=bank, basis_valid=torch.ones(1, 2, dtype=torch.bool), return_trace=True)
    assert count.tolist() == [0] and truncated.all()
    assert actions.tolist() == [[-1, -1, -1, -1]]
    assert torch.isfinite(out).all()


def test_free_policy_has_no_pathwise_credit_between_supported_operations(monkeypatch):
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _ = decoder()
    language = model.languageSpace
    class Difference:
        inverse_kind = 'search'
        @staticmethod
        def compose(left, right):
            return left - right
    language._generate_binary_ops += (Difference(),)
    language.generate_policy = torch.nn.Linear(2, 3)
    with torch.no_grad():
        language.generate_policy.weight.zero_()
        language.generate_policy.bias.copy_(torch.tensor([1., 0., 20.]))
    event = torch.tensor([[[1., 1.], [0., 0.], [0., 0.]]], requires_grad=True)
    bank = torch.tensor([[[1., 0.], [0., 1.], [1., 2.]]])
    out, count, truncated, _, actions, alternatives = model._output_generate_walk(
        event, 6, basis=bank, basis_valid=torch.ones(1, 3, dtype=torch.bool), return_candidates=True)
    assert alternatives[0, 0] and actions[0, 0] == 0
    assert count.tolist() == [2] and not truncated.any()
    out[:, 0].sum().backward()
    gradient = language.generate_policy.bias.grad
    assert gradient is None or not gradient.any()  # CE is supplied only by the teacher


@pytest.mark.parametrize('configuration', ['XOR_grammar.xml', 'MM_xor.xml'])
def test_actual_numeric_routes_never_retrieve(monkeypatch, configuration):
    from test_mm_xor import _fresh_model
    model, _, data = _fresh_model('data/' + configuration)
    def forbidden(*args, **kwargs):
        raise AssertionError('nonlinear reader reached a numeric gate')
    monkeypatch.setattr(model.answer_attention, 'forward', forbidden)
    raw, _ = next(iter(data.data_loader(split='train', num_streams=4)))
    with torch.no_grad():
        result = model.forward(model.inputSpace.prepInput(raw))
    assert torch.isfinite(result[2]).all()
    model.End()


def test_thought_retrieval_is_live_bounded_and_does_not_write_keys():
    from Attention import PrimedSymbolReader
    from Queries import ConceptualMeaning
    from QueryWork import QueryWorkBudget
    from ThoughtFeatures import attend_meanings
    reader = PrimedSymbolReader()
    query = ConceptualMeaning.from_description(torch.tensor([1., 1.]))
    first = ConceptualMeaning.from_description(torch.tensor([1., 0.], requires_grad=True))
    second = ConceptualMeaning.from_description(torch.tensor([0., 2.], requires_grad=True))
    records = ((first, 1.), (second, .5))
    base = attend_meanings(query, records)
    with torch.no_grad():
        reader.consume_gate.fill_(.5)
    work = QueryWorkBudget(1)
    fed = attend_meanings(query, records, reader=reader, work=work)
    assert not torch.equal(fed, base) and work.spent == 1
    torch.testing.assert_close(fed[-10:], base[-10:])
    fed.square().sum().backward()
    assert reader.consume_gate.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in reader.scorer.parameters())
    exhausted = attend_meanings(query, records, reader=reader, work=work)
    torch.testing.assert_close(exhausted, base)
    assert work.spent == 1
