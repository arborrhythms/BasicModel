"""Inverse work reductions preserve the complete menu and walk pullbacks."""
from types import MethodType, SimpleNamespace

import torch


class Unary(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float64))
        self.calls = 0
        self.activation = None

    def generate(self, parent):
        self.calls += 1
        value = .5 + (parent-.5) * self.scale
        self.activation = value.detach()
        return value

    def compose(self,child):
        return .5 + (child-.5) / self.scale


def language(ops):
    from Language import LanguageSpace
    owner = SimpleNamespace(_generate_binary_ops=(), _generate_unary_ops=ops)
    owner.generate_unary_step = MethodType(LanguageSpace.generate_unary_step, owner)
    return owner


def test_inverse_menu_evaluates_each_unary_once_with_the_same_values_and_gradients():
    from Generative import inverse_menu
    ops = tuple(Unary(value) for value in (1.1, .9, -.5))
    owner = language(ops)
    parent = torch.tensor([[.2, .4], [.3, .6]], dtype=torch.float64, requires_grad=True)
    live = torch.tensor([True, False])
    options = dict(basis=None, basis_valid=None, require_symbols=False)
    left, right, legal = inverse_menu(owner, parent, live, **options)[:3]
    assert [op.calls for op in ops] == [1, 1, 1]
    expected = torch.stack([torch.where(live[:, None], (.5+(parent-.5)*op.scale), parent)
                            for op in ops] + [parent], 1)
    torch.testing.assert_close(left, expected, rtol=0, atol=0)
    torch.testing.assert_close(right, parent[:, None].expand_as(right), rtol=0, atol=0)
    assert legal.tolist() == [[True]*4, [False]*3+[True]]
    parameters = (parent, *(op.scale for op in ops))
    weights = torch.arange(left.numel(), dtype=left.dtype).reshape_as(left) / left.numel()
    actual_grad = torch.autograd.grad((left*weights).sum(), parameters, retain_graph=True)
    expected_grad = torch.autograd.grad((expected*weights).sum(), parameters)
    for actual, wanted in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, wanted, rtol=1e-15, atol=1e-15)


def test_repeated_inverse_states_preserve_walk_choices_gradients_and_optimizer_steps(monkeypatch, tmp_path):
    from Models import BasicModel, _ensure_grad_anchors
    import Generative
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    op = Unary(-1.)
    owner = language((op,)*9)
    owner.generate_policy_logits = lambda value,features=None: value.new_tensor([0.]*9+[1.]).expand(len(value), -1)
    owner.choose_generate = lambda logits: logits.argmax(-1)
    model = SimpleNamespace(languageSpace=owner)
    original = Generative.inverse_menu
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(Generative, 'inverse_menu', counted)
    cached_type = Generative.InverseMenuRead
    outcomes = []
    for cached in (False, True):
        monkeypatch.setattr(Generative, 'InverseMenuRead', cached_type if cached else lambda *a, **k: None)
        with torch.no_grad():
            op.scale.fill_(-1.)
        event = torch.tensor([[[.25, .375], [0., 0.]], [[.5, .75], [0., 0.]]],
                             dtype=torch.float64, requires_grad=True)
        _ensure_grad_anchors(event.device, (event.dtype,))
        optimizer = torch.optim.SGD(op.parameters(), lr=.01, momentum=.9)
        optimizer.state[op.scale]['momentum_buffer'] = torch.full_like(op.scale, .1)
        optimizer.zero_grad(set_to_none=True)
        calls.clear()
        trace = torch.zeros(2, 12, dtype=torch.long)
        result = BasicModel._output_generate_walk(model, event, 12, require_symbols=False,
            exploit_actions=trace, departure=torch.tensor([9, 9]), return_candidates=True)
        result[0].sum().backward()
        gradient = (event.grad.clone(), op.scale.grad.clone())
        optimizer.step()
        outcomes.append((result, gradient, op.scale.detach().clone(), len(calls), op.activation.clone()))
    for actual, wanted in zip(outcomes[1][:3], outcomes[0][:3]):
        if isinstance(actual, tuple):
            for a, b in zip(actual, wanted):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    torch.testing.assert_close(outcomes[1][4], outcomes[0][4], rtol=0, atol=0)
    assert outcomes[0][3] == 10 and outcomes[1][3] == 2
    assert outcomes[1][0][4].tolist() == [[0]*9+[9, -1, -1]]*2
    import json, os
    from pathlib import Path
    folder = Path(os.environ.get('ITEM27_RECEIPT', tmp_path))
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'inverse-work.json').write_text(json.dumps(dict(
        budget=12, native_stack_capacity=2, executed_steps_per_stream=10,
        uncached_menu_evaluations=outcomes[0][3], cached_menu_evaluations=outcomes[1][3],
        exact_values=True, exact_trace=True, exact_gradients=True,
        exact_optimizer_step=True, exact_activation=True, seed=None), indent=2)+'\n')


def test_inverse_cache_is_stack_bounded_and_distinguishes_active_rows(monkeypatch):
    import Generative
    calls = []
    def menu(language, parent, live, **options):
        calls.append(1)
        value = torch.where(live[:, None], parent, 0.)
        from generation_fixtures import exact_menu
        return exact_menu(value[:,None],value[:,None],live[:,None])
    monkeypatch.setattr(Generative, 'inverse_menu', menu)
    cache = Generative.InverseMenuRead(language(()), capacity=2)
    parent, live = torch.ones(2, 3), torch.tensor([True, False])
    with torch.no_grad():
        first = cache.read(parent, live)
        assert torch.equal(cache.read(parent.clone(), live.clone())[0], first[0])
        other = cache.read(parent, ~live)
        assert not torch.equal(other[0], first[0]) and len(calls) == 2
        cache.read(parent*2, live)
        assert len(cache.entries) == 2
        cache.read(parent, live)
        assert len(calls) == 4 and len(cache.entries) == 2


def test_selected_unary_inverse_evaluates_only_live_choices():
    from Generative import selected_inverse
    ops = tuple(Unary(value) for value in (1.1, .9, -.5))
    owner = language(ops)
    parent = torch.tensor([[.2, .4], [.3, .6], [.1, .5]], dtype=torch.float64,
                          requires_grad=True)
    left, right = selected_inverse(owner, parent, torch.tensor([0, 2, 1]),
        torch.tensor([True, True, False]), basis=None, basis_valid=None,
        require_symbols=False)
    assert [op.calls for op in ops] == [1, 0, 1]
    expected = torch.stack(((.5+(parent[0]-.5)*ops[0].scale), (.5+(parent[1]-.5)*ops[2].scale), parent[2]))
    torch.testing.assert_close(left, expected, rtol=0, atol=0)
    torch.testing.assert_close(right, parent, rtol=0, atol=0)
    actual = torch.autograd.grad(left.sum(), (parent, ops[0].scale, ops[2].scale), retain_graph=True)
    wanted = torch.autograd.grad(expected.sum(), (parent, ops[0].scale, ops[2].scale))
    for a, b in zip(actual, wanted):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_unselected_unary_keeps_zero_gradient_optimizer_participation(monkeypatch):
    from Models import BasicModel, _ensure_grad_anchors
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    selected, rejected = Unary(-1.), Unary(.5)
    owner = language((selected, rejected))
    owner.generate_policy_logits = lambda x,features=None: x.new_tensor([0., 0., 1.]).expand(len(x), -1)
    owner.choose_generate = lambda logits: logits.argmax(-1)
    model = SimpleNamespace(languageSpace=owner)
    event = torch.tensor([[[.2, .4], [0., 0.]]], dtype=torch.float64, requires_grad=True)
    _ensure_grad_anchors(event.device, (event.dtype,))
    optimizer = torch.optim.SGD([selected.scale, rejected.scale], lr=.01, momentum=.9)
    optimizer.state[rejected.scale]['momentum_buffer'] = torch.full_like(rejected.scale, .1)
    result = BasicModel._output_generate_walk(model, event, 6, require_symbols=False,
        exploit_actions=torch.tensor([[0, 0, 2, 2, 2, 2]]), departure=torch.tensor([6]),
        return_candidates=True)
    result[0].sum().backward()
    assert rejected.scale.grad is not None and rejected.scale.grad.item() == 0
    optimizer.step()
    torch.testing.assert_close(rejected.scale, torch.tensor(.5-.01*.9*.1, dtype=torch.float64),
                               rtol=0, atol=0)


def test_inverse_whole_preparation_reuses_only_the_detached_dictionary_read(monkeypatch):
    from FormCentroids import CandidateWholes, prepare_inverse, unary
    forms = torch.tensor([[[.2, .8], [.7, .3]]], dtype=torch.float64)
    support = torch.zeros(1, 2, 256, dtype=torch.bool)
    support[0, 0, 0], support[0, 1, 1] = True, True
    properties = torch.ones(2, 256, dtype=torch.float64)
    properties[0, 1], properties[1, 0] = .2, .3
    bank = CandidateWholes(forms, support, torch.ones(1, 2, dtype=torch.bool),
                           properties, torch.eye(2, dtype=torch.float64))
    read, calls = CandidateWholes.read, []
    def counted(self, *parts):
        calls.append(1)
        return read(self, *parts)
    monkeypatch.setattr(CandidateWholes, 'read', counted)
    prepared = prepare_inverse(bank)
    assert len(calls) == 1
    parent = torch.tensor([[.45, .55, .3]], dtype=torch.float64, requires_grad=True)
    for _ in range(6):
        actual = unary(parent, prepared)
    assert len(calls) == 1
    expected = unary(parent, bank)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    a, = torch.autograd.grad(actual.sum(), parent)
    b, = torch.autograd.grad(expected.sum(), parent)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    before = len(calls)
    per_stream = prepared.for_stream(0)
    per_stream.read(per_stream.forms)
    assert len(calls) == before
    prepared.read(parent)
    assert len(calls) == before+1, 'non-dictionary arguments must be read normally'
    differentiable = bank._replace(codes=bank.codes.clone().requires_grad_(True))
    assert prepare_inverse(differentiable) is differentiable
