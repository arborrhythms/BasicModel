"""Matrix-free forward actions preserve the retained dense kernel and credit."""
import copy

import pytest
import torch

import MeaningKernels
from MeaningKernels import MeaningAction, householder, householder_action


@pytest.mark.parametrize('transpose', [False, True])
def test_reflection_action_matches_dense_product_and_all_gradients(transpose):
    vectors = (torch.arange(2*4*5, dtype=torch.float64).reshape(2, 4, 5)/40-.4).requires_grad_()
    values = (torch.arange(2*2*5, dtype=torch.float64).reshape(2, 2, 5)/20-.3).requires_grad_()
    with torch.no_grad():
        vectors[0, 1].zero_()
    matrix = householder(vectors)
    if transpose:
        matrix = matrix.transpose(-1, -2)
    expected = (matrix[..., None, :, :] @ values[..., None])[..., 0]
    actual = householder_action(values, vectors, transpose=transpose)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    a = torch.autograd.grad(actual.square().sum(), (values, vectors), retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), (values, vectors))
    for found, wanted in zip(a, b):
        torch.testing.assert_close(found, wanted, rtol=1e-11, atol=1e-11)


def dense_action(action, value, verb, **basis):
    from Layers import cube_chart, cube_unchart
    q, allowed = action.basis(value, **basis)
    w = action.spectrum(verb) * allowed
    lanes = value.unflatten(-1, (2, -1))
    present = lanes.ne(0)
    chart = torch.where(present, cube_chart(lanes, bounded=False), 0.)
    projected = (q.transpose(-1, -2)[..., None, :, :] @ chart[..., None])[..., 0]
    changed = (q[..., None, :, :] @ (projected*w.exp()[..., None, :])[..., None])[..., 0]
    result = torch.where(present, cube_unchart(changed), 0.).flatten(-2)
    result = torch.where(allowed[..., None, :].expand_as(lanes).flatten(-2), result, value)
    return MeaningKernels.exact_initial(value, result, (w == 0).all(-1, keepdim=True))


@pytest.mark.parametrize('modifier', [False, True])
@pytest.mark.parametrize('source', ['group', 'object', 'mixed'])
def test_forward_action_matches_dense_values_and_parameter_credit(modifier, source):
    action = MeaningAction(8, modifier=modifier).double()
    with torch.no_grad():
        for i, parameter in enumerate(action.parameters()):
            parameter.add_((torch.arange(parameter.numel(), dtype=parameter.dtype)
                            .reshape_as(parameter)+i+1)/300)
    reference = copy.deepcopy(action)
    value = (torch.arange(16, dtype=torch.float64).reshape(2, 8)/40+.15).requires_grad_()
    verb = value.detach().flip(-1).requires_grad_()
    obj = value.detach().roll(1, -1).requires_grad_()
    with torch.no_grad():
        value[0, 2] = 0
        obj[0, 1] = obj[0, 5] = 0
    args = dict(group=torch.tensor([0, -1]))
    if source != 'group':
        args['object_value'] = obj
        args['object_present'] = None if source == 'object' else torch.tensor([True, False])
    expected = dense_action(reference, value, verb, **args)
    actual = action(value, verb, **args)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    a = torch.autograd.grad(actual.square().sum(), (value, verb, obj, *action.parameters()),
                            retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(expected.square().sum(), (value, verb, obj, *reference.parameters()),
                            allow_unused=True)
    for found, wanted in zip(a, b):
        assert (found is None) == (wanted is None)
        if found is not None:
            torch.testing.assert_close(found, wanted, rtol=1e-10, atol=1e-10)


def test_forward_object_action_does_not_construct_a_dense_basis(monkeypatch):
    action = MeaningAction(896)
    value = torch.full((28, 1, 896), .3, requires_grad=True)
    obj = torch.full_like(value, .4, requires_grad=True)
    calls = []
    original = MeaningKernels.householder
    def observed(vectors):
        calls.append(tuple(vectors.shape))
        return original(vectors)
    monkeypatch.setattr(MeaningKernels, 'householder', observed)
    result = action(value, obj, object_value=obj, group=torch.full((28, 1), -1),
                    object_present=torch.ones((28, 1), dtype=torch.bool))
    torch.testing.assert_close(result, value, rtol=0, atol=0)
    result.sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert calls == [], calls


@pytest.mark.parametrize('modifier', [False, True])
def test_forward_reflection_action_compiles_with_unchanged_credit(modifier):
    action = MeaningAction(8, modifier=modifier).double()
    with torch.no_grad():
        action.readout.fill_(.08)
        action.object_vectors[0, 1, 0] = .2
    value = torch.linspace(.15, .75, 16, dtype=torch.float64).reshape(2, 8).requires_grad_()
    context = value.detach().flip(-1).requires_grad_()
    args = dict(object_value=context, group=torch.tensor([0, -1]),
                object_present=torch.tensor([True, False]))
    expected = action(value, context, **args)
    compiled = torch.compile(action, backend='eager', fullgraph=True, dynamic=False)
    actual = compiled(value, context, **args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameters = (value, context, *action.parameters())
    a = torch.autograd.grad(actual.sum(), parameters, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(expected.sum(), parameters, allow_unused=True)
    for found, wanted in zip(a, b):
        assert (found is None) == (wanted is None)
        if found is not None:
            torch.testing.assert_close(found, wanted, rtol=0, atol=0)
