"""Absolute generation errors in the symbolic and numeric walk charts."""
from types import SimpleNamespace
import pytest
import torch
from Generative import absolute_residual, eligibility_menu, selected_residual, walk_residual


@pytest.mark.parametrize('numeric', [False, True])
@pytest.mark.parametrize('magnitude', [0., 1e-30, .5, 1.])
def test_absolute_lane_errors_match_the_chart_and_native_gradient(numeric, magnitude):
    parent = torch.full((1, 6), magnitude, requires_grad=True)
    recovered = torch.tensor([[.1, .2, .3, .4, .5, .6]], requires_grad=True)
    form, meaning = absolute_residual(parent, recovered, 2, numeric=numeric)
    actual = torch.cat((form[:, None], meaning), -1)
    p, r = (parent.asinh(), recovered.asinh()) if numeric else (parent, recovered)
    expected = (r-p).reshape(1, 3, 2).square().mean(-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    a = torch.autograd.grad(actual.sum(), (parent, recovered), retain_graph=True)
    b = torch.autograd.grad(expected.sum(), (parent, recovered))
    for gradient, reference in zip(a, b):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, reference, rtol=0, atol=0)


def test_zero_and_nearly_zero_parent_lanes_have_the_same_absolute_error():
    parent = torch.zeros(2, 6)
    parent[1] = 1e-30
    for numeric in (False, True):
        form, meaning = absolute_residual(parent, torch.full_like(parent, .5), 2, numeric=numeric)
        errors = torch.cat((form[:, None], meaning), -1)
        assert (errors[0]-errors[1]).abs().max() <= 1e-12


def test_unselected_terminal_is_masked_before_its_residual_and_gradient():
    language = SimpleNamespace(_generate_binary_ops=(), _generate_unary_ops=())
    parent = torch.full((2, 6), 1e-15, requires_grad=True)
    left = torch.full((2, 6), .5, requires_grad=True)
    actual = selected_residual(language, parent, left, left, torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.bool), form_width=2)
    gradients = torch.autograd.grad(actual.sum(), (parent, left))
    assert actual.eq(0).all()
    assert all(torch.isfinite(g).all() and g.eq(0).all() for g in gradients)


@pytest.mark.parametrize('numeric', [False, True])
def test_walk_sums_three_lane_errors_and_stack_energy_without_step_dilution(numeric):
    frontier = torch.tensor([[[.2, .4, .6, .8, -1., 2.], [1., 1., 2., 2., 3., 3.]]], requires_grad=True)
    steps = torch.tensor([[[.1, .2, .3], [.2, .1, .4], [0., 0., 0.]]])
    actual = walk_residual(steps, frontier, frontier.detach()*5, torch.tensor([2]), form_width=2, numeric=numeric)
    chart = frontier.asinh() if numeric else frontier
    expected = chart.reshape(1, 2, 3, 2).square().mean(-1).sum(1) + torch.tensor([[.3, .3, .7]])
    assert actual.shape == (1, 3)
    torch.testing.assert_close(actual, expected)
    padded = torch.cat((steps, torch.zeros(1, 13, 3)), 1)
    torch.testing.assert_close(actual, walk_residual(padded, frontier, frontier.detach(), torch.tensor([2]), form_width=2, numeric=numeric))
    assert torch.isfinite(torch.autograd.grad(actual.sum(), frontier)[0]).all()


def terminal_menu(parent, terminal, *, numeric=False):
    return eligibility_menu(parent, [terminal], [torch.zeros_like(terminal)],
        torch.ones(len(parent), 1, dtype=torch.bool), (), (), [], [],
        torch.zeros(len(parent), 0, dtype=torch.bool), torch.zeros(len(parent), dtype=torch.bool),
        terminal, torch.ones(len(parent), dtype=torch.bool), 2, cube_domain=not numeric)


def test_cube_menu_features_keep_all_three_lane_errors_in_the_unit_interval():
    parent = torch.tensor([[.5, .5, 1e-30, 1e-30, 0., 0.], [0., 1., .2, .8, 1., 0.]])
    terminal = 1-parent
    menu = terminal_menu(parent, terminal)
    assert menu.admissible.all() and menu.features.shape == (2, 1, 5)
    assert ((menu.features[..., :3] >= 0) & (menu.features[..., :3] <= 1)).all()
    torch.testing.assert_close(menu.features[:, 0, :3], (terminal-parent).reshape(2, 3, 2).square().mean(-1))


def test_numeric_terminal_large_signed_values_use_asinh_before_squaring():
    parent = torch.full((1, 6), -1e30, requires_grad=True)
    terminal = torch.full_like(parent, 1e30, requires_grad=True)
    menu = terminal_menu(parent, terminal, numeric=True)
    assert menu.admissible.all() and torch.isfinite(menu.features).all()
    loss = selected_residual(SimpleNamespace(_generate_binary_ops=(), _generate_unary_ops=()),
        parent, terminal, terminal, torch.tensor([0]), torch.tensor([True]), form_width=2, numeric=True).sum()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(loss, (parent, terminal)))


def test_admissible_nonfinite_recomposition_fails_at_its_producer():
    parent = torch.full((1, 6), .5)
    left, right = parent*.5, parent*1.5
    with pytest.raises(RuntimeError, match='admissible generation proposal requires a finite absolute residual'):
        eligibility_menu(parent, [left, left], [right, right], torch.ones(1, 2, dtype=torch.bool),
            (object(),), (), [torch.tensor([True])], [torch.full_like(parent, float('inf'))],
            torch.zeros(1, 1, dtype=torch.bool), torch.tensor([False]), left, torch.tensor([True]), 2)


def test_native_admissible_terminal_has_finite_float32_loss_and_parameter_gradients(tmp_path):
    from Models import BasicModel
    from test_one_backward import receipt
    model = SimpleNamespace(_readback_percept_width=300, _image_form_width=328)
    model.conditioner = torch.nn.Linear(29, 392, bias=False, dtype=torch.float32)
    with torch.no_grad():
        model.conditioner.weight.zero_()
    model._question_conditioner = lambda width, device, dtype: model.conditioner
    idea = torch.full((1, 3, 392), .5)
    idea[:, :, 328:360] = 1e-15
    idea[:, :, 360:] = 0
    parent = BasicModel._condition_answer_on_question(model, idea, torch.ones(1, 29))[:, 0]
    parent.retain_grad()
    terminal = parent.detach().clone()
    terminal[:, 328:360] = .5
    menu = eligibility_menu(parent, [terminal], [torch.zeros_like(terminal)],
        torch.ones(1, 1, dtype=torch.bool), (), (), [], [], torch.zeros(1, 0, dtype=torch.bool),
        torch.tensor([False]), terminal, torch.tensor([True]), 328)
    assert menu.admissible.all() and not menu.exact.any()
    assert torch.isfinite(menu.features).all()
    loss = selected_residual(SimpleNamespace(_generate_binary_ops=(), _generate_unary_ops=()),
        parent, terminal, torch.zeros_like(terminal), torch.tensor([0]), torch.tensor([True]), form_width=328).sum()
    assert torch.isfinite(loss)
    loss.backward()
    gradient = model.conditioner.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
    assert torch.isfinite(parent.grad).all()
    receipt(tmp_path, 'native-generation-float32', dict(loss=float(loss.detach()),
        parent_gradient_max=float(parent.grad.abs().max()),
        parameter_gradient_max=float(gradient.abs().max()), nonfinite=0, admissible=True, dtype='float32'))
