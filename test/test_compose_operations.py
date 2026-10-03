"""Item 7.5: one hard operation, one global softmax, two distinct paths."""
import torch
from torch import nn
import Language


class Add(nn.Module):
    def forward(self, left, right):
        return left + right


class Negate(nn.Module):
    def forward(self, value):
        return -value


def layer():
    result = Language.OperationSelectionLayer(
        d_model=1, ops=[Add()], unary_ops=[Negate()])
    with torch.no_grad():
        result.reduce_anchor.fill_(1)
        result.apply_anchor.fill_(0)
        result.stop_anchor.fill_(100)
    return result


def test_one_binary_operation_across_all_locations():
    step = layer()
    x = torch.tensor([[[1.], [2.], [3.], [4.]]])
    hard, path, route = step(x, slots=1)
    assert route['kind'].tolist() == [1]
    assert route['position'].tolist() == [2]
    assert route['depth'].tolist() == [3]
    torch.testing.assert_close(path, torch.tensor([[[1.], [2.], [7.], [0.]]]))
    assert route['probabilities'][0, -1] == 0
    torch.testing.assert_close(hard, path)


def test_unary_and_binary_compete_in_the_same_softmax():
    step = layer()
    with torch.no_grad():
        step.apply_anchor.fill_(-10)
    x = torch.tensor([[[1.], [2.], [3.]]])
    _, path, route = step(x, slots=1)
    assert route['kind'].tolist() == [2]
    assert route['depth'].tolist() == [3]
    torch.testing.assert_close(path, torch.tensor([[[1.], [2.], [-3.]]]))
    assert route['probabilities'].shape == (1, 6)
    torch.testing.assert_close(route['probabilities'].sum(-1), torch.ones(1))


def test_stop_is_eligible_only_when_each_row_fits():
    step = layer()
    x = torch.ones(2, 3, 1)
    _, _, route = step(x, depth=torch.tensor([3, 2]), slots=torch.tensor([2, 2]))
    assert route['kind'].tolist() == [1, 0]
    assert route['depth'].tolist() == [2, 2]


def test_hard_selected_candidate_keeps_probability_gradient():
    step = layer()
    x = torch.tensor([[[1.], [2.], [3.]]], requires_grad=True)
    _, path, route = step(x, slots=1)
    path.sum().backward()
    assert step.reduce_anchor.grad is not None
    assert step.reduce_anchor.grad.abs().sum() > 0
    assert step.apply_anchor.grad is not None
    assert step.apply_anchor.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()


def test_explore_differs_and_static_budget_can_stop_early():
    step = layer()
    x = torch.tensor([[[1.], [2.], [3.]]])
    exploit, explore = step.derive_pair(x, slots=1, rounds=6, greedy=True)
    assert exploit['actions'].shape == explore['actions'].shape == (1, 6)
    assert (exploit['actions'] != explore['actions']).any(-1).all()
    assert exploit['used'].tolist() == [3]  # two binaries followed by STOP
    assert exploit['complete'].all()
    assert exploit['value'].shape == x.shape
    assert explore['forced_round'].ge(0).all()
    assert (explore['forced_round'] < exploit['used']).all()




def test_fullgraph_round_and_backward_match_eager():
    step = layer()
    x = torch.tensor([[[1.], [2.], [3.]], [[-3.], [-2.], [0.]]], requires_grad=True)
    depth = torch.tensor([3, 2])
    def run(value, count):
        _, result, route = step(value, depth=count, slots=1)
        return result, route['depth'], route['probabilities']
    compiled = torch.compile(run, backend='aot_eager', fullgraph=True)
    eager = run(x, depth)
    captured = compiled(x, depth)
    for a, b in zip(eager, captured):
        torch.testing.assert_close(a, b)
    grad_eager = torch.autograd.grad(eager[0].square().sum(), x, retain_graph=True)[0]
    grad_compiled = torch.autograd.grad(captured[0].square().sum(), x)[0]
    torch.testing.assert_close(grad_eager, grad_compiled)


def test_padding_never_supplies_an_operation_and_inactive_rows_freeze():
    step = layer()
    x = torch.tensor([[[1.], [2.], [1e6]], [[7.], [8.], [9.]]])
    _, path, route = step(x, depth=torch.tensor([2, 3]), slots=1,
                          active=torch.tensor([True, False]))
    torch.testing.assert_close(path[0], torch.tensor([[3.], [0.], [0.]]))
    torch.testing.assert_close(path[1], x[1])
    assert route['valid'].tolist() == [True, False]
    assert route['action'][1] == -1


def test_budget_exhaustion_reports_overlarge_derivation():
    step = layer()
    with torch.no_grad():
        step.apply_anchor.fill_(-100)
    result = step.derive(torch.tensor([[[1.], [2.], [3.]]]), rounds=1, slots=1, greedy=True)
    assert not result['complete'].any()
    # Two reductions cannot fit in one round. The deadline still makes
    # progress instead of spending the infeasible budget on a unary rewrite.
    assert result['depth'].tolist() == [2]
    assert result['traces'][0]['kind'].tolist() == [1]


def test_forced_round_is_uniform_over_exploit_used_rounds(monkeypatch):
    step = layer()
    x = torch.tensor([[[1.], [2.], [3.]]]).expand(6, -1, -1)
    exploit = step.derive(x, slots=1, rounds=6, greedy=True)
    monkeypatch.setattr(torch, 'rand', lambda *a, **kw:
                        torch.tensor([.0, .32, .34, .66, .67, .99], device=kw.get('device')))
    explore = step.derive(x, slots=1, rounds=6, exploit=exploit)
    assert explore['forced_round'].tolist() == [0, 0, 1, 1, 2, 2]
    assert (explore['actions'] != exploit['actions']).any(-1).all()


def test_impossible_distinct_path_is_reported_without_a_fake_noop():
    import pytest
    step = Language.OperationSelectionLayer(d_model=1, ops=[Add()])
    exploit = step.derive(torch.ones(1, 2, 1), slots=1, rounds=1, greedy=True)
    with pytest.raises(RuntimeError, match='no legal alternative'):
        step.derive(torch.ones(1, 2, 1), slots=1, rounds=1, exploit=exploit)
