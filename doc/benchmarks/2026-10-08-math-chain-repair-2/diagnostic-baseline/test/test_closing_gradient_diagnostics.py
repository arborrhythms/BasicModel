"""Read-only diagnostic gradients survive the per-sentence optimizer boundary."""
import math

import pytest
import torch

import GradientDiagnostics as diagnostics


def test_closing_gradients_are_summed_before_cosines_without_retaining_graphs():
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    groups = {'operator.test': [p, p]}
    collected = {}
    first = p.square().sum()
    p.grad = torch.tensor([7., 8.])
    diagnostics.accumulate_objective_gradients(collected,
        {'reconstruction': first, 'expectation': p[0]}, groups)
    torch.testing.assert_close(p.grad, torch.tensor([7., 8.]))
    first.backward()  # the diagnostic does not consume the training graph
    with torch.no_grad():
        p.add_(1.)
    p.grad = None
    second = -p[0] + 3 * p[1]
    diagnostics.accumulate_objective_gradients(collected, {'reconstruction': second}, groups)
    second.backward()  # both closing graphs are now released
    before = p.grad.clone()
    report = diagnostics.objective_agreement({'output': -7 * p[0] + p[1]},
        groups, accumulated=collected)['operator.test']
    assert report['reconstruction_norm'] == pytest.approx(math.sqrt(50.))
    assert report['output_norm'] == pytest.approx(math.sqrt(50.))
    assert report['reconstruction_output_cosine'] == pytest.approx(0., abs=1e-7)
    assert report['reconstruction_expectation_cosine'] == pytest.approx(1 / math.sqrt(50.))
    torch.testing.assert_close(p.grad, before)
    assert all(not g.requires_grad for values in collected.values() for g in values.values())


def test_sparse_closing_gradients_accumulate_without_dense_capacity_buffers(monkeypatch):
    table = torch.nn.Embedding(10000, 2, sparse=True)
    groups = {'operator.table': [table.weight]}
    def forbidden(*args, **kwargs):
        raise AssertionError('diagnostics must not densify sparse operator gradients')
    monkeypatch.setattr(torch.Tensor, 'to_dense', forbidden)
    collected = {}
    for ids in ([3, 42], [42, 9000]):
        diagnostics.accumulate_objective_gradients(collected,
            {'reconstruction': table(torch.tensor(ids)).sum()}, groups)
    report = diagnostics.objective_agreement({'output': -table(torch.tensor([42])).sum()},
        groups, accumulated=collected)['operator.table']
    assert report['reconstruction_norm'] == pytest.approx(math.sqrt(12.))
    assert report['reconstruction_output_cosine'] == pytest.approx(-math.sqrt(2 / 3))
    assert table.weight.grad is None


def test_read_only_perception_pullback_matches_both_real_updates():
    from SentenceCompose import fork_perception, saved_sentence_values
    p = torch.nn.Parameter(torch.tensor([2., 3.]))
    optimizer = torch.optim.SGD([p], lr=.01)
    original = p.detach().clone()
    with saved_sentence_values((p,)):
        cache = [p.square()]
        for weight in (1., 3.):
            trial, pullback = fork_perception(cache)
            cost = weight * trial[0].sum() + p.square().sum()
            p.grad = torch.tensor([11., 13.])
            measured, = pullback.gradients(cost, (p,))
            expected = 2 * weight * original + 2 * p.detach()
            torch.testing.assert_close(measured, expected)
            torch.testing.assert_close(p.grad, torch.tensor([11., 13.]))
            assert trial[0].grad is None
            optimizer.zero_grad()
            cost.backward()
            pullback()
            torch.testing.assert_close(p.grad, measured)
            optimizer.step()
