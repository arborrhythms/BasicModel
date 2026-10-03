"""Uninformed baselines preserve solved errors and keep penalties separate."""
import math
import pytest
import torch
from Layers import Error


def test_squared_error_is_ratio_of_means_and_baseline_is_detached():
    pred = torch.tensor([2., 2.], requires_grad=True)
    target = torch.tensor([1., 10.], requires_grad=True)
    costs = Error()
    costs.squared('answer', pred, target)
    result = costs.total()
    torch.testing.assert_close(result, torch.tensor(65. / 101.))
    gp, gt = torch.autograd.grad(result, (pred, target))
    torch.testing.assert_close(gp, 2 * (pred-target) / 101)
    torch.testing.assert_close(gt, -gp)
    assert costs.breakdown()['answer']['baseline'] == 50.5


@pytest.mark.parametrize('outcomes', [2, 7, 256])
def test_uniform_categorical_prediction_is_one_even_for_constant_targets(outcomes):
    costs = Error()
    logits = torch.zeros(3, outcomes, requires_grad=True)
    costs.categorical('category', logits, torch.zeros(3, dtype=torch.long))
    torch.testing.assert_close(costs.total(), torch.tensor(1.))
    assert costs.breakdown()['category']['baseline'] == pytest.approx(math.log(outcomes))


def test_binary_zero_logits_have_one_error_for_constant_targets():
    costs = Error()
    costs.binary('kind', torch.zeros(5), torch.zeros(5))
    torch.testing.assert_close(costs.total(), torch.tensor(1.))


def test_zero_targets_are_an_unnormalized_penalty_and_empty_mask_is_inactive():
    costs = Error()
    costs.squared('zero_target', torch.tensor([2., 4.]), torch.zeros(2), weight=.25)
    costs.squared('absent', torch.tensor([99.]), torch.tensor([99.]), mask=torch.tensor([False]))
    torch.testing.assert_close(costs.total(), torch.tensor(2.5))
    torch.testing.assert_close(costs.total(kind='penalty'), torch.tensor(2.5))
    torch.testing.assert_close(costs.total(kind='relative'), torch.tensor(0.))
    assert costs.breakdown()['zero_target']['kind'] == 'penalty'
    assert costs.breakdown()['absent']['active_entries'] == 0


def test_repeated_contributions_share_one_target_baseline():
    costs = Error()
    costs.squared('code', torch.tensor([2.]), torch.tensor([1.]))
    costs.squared('code', torch.tensor([2.]), torch.tensor([10.]))
    torch.testing.assert_close(costs.total(), torch.tensor(65./101.))


def test_row_costs_train_one_ratio_without_amplifying_small_target_rows():
    active = torch.tensor([True, True, False])
    costs = Error(row_mask=active)
    costs.squared('answer', torch.tensor([[2.], [2.], [99.]]),
                  torch.tensor([[1.], [10.], [99.]]))
    rows = costs.total()
    torch.testing.assert_close(rows, torch.tensor([2./101., 128./101., 0.]))
    torch.testing.assert_close(rows[active].mean(), torch.tensor(65./101.))


def test_single_row_registry_preserves_selection_axis():
    costs = Error(row_mask=[True])
    costs.binary('kind', torch.tensor(0.), torch.tensor(1.), row=0)
    torch.testing.assert_close(costs.total(), torch.ones(1))


def test_nearly_solved_error_stays_small_and_regularizer_keeps_strength():
    costs = Error()
    costs.squared('solved', torch.tensor([1.001]), torch.tensor([1.]))
    costs.add('sparsity', torch.tensor(2.), weight=.03, kind='penalty')
    assert float(costs.total(kind='relative')) < .000002
    torch.testing.assert_close(costs.total(kind='penalty'), torch.tensor(.06))


def test_reporting_only_penalty_does_not_enter_trained_total():
    costs = Error()
    costs.squared('fit', torch.tensor([0.]), torch.tensor([1.]))
    costs.add('proximal_l1', torch.tensor(5.), kind='penalty', trained=False)
    torch.testing.assert_close(costs.total(), torch.tensor(1.))
    assert costs.breakdown()['proximal_l1']['weighted'] == 5


def test_row_receipt_reports_the_trained_mean_and_contextual_weight():
    costs = Error(row_mask=torch.tensor([True, True, False]))
    costs.squared('fit', torch.tensor([2., 2., 99.]), torch.tensor([1., 10., 99.]), weight=2.)
    costs.scale(torch.tensor(3.))
    receipt = costs.breakdown()['fit']
    assert receipt['value'] == pytest.approx(65./101.)
    assert receipt['weighted'] == pytest.approx(6.*65./101.)
    assert receipt['row_values'] == pytest.approx([2./101., 128./101., 0.])


def test_scalar_receipt_and_history_include_contextual_weight():
    costs = Error()
    costs.squared('fit', torch.tensor([0.]), torch.tensor([1.]), weight=2.)
    costs.scale(torch.tensor(3.))
    assert costs.breakdown()['fit']['weighted'] == 6.
    costs.snapshot()
    assert costs._history[-1]['fit'] == 6.


def test_zero_weight_does_not_read_a_missing_registry():
    errors = Error()
    errors.merge(object(), weight=0.)
    assert errors.total() is None
    assert errors.breakdown() == {}
