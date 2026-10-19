"""Two pre-update cost previews keep separate prediction graphs."""
import torch


def test_sentence_prediction_trials_backpropagate_independently():
    from Layers import SymbolExpectation
    layer = SymbolExpectation(n_symbols=4, max_depth=8, n_dim=4,
        concept_dim=4, batch=1)
    layer.set_symbol_bank(torch.eye(layer.concept_dim))
    layer.set_inter_loss_weight(1.)
    payloads = [torch.ones(3, 4)]
    mask = torch.ones(1, dtype=torch.bool)
    layer.observe_stm_end_state([3], payloads, train_prediction=False)
    layer.expect_next_meaning(0, refresh=True)
    pending = layer._inter_last_meaning[0]
    size = len(layer._inter_context[0])
    first, prior_a = layer.sentence_prediction_cost([3], payloads, mask)
    second, prior_b = layer.sentence_prediction_cost([3], payloads, mask)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert prior_a[0] is prior_b[0] is pending
    assert layer._inter_last_meaning[0] is pending
    assert len(layer._inter_context[0]) == size
    parameters = tuple(layer._inter_predictor.parameters())
    gradients_a = torch.autograd.grad(first.sum(), parameters, allow_unused=True)
    gradients_b = torch.autograd.grad(second.sum(), parameters, allow_unused=True)
    assert any(g is not None and bool(g.abs().sum() > 0) for g in gradients_a)
    for first_grad, second_grad in zip(gradients_a, gradients_b):
        if first_grad is None:
            assert second_grad is None
        else:
            torch.testing.assert_close(first_grad, second_grad, rtol=0, atol=0)
