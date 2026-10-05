"""Inactive inference columns repeat one unchanged forecast."""
import torch


def test_inference_reuses_constant_padding_forecast_without_changing_results():
    from Layers import BracketExpectation
    owner = BracketExpectation(n_symbols=4, max_depth=4, n_dim=4, concept_dim=4, p=2, q=1)
    bank = torch.eye(4)[None].expand(2, -1, -1)
    words = torch.nn.functional.pad(bank[:, :2], (0, 0, 0, 30))
    active = torch.zeros(2, 32, dtype=torch.bool)
    active[:, :2] = True
    active[1, 0] = False
    targets = torch.full((2, 32), -1, dtype=torch.long)
    targets[:, :2] = torch.tensor([0, 1])
    args = ('word', words, bank, torch.ones(2, 4, dtype=torch.bool), targets)
    reference = owner.expect(*args, teacher_forcing=False, active=active)
    calls = []
    hook = owner.word_predictor.register_forward_hook(lambda *args: calls.append(True))
    try:
        with torch.no_grad():
            actual = owner.expect(*args, teacher_forcing=False, active=active)
    finally:
        hook.remove()
    for before, after in zip(reference, actual):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert len(calls) == 3, 'inactive trailing columns recomputed an unchanged forecast'
