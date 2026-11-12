"""The owner-step total chooses between the two trials."""
import torch
import pytest


def test_only_strictly_lower_total_wins_and_tie_keeps_greedy():
    from SentenceCredit import comparison
    active = torch.tensor([True, True, True])
    costs = torch.tensor([[1., 2.], [2., 1.], [3., 3.]])
    parts = torch.stack((costs, torch.zeros_like(costs), torch.zeros_like(costs)), -1)
    audit = comparison(parts, active)
    wins, chosen = audit['wins'], audit['winner'].float()
    assert wins.tolist() == [False, True, False]
    assert chosen.tolist() == [0., 1., 0.]










