"""Reconstruction has precedence in choosing between the two trials."""
import torch
import pytest


def test_only_strictly_lower_reconstruction_wins_and_tie_keeps_greedy():
    from SentenceCompose import sentence_pair
    active = torch.tensor([True, True, True])
    r = [torch.tensor([1., 2., 3.]), torch.tensor([2., 1., 3.])]
    def compose(cache, prior):
        return torch.zeros(3) if prior is None else torch.ones(3)
    def score(path, alternative):
        return r[int(alternative)], path
    chosen, _, wins = sentence_pair(None, compose, score, lambda loss: None,
        active=active)
    assert wins.tolist() == [False, True, False]
    assert chosen.tolist() == [0., 1., 0.]










