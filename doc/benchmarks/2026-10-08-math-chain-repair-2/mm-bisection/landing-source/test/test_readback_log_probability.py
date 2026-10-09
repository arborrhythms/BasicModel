"""Wrong candidate winners retain finite byte loss and reconstruction credit."""
from types import SimpleNamespace

import torch


def test_wrong_winner_keeps_leaf_gradient_with_detached_competitor_codes():
    from Models import BasicModel
    from SentenceUnderstanding import readback_scores
    leaf = torch.tensor([[.2, 1.]], requires_grad=True)
    codes = torch.eye(2).unsqueeze(0).requires_grad_()
    priming = torch.full((1, 2), 20.)
    surfaces = torch.tensor([[[97], [98]]])
    target = torch.tensor([[[97]]])
    owner = SimpleNamespace(_BYTE_ASSIGNMENT_TAU=BasicModel._BYTE_ASSIGNMENT_TAU)
    logits = readback_scores(leaf, codes, priming) / owner._BYTE_ASSIGNMENT_TAU
    normalizer = torch.logsumexp(torch.cat((logits, torch.zeros(1, 1)), -1), -1)
    byte = torch.logsumexp(torch.stack((logits[:, 0], logits.new_full((1,), -torch.log(torch.tensor(256.)).item())), -1), -1)
    end = torch.logsumexp(torch.cat((logits, logits.new_full((1, 1), -torch.log(torch.tensor(256.)).item())), -1), -1)
    expected = normalizer - (byte + end) / 2
    actual = BasicModel._byte_word_cost(owner, leaf, torch.tensor(0), codes,
        surfaces, torch.ones_like(surfaces, dtype=torch.bool), target,
        torch.ones_like(target, dtype=torch.bool), True, priming=priming)
    torch.testing.assert_close(actual, expected)
    d_leaf, d_codes = torch.autograd.grad(actual.sum(), (leaf, codes), allow_unused=True)
    assert torch.isfinite(d_leaf).all() and d_leaf.norm() > 0
    assert d_codes is None
