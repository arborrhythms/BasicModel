"""Regression inputs captured at the byte scorer in the rejected receipt.

They preserve the observed column-placement failure; no random seed or model
initialization is used. The full native packing assertion remains unchanged.
"""
from pathlib import Path
from types import SimpleNamespace

import torch


def test_byte_scoring_ignores_empty_candidate_columns():
    from Models import BasicModel
    fixture = torch.load(Path(__file__).with_name('fixtures') / 'item7_byte_columns.pt',
                         weights_only=True)
    owner = SimpleNamespace(_BYTE_ASSIGNMENT_TAU=BasicModel._BYTE_ASSIGNMENT_TAU)
    values, gradients = [], []
    for layout in ('packed', 'single'):
        inputs = fixture[layout]
        inputs['idea'].requires_grad_(True)
        cost = BasicModel._byte_word_cost(owner, **inputs)
        gradient, = torch.autograd.grad(cost[1], inputs['idea'])
        values.append(cost[1])
        gradients.append(gradient[1])
    torch.testing.assert_close(values[0], values[1], atol=0, rtol=0)
    torch.testing.assert_close(gradients[0], gradients[1], atol=0, rtol=0)
