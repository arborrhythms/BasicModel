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


from functools import wraps

import pytest

import torch


def test_compiled_byte_reconstruction_handles_lookahead_and_nul():
    from types import SimpleNamespace
    from Models import BasicModel
    owner = SimpleNamespace(_BYTE_ASSIGNMENT_TAU=.1)
    idea = torch.arange(1., 129.).reshape(2, 64).requires_grad_()
    bank = torch.nn.functional.normalize(torch.arange(1., 2049.).reshape(2, 16, 64), dim=-1)
    tokens = torch.arange(2 * 16 * 9).reshape(2, 16, 9) % 256
    tokens[:, :, 7] = 0
    valid = torch.ones_like(tokens, dtype=torch.bool)
    target = tokens[:, :8, :8].clone()
    target_mask = torch.ones_like(target, dtype=torch.bool)
    def score(value):
        return BasicModel._byte_word_cost(owner, value, torch.tensor(0), bank,
            tokens, valid, target, target_mask, True)
    expected = score(idea)
    compiled = torch.compile(score, backend='inductor', fullgraph=True)
    actual = compiled(idea)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), idea)[0],
                               torch.autograd.grad(expected.sum(), idea)[0])

