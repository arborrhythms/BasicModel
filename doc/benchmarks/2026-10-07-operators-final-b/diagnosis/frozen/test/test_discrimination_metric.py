"""Descriptive metric arithmetic; the three-seed experiment is archived."""
import torch


def test_categorical_discrimination_counts_each_pair_once():
    from CategoricalDiscrimination import categorical_discrimination
    readings = torch.tensor([[0.], [1.], [4.], [5.]])
    measured = categorical_discrimination(readings, [0, 0, 1, 1])
    assert measured == dict(cp=3., within=1., between=4., within_pairs=2, between_pairs=4)


from functools import wraps

from types import SimpleNamespace

import pytest

import torch


def test_fixed_discrimination_metric_keeps_all_probes_without_training():
    from CategoricalDiscrimination import FIXED_PROBES, fixed_probe_discrimination
    assert {name: len(probes['texts']) for name, probes in FIXED_PROBES.items()} == {
        'xor': 4, 'fineweb': 68}
    readings = {name: torch.zeros(len(probes['texts']), 8, requires_grad=True)
                for name, probes in FIXED_PROBES.items()}
    result = fixed_probe_discrimination(readings)
    assert result['categorical_discrimination']['xor']['cp'] == 0.
    assert result['categorical_discrimination']['fineweb']['cp'] == 0.
    assert all(value.grad is None for value in readings.values())

