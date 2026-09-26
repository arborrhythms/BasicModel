"""Descriptive metric arithmetic; the three-seed experiment is archived."""
import torch


def test_categorical_discrimination_counts_each_pair_once():
    from CategoricalDiscrimination import categorical_discrimination
    readings = torch.tensor([[0.], [1.], [4.], [5.]])
    measured = categorical_discrimination(readings, [0, 0, 1, 1])
    assert measured == dict(cp=3., within=1., between=4., within_pairs=2, between_pairs=4)
