"""Live attention validates its floor and preserves meaning lanes."""
import pytest
import torch

from AttentionObjective import validate_floor, tolerate_heterogeneity


@pytest.mark.parametrize('floor', [0., 1., -1., float('nan')])
def test_floor_must_be_strictly_between_zero_and_one(floor):
    with pytest.raises(ValueError, match='attentionFloor'):
        validate_floor(floor)


def test_heterogeneity_retains_both_lanes_and_default_is_identity():
    lanes = torch.tensor([[[.8, .6], [0., 0.], [1., 0.], [0., 1.]]])
    torch.testing.assert_close(tolerate_heterogeneity(lanes, 1.), lanes, rtol=0, atol=0)
    torch.testing.assert_close(tolerate_heterogeneity(lanes, 0.),
        torch.tensor([[[.2, 0.], [0., 0.], [1., 0.], [0., 1.]]]))
    with pytest.raises(ValueError, match='hetTolerance'):
        tolerate_heterogeneity(lanes, 1.1)
