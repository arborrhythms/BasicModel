"""Live attention preserves meaning lanes."""
import pytest
import torch

from AttentionObjective import tolerate_heterogeneity


def test_heterogeneity_retains_both_lanes_and_default_is_identity():
    lanes = torch.tensor([[[.8, .6], [0., 0.], [1., 0.], [0., 1.]]])
    torch.testing.assert_close(tolerate_heterogeneity(lanes, 1.), lanes, rtol=0, atol=0)
    torch.testing.assert_close(tolerate_heterogeneity(lanes, 0.),
        torch.tensor([[[.2, 0.], [0., 0.], [1., 0.], [0., 1.]]]))
    with pytest.raises(ValueError, match='hetTolerance'):
        tolerate_heterogeneity(lanes, 1.1)
