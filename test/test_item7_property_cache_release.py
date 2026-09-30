"""Stage-zero diagnostics do not retain the previous sentence's property graph."""
import torch
from test_wholespace_property_migration import _small_property_model


def test_property_diagnostic_releases_its_graph_at_tick_end(tmp_path):
    model = _small_property_model(tmp_path)
    owner = model.wholeSpace
    owner._stage0_carrier(torch.tensor([[97, 98]]), None)
    assert owner._stage0_property_membership.grad_fn is not None
    model.post_tick_compact()
    assert owner._stage0_property_membership.grad_fn is None
