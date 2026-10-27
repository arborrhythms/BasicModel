"""The exported prediction uses its staged tensor and the ordinary head."""
import copy

import torch

from export_mlx import forward_core, _run_torch_export
from test_mlx_export import _staged_input


def test_exported_prediction_matches_both_staged_arguments():
    with torch.no_grad():
        model, staged = _staged_input()
        changed = staged.clone()
        changed[:, 0, 0] += .5
        reference = copy.deepcopy(model)
        other_reference = copy.deepcopy(model)
        expected = forward_core(reference, staged).clone()
        different = forward_core(other_reference, changed).clone()
        exported = _run_torch_export(model, staged).module()
        actual = exported(staged)
        other = exported(changed)
        repeated = exported(staged)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(other, different)
    torch.testing.assert_close(repeated, expected)
    assert not torch.equal(actual, other), 'exported prediction ignored its staged argument'
