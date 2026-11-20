"""Real phase boundaries for numeric configurations."""
from pathlib import Path
import pytest
import torch


def test_numeric_understanding_owns_evidence_across_next_forward():
    from recon_bench import _build_model
    model, *_ = _build_model('data/MM_xor.xml')
    values = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]).unsqueeze(-1)
    first = model.understand(values)
    model.forward(values * 2.)
    restored, _ = model.reverseReconstruct(first)
    torch.testing.assert_close(restored, values)
