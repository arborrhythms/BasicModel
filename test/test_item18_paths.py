"""Real phase boundaries for numeric configurations."""
from pathlib import Path
import pytest
import torch


def test_numeric_understanding_owns_evidence_across_next_forward():
    from recon_bench import _build_model
    model, *_ = _build_model('data/MM_xor.xml')
    values = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]).unsqueeze(-1)
    # An input is put back once its places are known: it is presented as
    # often as nominates a place's row.
    for _ in range(model._concept_owner().concept_mint_count):
        model.forward(values)
    first = model.understand(values)
    model.forward(values * .5)
    restored, _ = model.reverseReconstruct(first)
    torch.testing.assert_close(restored, values)
