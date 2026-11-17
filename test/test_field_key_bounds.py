"""A field centroid stays in its code coordinates' convex hull."""
from types import SimpleNamespace
import torch
from FieldProjection import key
from Understanding import ConceptualField


def test_identical_field_codes_keep_their_exact_boundary_after_accumulation():
    codes = torch.ones(160, 128)
    owner = SimpleNamespace(_csw_row_of=lambda cid: cid, _row_order=lambda row: 0,
        similarity_codebook=SimpleNamespace(lookup_rows=lambda rows: codes[rows]))
    weights = torch.full((160, 1, 1, 2), .1, requires_grad=True)
    field = ConceptualField(torch.arange(160), weights,
        torch.tensor([[[0, 1]]]), torch.tensor([[[0, 1]]]), None)
    point = key(owner, field, order=0)
    torch.testing.assert_close(point, torch.ones(1, 1, 128), atol=0, rtol=0)
    point.sum().backward()
    assert torch.isfinite(weights.grad).all()
    empty = ConceptualField(field.concept_ids, weights * 0, field.where, field.when, None)
    assert torch.equal(key(owner, empty, order=0), torch.zeros_like(point))
