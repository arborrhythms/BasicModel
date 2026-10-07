"""Direct Codebook readers obey logical occupancy, not physical capacity."""

from __future__ import annotations

import os
import sys

import torch

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Spaces import Codebook


def _codebook(rows, active):
    rows = torch.as_tensor(rows, dtype=torch.float32)
    cb = Codebook()
    cb.create(
        nInput=1,
        nVectors=int(rows.shape[0]),
        nDim=int(rows.shape[1]),
        customVQ=True,
        monotonic=False,
    )
    with torch.no_grad():
        cb.W.copy_(rows)
        cb.vq.embed_avg.copy_(rows)
        cb.vq._b_norms_sq.copy_((rows * rows).sum(dim=-1))
    cb.vq.set_active_rows(active)
    return cb




def test_active_prototypes_and_reverse_ignore_inactive_exact_match():
    cb = _codebook(
        [[0.0, 0.0], [0.5, 0.0], [-0.5, 0.0], [0.8, 0.8]],
        active=2,
    )
    query = torch.tensor([[[0.8, 0.8]]])

    assert cb.active_row_count() == 2
    assert cb.active_prototypes().shape == (2, 2)
    assert cb.active_prototypes().data_ptr() == cb.getW().data_ptr()

    snapped = cb.reverse(query)
    assert any(torch.equal(snapped[0, 0], row) for row in cb.getW()[:2])
    assert not torch.equal(snapped[0, 0], cb.getW()[3])


def test_standalone_default_remains_all_active():
    cb = _codebook([[0.0, 0.0], [0.25, 0.5], [0.8, 0.8]], active=3)
    query = torch.tensor([[[0.8, 0.8]]])

    assert cb.active_row_count() == 3
    assert torch.equal(cb.reverse(query)[0, 0], cb.getW()[2])






def test_membership_read_is_independent_of_active_and_inactive_codes():
    from test_cs_sparse_weights import _cs
    from test_concept_memberships import binary_features
    cs = _cs(nS=16, order=1)
    native, extents = binary_features(cs, torch.tensor([[48, 49]]))
    before = cs.cs_read_memberships(native, extents)
    with torch.no_grad():
        cs.similarity_codebook.getW().normal_()
    after = cs.cs_read_memberships(native, extents)
    torch.testing.assert_close(after, before, atol=0, rtol=0)
