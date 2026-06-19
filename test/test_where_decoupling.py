"""Phase 4 / Task 4.1: decouple .where from codebook row-indexing.

RECONCILIATION (Alec, 2026-06-04) with doc/plans/2026-05-28-where-keyed-
taxonomy.md:
  * That plan OWNS: the META taxonomy + LBG split + the WholeSpace position
    counter (``allocate_position`` + the explicit row<->position dicts) + the
    content-match (nearest-row) reverse decode. Those are row/position-based,
    NOT .where-quadrature-based, and are KEPT.
  * THIS task reverts the vestigial ".where indexing on the codebook": the
    cross-codebook ``WhereEncoding`` slice registry (``allocate_codebook_slice``
    / ``_codebook_registry`` / ``global_max_val`` / ``reset_codebook_registry``),
    the per-codebook ``where_offset`` global key, and the dead ``recover``
    (.where -> int inverse). Codebook identity is the row index (the ``_index``
    selection); .where keeps its input-offset + positional/spatial-extent roles
    (``WhereEncoding.forward``, period = architecture.nObjects). Consequence
    (accepted): CS->SS reverse is approximate (content match), not an exact
    .where inversion.
"""

import os, sys, unittest
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from Spaces import SubSpace, WhereEncoding, WhenRangeEncoding, Codebook


def _ws_subspace(nWhat=4, nVectors=5):
    cb = Codebook()
    cb.create(nInput=3, nVectors=nVectors, nDim=nWhat, customVQ=True)
    return SubSpace(
        whereEncoding=WhereEncoding(64, 0, 0),
        whenEncoding=WhenRangeEncoding(64, 0),
        what=cb, inputShape=[3, nWhat], outputShape=[3, nWhat]), cb, nVectors


class TestWhereDecoupling(unittest.TestCase):

    def test_codebook_identity_is_the_active_row_index(self):
        ws, _cb, nVectors = _ws_subspace()
        sel = torch.tensor([[0, 1, 2], [3, 4, 0]])
        rows = ws.lookup(sel)                      # identity via _index selection
        proto = ws.prototype()
        self.assertTrue(torch.allclose(rows[0, 0], proto[0]))
        self.assertTrue(torch.allclose(rows[1, 0], proto[3]))
        self.assertEqual(tuple(rows.shape), (2, 3, 4))

    def test_where_is_not_a_codebook_row_key(self):
        # The cross-codebook .where slice registry (role a) is retired:
        # codebooks are identified by their local row index, not a global
        # .where offset.
        self.assertFalse(hasattr(WhereEncoding, "allocate_codebook_slice"),
                         "WhereEncoding.allocate_codebook_slice (the .where-as-"
                         "codebook-key registry) must be retired")
        self.assertFalse(hasattr(WhereEncoding, "recover"),
                         "WhereEncoding.recover (the dead .where->int identity "
                         "inverse) must be retired")
        _ss, cb, _n = _ws_subspace()
        self.assertEqual(int(getattr(cb, "where_offset", 0)), 0,
                         "a Codebook must not own a global .where-space offset")

    def test_where_still_carries_spatial_extent(self):
        # Role (c) kept: a CS-shaped where=2 SubSpace round-trips a .where
        # position (spatial extent), independent of codebook identity.
        whereEnc = WhereEncoding(64, 2, 2)
        whenEnc = WhenRangeEncoding(64, 2)
        cb = Codebook(); cb.create(nInput=3, nVectors=5, nDim=8, customVQ=True)
        cs = SubSpace(whereEncoding=whereEnc, whenEncoding=whenEnc,
                      object=cb, inputShape=[3, 8], outputShape=[3, 8])
        ev = cs.lookup(torch.tensor([[0, 1, 2]])).clone()
        where_idx = whereEnc.resolve(8)
        ev[..., where_idx] = whereEnc.encode(torch.tensor(3.0)).expand(1, 3, -1)
        _cleaned, space, _time = cs.decode(ev.clone())
        self.assertTrue(torch.allclose(space, torch.full_like(space, 3.0), atol=1e-3),
                        ".where must still round-trip a spatial position")


if __name__ == "__main__":
    unittest.main()
