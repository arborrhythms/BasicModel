"""Width guards for the modality re-architecture (Phase 1, Task 1.2 of
doc/plans/2026-06-03-modality-architecture-plan.md).

Re-guards the REVISED per-tier shapes (supersedes test_convergence_widths.py,
whose tier shapes were the SS-promotion convergence target):

  - CS (ConceptualSpace) is the muxed event carrier: ``where=2, when=2`` with
    a codebook on ``.event`` (``codebook_slot == 'event'``, ``muxed == True``)
    at the full muxed width. It must round-trip a ``.where`` position + a
    ``.when`` bracket and reconstruct from the codebook selection.
  - SS (SymbolicSpace) carries NEITHER where nor when: ``where=0, when=0`` with
    a codebook on ``.what`` (``codebook_slot == 'what'``, ``muxed == False``).
    It must round-trip content only.

SubSpaces are built by hand at the target widths (NOT via config), proving the
muxing machinery is width-agnostic, so the Phase-2 substrate flip (sourcing the
widths from ``canonical_shape``) is safe. Construction idiom mirrors
test_when_range_encoding.py + the real ``_build_*_basis`` wiring
(``SubSpace(object=cb)`` for the muxed event slot, ``what=cb`` for the unmuxed
content slot).
"""

import math, os, sys, unittest
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from Spaces import SubSpace, WhereEncoding, WhenRangeEncoding, Codebook


def _fresh_codebook(nInput, nVectors, nDim, customVQ=True):
    """Build a minimal codebook-bearing Basis at a chosen width.

    Mirrors the real construction (``Codebook()`` then ``create``). The
    cross-codebook .where slice registry was retired (Phase 4); codebook
    identity is the row index, so no per-test registry reset is needed.
    """
    cb = Codebook()
    cb.create(nInput=nInput, nVectors=nVectors, nDim=nDim, customVQ=customVQ)
    return cb


# ===========================================================================
# CS-shaped SubSpace: where=2, when=2, codebook muxed on .event
# ===========================================================================

class TestCSWidthGuard(unittest.TestCase):

    def _build_cs(self, nWhat=4, nWhere=2, nWhen=2, nVectors=5, customVQ=True):
        dim = nWhat + nWhere + nWhen
        whenEnc = WhenRangeEncoding(64, nWhen)
        whereEnc = WhereEncoding(64, nWhere, nWhen)
        # Muxed codebook: prototype width == muxedSize (full event vector).
        cb = _fresh_codebook(nInput=3, nVectors=nVectors, nDim=dim, customVQ=customVQ)
        cs = SubSpace(
            whereEncoding=whereEnc,
            whenEncoding=whenEnc,
            object=cb,                     # CS-style codebook on .event
            inputShape=[3, dim], outputShape=[3, dim],
        )
        return cs, whenEnc, whereEnc, dim, nWhat, nVectors

    def test_cs_shape_and_codebook_slot(self):
        cs, _whenEnc, _whereEnc, _dim, nWhat, nVectors = self._build_cs()
        self.assertEqual(cs.codebook_slot, 'event')
        self.assertTrue(cs.muxed)
        # Widths read from the encodings (not hardcoded literals).
        self.assertEqual(cs.nWhere, 2)
        self.assertEqual(cs.nWhen, 2)
        self.assertEqual(cs.nWhat, nWhat)
        self.assertEqual(cs.muxedSize, nWhat + 2 + 2)
        # Muxed codebook prototype spans the full event width.
        proto = cs.prototype()
        self.assertIsNotNone(proto)
        self.assertEqual(tuple(proto.shape), (nVectors, nWhat + 4))

    def test_cs_demux_round_trips_where_span_and_when_bracket(self):
        cs, whenEnc, whereEnc, dim, _nWhat, _nVectors = self._build_cs()
        B, V = 2, 3
        # Build a muxed event from real codebook rows, then stamp a fresh
        # .where position (p=3) and a present-perfect .when bracket (-1, 0).
        sel = torch.tensor([[0, 1, 2], [3, 4, 0]])
        event = cs.lookup(sel).clone()
        when_idx = whenEnc.resolve(dim)
        event[..., when_idx] = whenEnc.encode_range(-1.0, 0.0).expand(B, V, -1)
        where_idx = whereEnc.resolve(dim)
        event[..., where_idx] = whereEnc.encode(torch.tensor(3.0)).expand(B, V, -1)

        cleaned, space, time = cs.decode(event.clone())

        # (a) .when round-trips to (start, end) = (-1, 0).
        start, end = time
        self.assertTrue(torch.allclose(start, torch.full_like(start, -1.0), atol=1e-3),
                        f"when start did not round-trip: {start}")
        self.assertTrue(torch.allclose(end, torch.zeros_like(end), atol=1e-3),
                        f"when end did not round-trip: {end}")
        # (b) .where round-trips to the stamped position (3).
        self.assertTrue(torch.allclose(space, torch.full_like(space, 3.0), atol=1e-3),
                        f"where did not round-trip: {space}")
        # (c) reverse zeroed both the where and when slots.
        self.assertTrue(torch.allclose(cleaned[..., when_idx],
                                       torch.zeros_like(cleaned[..., when_idx]), atol=1e-6))
        self.assertTrue(torch.allclose(cleaned[..., where_idx],
                                       torch.zeros_like(cleaned[..., where_idx]), atol=1e-6))
        # Fail-loud: no NaN/Inf escaped the demux.
        self.assertTrue(torch.isfinite(cleaned).all())
        self.assertTrue(torch.isfinite(start).all() and torch.isfinite(end).all())
        self.assertTrue(torch.isfinite(space).all())

    def test_cs_event_muxing_coexists_and_reconstructs(self):
        cs, whenEnc, _whereEnc, dim, _nWhat, _nVectors = self._build_cs()
        B, V = 2, 3
        sel = torch.tensor([[0, 1, 2], [3, 4, 0]])
        event = cs.lookup(sel).clone()
        when_idx = whenEnc.resolve(dim)
        event[..., when_idx] = whenEnc.encode_range(-1.0, 0.0).expand(B, V, -1)

        # Store the muxed event (snaps through the codebook), reconstruct from
        # the selection. The muxed .event view must come back at the full width
        # with finite values -- the where(2)+when(2) tail coexists with the mux.
        cs.set_event(event)
        recon = cs.materialize(mode="event")
        self.assertIsNotNone(recon)
        self.assertEqual(tuple(recon.shape), (B, V, cs.muxedSize))
        self.assertEqual(recon.shape[-1], dim)
        self.assertTrue(torch.isfinite(recon).all())
        recon_default = cs.materialize()
        self.assertIsNotNone(recon_default)
        self.assertTrue(torch.isfinite(recon_default).all())


# ===========================================================================
# SS-shaped SubSpace: where=0, when=0, codebook on .what (content only)
# ===========================================================================

class TestSSWidthGuard(unittest.TestCase):

    def _build_ss(self, nWhat=4, nVectors=5):
        nWhere, nWhen = 0, 0
        dim = nWhat + nWhere + nWhen
        whenEnc = WhenRangeEncoding(64, nWhen)
        whereEnc = WhereEncoding(64, nWhere, nWhen)
        cb = _fresh_codebook(nInput=3, nVectors=nVectors, nDim=nWhat)
        ss = SubSpace(
            whereEncoding=whereEnc,
            whenEncoding=whenEnc,
            what=cb,                       # SS-style codebook on .what (unmuxed)
            inputShape=[3, dim], outputShape=[3, dim],
        )
        return ss, dim, nWhat, nVectors

    def test_ss_shape_and_codebook_slot(self):
        ss, _dim, nWhat, nVectors = self._build_ss()
        # Slot identity: codebook on .what, unmuxed; no where/when carriers.
        self.assertEqual(ss.codebook_slot, 'what')
        self.assertFalse(ss.muxed)
        self.assertEqual(ss.nWhere, 0)
        self.assertEqual(ss.nWhen, 0)
        self.assertEqual(ss.nWhat, nWhat)
        self.assertEqual(ss.muxedSize, nWhat)          # no where/when tail
        proto = ss.prototype()
        self.assertIsNotNone(proto)
        self.assertEqual(tuple(proto.shape), (nVectors, nWhat))

    def test_ss_content_round_trips(self):
        ss, _dim, nWhat, nVectors = self._build_ss()
        # Content reconstruction: a per-position selection gathers [.., nWhat]
        # prototype rows, matching the prototype matrix exactly.
        sel = torch.tensor([[0, 1, 2], [3, 4, 0]])
        rows = ss.lookup(sel)
        self.assertEqual(tuple(rows.shape), (2, 3, nWhat))
        self.assertTrue(torch.isfinite(rows).all())
        proto = ss.prototype()
        self.assertTrue(torch.allclose(rows[0, 0], proto[0]))
        self.assertTrue(torch.allclose(rows[1, 0], proto[3]))

    def test_ss_decode_is_content_passthrough(self):
        ss, dim, nWhat, _nVectors = self._build_ss()
        B, V = 2, 3
        # With no where/when slots there is nothing to demux: decode must leave
        # the content untouched and return finite tensors.
        content = ss.lookup(torch.tensor([[0, 1, 2], [3, 4, 0]])).clone()
        self.assertEqual(content.shape[-1], nWhat)     # == muxedSize == dim
        cleaned, _space, _time = ss.decode(content.clone())
        self.assertTrue(torch.allclose(cleaned, content, atol=1e-6),
                        "SS decode must be a content pass-through (no where/when)")
        self.assertTrue(torch.isfinite(cleaned).all())


if __name__ == "__main__":
    unittest.main()
