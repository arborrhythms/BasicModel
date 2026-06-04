"""Phase 3 (grammar ops operate event->event) of
doc/plans/2026-06-03-modality-architecture-plan.md.

C-tier grammar ops see the muxed event [what | where | when]:
  - LIFT composes the .what content (binary sigma fold) AND extends the
    result's .when span > 1, advancing the center (verb-advances-future).
  - LOWER is the inverse: pi fold over content, retract the .when span back
    toward a unit point with the center retreated.
  - PREPOSITION modifies the .where block, leaving .what / .when untouched.
Content-only operands (no where/when tail) pass through the legacy fold
unchanged (SS-tier route stays content-only).
"""

import math, os, sys, unittest
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from Language import LiftLayer, LowerLayer, PrepositionLayer
from Spaces import WhenRangeEncoding

_NWHAT, _NWHERE, _NWHEN = 4, 2, 2
_ENC = WhenRangeEncoding(64, _NWHEN)


def _event(what, where, when):
    """Pack a [1, 1, nWhat+nWhere+nWhen] muxed event."""
    return torch.cat([what.reshape(1, 1, -1),
                      where.reshape(1, 1, -1),
                      when.reshape(1, 1, -1)], dim=-1)


def _decode_when(ev):
    s, e = _ENC.decode(ev[..., -_NWHEN:].detach())
    return float(s.reshape(-1)[0]), float(e.reshape(-1)[0])


def _what(ev):   return ev[..., :_NWHAT]
def _where(ev):  return ev[..., _NWHAT:_NWHAT + _NWHERE]


class TestLiftLowerWhen(unittest.TestCase):

    def _point_event(self, t=0.0):
        what = torch.randn(_NWHAT).tanh()
        where = torch.tensor([0.3, -0.4])
        when = _ENC.encode_range(t - 0.5, t + 0.5)      # unit point at t
        return _event(what, where, when)

    def test_lift_extends_when_span_and_advances_center(self):
        lift = LiftLayer(nInput=_NWHAT)
        ev = self._point_event(t=0.0)                   # center 0, span 1
        out = lift.compose(ev, ev)
        self.assertEqual(out.shape[-1], _NWHAT + _NWHERE + _NWHEN)
        s, e = _decode_when(out)
        self.assertGreater(e - s, 1.0, f"LIFT must extend .when span; got ({s},{e})")
        self.assertGreater((s + e) / 2.0, 0.0,
                           f"LIFT must advance the .when center; got ({s},{e})")
        self.assertTrue(torch.isfinite(out).all())

    def test_lower_inverts_lift_on_when(self):
        lift, lower = LiftLayer(nInput=_NWHAT), LowerLayer(nInput=_NWHAT)
        ev = self._point_event(t=0.0)
        lifted = lift.compose(ev, ev)
        lowered = lower.compose(lifted, lifted)
        s, e = _decode_when(lowered)
        # LOWER collapses the span back toward a unit point (span ~1) with the
        # center retreated to the original (~0).
        self.assertLess(e - s, 1.5, f"LOWER must retract the span; got ({s},{e})")
        self.assertAlmostEqual((s + e) / 2.0, 0.0, delta=0.2,
                               msg=f"LOWER must retreat the center; got ({s},{e})")
        self.assertTrue(torch.isfinite(lowered).all())

    def test_content_only_operand_passes_through_legacy_fold(self):
        # No where/when tail: width == nInput -> legacy binary sigma fold,
        # output is content-width (unchanged contract for the SS-tier route).
        lift = LiftLayer(nInput=_NWHAT)
        a = torch.randn(1, 1, _NWHAT).tanh()
        out = lift.compose(a, a)
        self.assertEqual(out.shape[-1], _NWHAT)
        self.assertTrue(torch.isfinite(out).all())


class TestPrepositionWhere(unittest.TestCase):

    def test_preposition_modifies_where_only(self):
        prep = PrepositionLayer(nInput=_NWHAT)
        what = torch.randn(_NWHAT).tanh()
        where = torch.tensor([0.5, -0.2])
        when = _ENC.encode_range(-0.5, 0.5)
        P = _event(torch.randn(_NWHAT).tanh(), torch.tensor([0.1, 0.1]), when)
        X = _event(what, where, when)
        out = prep.compose(P, X)
        # .what and .when of X are preserved; .where is modified.
        self.assertTrue(torch.allclose(_what(out), _what(X), atol=0.05),
                        ".what must be preserved")
        self.assertTrue(torch.allclose(out[..., -_NWHEN:], X[..., -_NWHEN:], atol=0.05),
                        ".when must be preserved")
        self.assertFalse(torch.allclose(_where(out), _where(X), atol=1e-4),
                         ".where must be modified by PREPOSITION")
        self.assertTrue(torch.isfinite(out).all())

    def test_preposition_content_only_passthrough(self):
        # Parameter-free construction (no content width) -> safe pass-through
        # of X (grammar's existing PREPOSITION contract).
        prep = PrepositionLayer()
        P = torch.randn(1, 1, _NWHAT).tanh()
        X = torch.randn(1, 1, _NWHAT).tanh()
        out = prep.compose(P, X)
        self.assertTrue(torch.allclose(out, X))


if __name__ == "__main__":
    unittest.main()
