"""Phase 3 (grammar ops operate event->event) of
doc/plans/2026-06-03-modality-architecture-plan.md.

C-space_role grammar ops see the muxed event [what | where | when]:
  - LIFT composes the .what content (binary sigma fold) AND extends the
    result's .when span > 1, advancing the center (verb-advances-future).
  - LOWER is the inverse: pi fold over content, retract the .when span back
    toward a unit point with the center retreated.
  - PREPOSITION modifies the .where block, leaving .what / .when untouched.
Content-only operands (no where/when tail) pass through the legacy fold
unchanged (SS-space_role route stays content-only).
"""

import math, os, sys, unittest
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from Language import LiftLayer, LowerLayer, PrepositionLayer
from Spaces import WhenRangeEncoding, _WHEN_TENSE_STEP, _WHEN_PERIOD

_NWHAT, _NWHERE, _NWHEN = 4, 2, 2
_ENC = WhenRangeEncoding(_WHEN_PERIOD, _NWHEN)


def _event(what, where, when):
    """Pack a [1, 1, nWhat+nWhere+nWhen] muxed event."""
    return torch.cat([what.reshape(1, 1, -1),
                      where.reshape(1, 1, -1),
                      when.reshape(1, 1, -1)], dim=-1)


def _decode_when(ev):
    """Decode the trailing 2 .when columns to (center, extent): event-time
    center and duration (2026-06-16 .when bracket redesign)."""
    c, ext = _ENC.decode(ev[..., -_NWHEN:].detach())
    return float(c.reshape(-1)[0]), float(ext.reshape(-1)[0])


def _what(ev):   return ev[..., :_NWHAT]
def _where(ev):  return ev[..., _NWHAT:_NWHAT + _NWHERE]


class TestLiftLowerWhen(unittest.TestCase):

    def _point_event(self, t=0):
        _ENC.t = int(t)
        what = torch.randn(_NWHAT).tanh()
        where = torch.tensor([0.3, -0.4])
        when = _ENC.encode(t)                            # present instant at time t
        return _event(what, where, when)

    def test_lift_advances_when_toward_future(self):
        # 2026-06-16 .when bracket redesign: tense is the interval-vs-now
        # relation, so LIFT advances the event-time CENTER one step toward the
        # future (+step ticks), preserving the event duration (0 for an instant).
        T = _WHEN_PERIOD // 8
        lift = LiftLayer(nInput=_NWHAT)
        ev = self._point_event(t=T)                     # present instant at T
        out = lift.compose(ev, ev)
        self.assertEqual(out.shape[-1], _NWHAT + _NWHERE + _NWHEN)
        center, ext = _decode_when(out)
        self.assertAlmostEqual(center, float(T) + _WHEN_TENSE_STEP, delta=0.05,
                               msg=f"LIFT must advance time by +step; got center={center}")
        self.assertGreater(center, float(T),
                           f"LIFT must move the event toward future; got center={center}")
        self.assertAlmostEqual(ext, 0.0, delta=1e-3,
                               msg=f"LIFT must preserve the (zero) duration; got ext={ext}")
        self.assertTrue(torch.isfinite(out).all())

    def test_lower_inverts_lift_on_when(self):
        T = _WHEN_PERIOD // 8
        lift, lower = LiftLayer(nInput=_NWHAT), LowerLayer(nInput=_NWHAT)
        ev = self._point_event(t=T)
        lifted = lift.compose(ev, ev)                   # center T -> T+step
        lowered = lower.compose(lifted, lifted)         # center T+step -> T (back)
        center, ext = _decode_when(lowered)
        # LOWER retreats the event one step toward the past, returning to the
        # original event time T, duration preserved.
        self.assertAlmostEqual(center, float(T), delta=0.05,
                               msg=f"LOWER must invert LIFT's time step; got center={center}")
        self.assertAlmostEqual(ext, 0.0, delta=1e-3,
                               msg=f"LOWER must preserve the (zero) duration; got ext={ext}")
        self.assertTrue(torch.isfinite(lowered).all())

    def test_content_only_operand_passes_through_legacy_fold(self):
        # No where/when tail: width == nInput -> legacy binary sigma fold,
        # output is content-width (unchanged contract for the SS-space_role route).
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
        when = _ENC.encode(0)                           # present .when instant
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
