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
from Spaces import (WhenRangeEncoding, _WHEN_TENSE_DEFAULT, _WHEN_TENSE_STEP,
                    _WHEN_PERIOD)

_NWHAT, _NWHERE, _NWHEN = 4, 2, 2
_ENC = WhenRangeEncoding(_WHEN_PERIOD, _NWHEN)


def _event(what, where, when):
    """Pack a [1, 1, nWhat+nWhere+nWhen] muxed event."""
    return torch.cat([what.reshape(1, 1, -1),
                      where.reshape(1, 1, -1),
                      when.reshape(1, 1, -1)], dim=-1)


def _decode_when(ev):
    """Decode the trailing 2 .when columns to (t, D): absolute time, tense."""
    t, D = _ENC.decode(ev[..., -_NWHEN:].detach())
    return float(t.reshape(-1)[0]), float(D.reshape(-1)[0])


def _what(ev):   return ev[..., :_NWHAT]
def _where(ev):  return ev[..., _NWHAT:_NWHAT + _NWHERE]


class TestLiftLowerWhen(unittest.TestCase):

    def _point_event(self, t=0):
        _ENC.t = int(t)
        what = torch.randn(_NWHAT).tanh()
        where = torch.tensor([0.3, -0.4])
        when = _ENC.encode(t, D=_WHEN_TENSE_DEFAULT)    # present phasor at time t
        return _event(what, where, when)

    def test_lift_advances_when_tense_toward_future(self):
        # 2026-06-07 .when redesign: the magnitude is TENSE (not duration), so
        # LIFT advances the tense one step toward the future (D + step),
        # preserving the absolute time-angle.
        T = _WHEN_PERIOD // 8
        lift = LiftLayer(nInput=_NWHAT)
        ev = self._point_event(t=T)                     # present tense (D=0.5) at T
        out = lift.compose(ev, ev)
        self.assertEqual(out.shape[-1], _NWHAT + _NWHERE + _NWHEN)
        t_dec, D = _decode_when(out)
        self.assertAlmostEqual(D, _WHEN_TENSE_DEFAULT + _WHEN_TENSE_STEP, delta=1e-4,
                               msg=f"LIFT must advance tense by +step; got D={D}")
        self.assertGreater(D, _WHEN_TENSE_DEFAULT,
                           f"LIFT must move tense toward future; got D={D}")
        self.assertAlmostEqual(t_dec, float(T), delta=0.05,
                               msg=f"LIFT must preserve the time-angle; got t={t_dec}")
        self.assertTrue(torch.isfinite(out).all())

    def test_lower_inverts_lift_on_when(self):
        T = _WHEN_PERIOD // 8
        lift, lower = LiftLayer(nInput=_NWHAT), LowerLayer(nInput=_NWHAT)
        ev = self._point_event(t=T)
        lifted = lift.compose(ev, ev)                   # D 0.5 -> 0.6
        lowered = lower.compose(lifted, lifted)         # D 0.6 -> 0.5 (back)
        t_dec, D = _decode_when(lowered)
        # LOWER retreats the tense one step toward the past, returning to the
        # original present magnitude (~0.5) with the time-angle preserved.
        self.assertAlmostEqual(D, _WHEN_TENSE_DEFAULT, delta=1e-4,
                               msg=f"LOWER must invert LIFT's tense step; got D={D}")
        self.assertAlmostEqual(t_dec, float(T), delta=0.05,
                               msg=f"LOWER must preserve the time-angle; got t={t_dec}")
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
        when = _ENC.encode(0, D=_WHEN_TENSE_DEFAULT)    # present .when phasor
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
