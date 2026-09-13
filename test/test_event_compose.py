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
from Spaces import event_when_encoding, _WHEN_TENSE_STEP, _WHEN_PERIOD

# 2026-07-04 encoding pass: .when is the 4-dim start ladder; built through
# the one construction seam so LiftLayer/tense share omega pairs.
_NWHAT, _NWHERE, _NWHEN = 4, 2, 4
_ENC = event_when_encoding(_NWHEN)


def _event(what, where, when):
    """Pack a [1, 1, nWhat+nWhere+nWhen] muxed event."""
    return torch.cat([what.reshape(1, 1, -1),
                      where.reshape(1, 1, -1),
                      when.reshape(1, 1, -1)], dim=-1)


def _decode_when(ev):
    """Decode the trailing .when columns to (start, residue): the onset and
    the ladder decode-health residue (~0 for a clean stamp)."""
    s, res = _ENC.decode(ev[..., -_NWHEN:].detach())
    return float(s.reshape(-1)[0]), float(res.reshape(-1)[0])


def _what(ev):   return ev[..., :_NWHAT]
def _where(ev):  return ev[..., _NWHAT:_NWHAT + _NWHERE]


class TestLiftLowerWhen(unittest.TestCase):

    def _point_event(self, t=0):
        _ENC.t = int(t)
        what = torch.randn(_NWHAT).tanh()
        where = torch.tensor([0.3, -0.4])
        when = _ENC.encode(t)                            # present instant at time t
        return _event(what, where, when)

    def test_lift_and_lower_treat_the_event_as_opaque(self):
        # Concepts are full-width codes with no separable .where/.when
        # (Alec, 2026-09-13): sized to the muxed width, LIFT and LOWER fold
        # the whole event through their inner layer; nothing is copied
        # through or shifted around the fold.
        width = _NWHAT + _NWHERE + _NWHEN
        lift, lower = LiftLayer(nInput=width), LowerLayer(nInput=width)
        T = _WHEN_PERIOD // 8
        ev = self._point_event(t=T)
        out = lift.compose(ev, ev)
        self.assertEqual(out.shape[-1], width)
        self.assertTrue(torch.allclose(out, lift._sigma.compose(ev, ev)))
        low = lower.compose(ev, ev)
        self.assertTrue(torch.allclose(low, lower._pi.compose(ev, ev)))
        self.assertTrue(torch.isfinite(out).all() and torch.isfinite(low).all())

    def test_content_only_operand_passes_through_legacy_fold(self):
        # No where/when tail: width == nInput -> legacy binary sigma fold,
        # output is content-width (unchanged contract for the SS-space_role route).
        lift = LiftLayer(nInput=_NWHAT)
        a = torch.randn(1, 1, _NWHAT).tanh()
        out = lift.compose(a, a)
        self.assertEqual(out.shape[-1], _NWHAT)
        self.assertTrue(torch.isfinite(out).all())


class TestPrepositionWhere(unittest.TestCase):

    def test_preposition_passes_the_phrase_through_opaquely(self):
        # The event is opaque to the grammar ops: PREPOSITION absorbs the
        # marker and passes the phrase through unchanged (its .where
        # rotation lived on a split of the event and is gone).
        prep = PrepositionLayer(nInput=_NWHAT + _NWHERE + _NWHEN)
        when = _ENC.encode(0)
        P = _event(torch.randn(_NWHAT).tanh(), torch.tensor([0.1, 0.1]), when)
        X = _event(torch.randn(_NWHAT).tanh(), torch.tensor([0.5, -0.2]), when)
        out = prep.compose(P, X)
        self.assertTrue(torch.equal(out, X))
        left, right = prep.reverse(out)
        self.assertTrue(torch.equal(left, X) and torch.equal(right, X))

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
