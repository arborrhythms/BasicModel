"""Phase 7 / Task 7.2: MorphologyLayer (lemma + feature routing over the
converged substrate). doc/plans/2026-06-03-modality-architecture-plan.md.

MorphologyLayer is a parameter-free unary CS-space_role GrammarLayer that decomposes a
surface token (via surface_morphology.analyze) and routes the tense/aspect
features onto the event .when by DELEGATING to TenseLayer / AspectLayer (the
.when math is reused, not re-derived). The "morphology" rule loads from
complete.grammar in both directions.
"""

import math, os, sys
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from Language import (MorphologyLayer, TenseLayer, AspectLayer, Grammar,
                      GRAMMAR_LAYER_CLASSES)
from Spaces import WhenRangeEncoding, _WHEN_PERIOD

_NWHAT, _NWHERE, _NWHEN = 4, 2, 2
_ENC = WhenRangeEncoding(_WHEN_PERIOD, _NWHEN)


def _event(t=0):
    # Present .when instant at absolute time t (2026-06-16 bracket redesign).
    _ENC.t = t
    what = torch.randn(_NWHAT).tanh()
    where = torch.tensor([0.3, -0.4])
    when = _ENC.encode(t)
    return torch.cat([what, where, when]).reshape(1, 1, -1)


def test_layer_identity_and_registration():
    m = MorphologyLayer()                       # parameter-free
    assert m.rule_name == "morphology"
    assert m.space_role == 'CS'
    assert int(m.arity) == 1
    assert GRAMMAR_LAYER_CLASSES["morphology"] is MorphologyLayer


def test_no_token_passthrough():
    m = MorphologyLayer()                       # no token set
    ev = _event()
    assert torch.allclose(m.compose(ev), ev)


def test_routes_past_tense_via_tenselayer():
    ev = _event(0.0)
    m = MorphologyLayer(); m.set_token("ran")   # -> ("run", {tense:PAST, aspect:[]})
    got = m.compose(ev)
    # Delegation: identical to TenseLayer(PAST) applied directly.
    ref = TenseLayer(); ref.set_op("PAST")
    assert torch.allclose(got, ref.compose(ev), atol=1e-6), \
        "morphology must delegate the PAST .when rotation to TenseLayer"


def test_routes_progressive_aspect_via_aspectlayer():
    ev = _event(0)
    m = MorphologyLayer(); m.set_token("running")  # -> ("run", {PRESENT, [PROGRESSIVE]})
    got = m.compose(ev)
    # PRESENT tense is identity; AspectLayer is RETIRED to a no-op (2026-06-07
    # .when redesign -- duration/aspect gone). Delegation still holds: the
    # result equals AspectLayer(PROGRESSIVE).compose(ev), which is now identity.
    ref = AspectLayer(); ref.set_op("PROGRESSIVE")
    assert torch.allclose(got, ref.compose(ev), atol=1e-6), \
        "morphology must delegate the (now no-op) PROGRESSIVE aspect to AspectLayer"
    # PRESENT + no-op aspect leaves the present .when unchanged (instant at t=0).
    assert torch.allclose(got[..., -_NWHEN:], ev[..., -_NWHEN:], atol=1e-6)
    center, ext = _ENC.decode(got[..., -_NWHEN:].detach())
    assert math.isclose(float(center.reshape(-1)[0]), 0.0, abs_tol=0.05)
    assert math.isclose(float(ext.reshape(-1)[0]), 0.0, abs_tol=1e-3)


def test_reverse_runs_and_recovers_tense():
    ev = _event(0)
    m = MorphologyLayer(); m.set_token("ran")
    back = m.generate(m.compose(ev))
    # Tense reverse is an exact center-shift back; .when round-trips.
    assert torch.isfinite(back).all()
    assert torch.allclose(back[..., -_NWHEN:], ev[..., -_NWHEN:], atol=1e-4)


def test_morphology_rule_loads_in_both_directions():
    g = Grammar()
    g.load_from_grammar_file("complete.grammar")
    methods = {r.method_name for r in g.rules if r.method_name}
    assert "morphology" in methods, "morphology rule missing from grammar"
    up = {r.method_name for r in g.rules_upward if r.method_name}
    dn = {r.method_name for r in g.rules_downward if r.method_name}
    assert "morphology" in up and "morphology" in dn, \
        "morphology must have both a compose (upward) and generate (downward) rule"


if __name__ == "__main__":
    import unittest
    unittest.main()
