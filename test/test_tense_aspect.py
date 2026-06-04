"""Phase 4 (tense / aspect as ``.when`` operations).

doc/plans/2026-06-03-contextual-bind-preposition-when.md "Operation 3:
tense / aspect". Task 4.1 covers the pure surface normalizer
(``bin/surface_tense.py``); Tasks 4.2 cover the two unary C-tier ops
(``TenseLayer`` / ``AspectLayer``) that rewrite the ``.when`` tail of a
materialized muxed event. Hard rule: no global POS inventory.
"""

import math, os, sys, unittest
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from surface_tense import normalize_surface


# --- Task 4.1: pure table-driven surface normalizer (no torch) -------------
def test_ran():            assert normalize_surface(["ran"]) == ("PAST", [], "run")
def test_is_running():     assert normalize_surface(["is", "running"]) == ("PRESENT", ["PROGRESSIVE"], "run")
def test_has_run():        assert normalize_surface(["has", "run"]) == ("PRESENT", ["PERFECT"], "run")
def test_had_been_running(): assert normalize_surface(["had", "been", "running"]) == ("PAST", ["PERFECT", "PROGRESSIVE"], "run")
def test_did_run():        assert normalize_surface(["did", "run"]) == ("PAST", [], "run")
def test_will_run():       assert normalize_surface(["will", "run"]) == ("FUTURE", [], "run")  # MODAL hook noted


# --- Task 4.2: TenseLayer / AspectLayer unary .when ops --------------------
from Language import TenseLayer, AspectLayer
from Spaces import WhenRangeEncoding


def _event_with_present_when(B=2, V=3, nhead=6):       # nhead = nWhat+nWhere; +2 for .when
    enc = WhenRangeEncoding(64, 2)
    head = torch.randn(B, V, nhead)
    when = enc.encode_range(-0.5, 0.5).expand(B, V, -1)  # present default tail (unit bracket)
    return torch.cat([head, when], dim=-1), head, enc


def test_class_contracts():
    assert TenseLayer.tier == 'C' and TenseLayer.arity == 1
    assert AspectLayer.tier == 'C' and AspectLayer.arity == 1


def test_past_rotates_when_backward():
    t = TenseLayer(); t.set_op("PAST")
    x, _head, enc = _event_with_present_when()
    y = t.forward(x)
    ds, de = enc.decode(y[..., -2:])
    # PAST shifts the present bracket (-0.5, 0.5) back by 1 -> (-1.5, -0.5)
    assert math.isclose(float(ds.reshape(-1)[0]), -1.5, abs_tol=0.05)
    assert math.isclose(float(de.reshape(-1)[0]), -0.5, abs_tol=0.05)


def test_present_is_identity():
    t = TenseLayer(); t.set_op("PRESENT")
    x, _h, _e = _event_with_present_when()
    assert torch.allclose(t.forward(x), x, atol=1e-6)


def test_perfect_shapes_interval_around_end():
    a = AspectLayer(); a.set_op("PERFECT")
    x, _head, enc = _event_with_present_when()
    y = a.forward(x)
    ds, de = enc.decode(y[..., -2:])                    # present r=0 -> (-1, 0)
    assert math.isclose(float(ds.reshape(-1)[0]), -1.0, abs_tol=0.05)
    assert math.isclose(float(de.reshape(-1)[0]),  0.0, abs_tol=0.05)


def test_what_where_pass_through():
    t = TenseLayer(); t.set_op("PAST")
    x, head, _e = _event_with_present_when()
    y = t.forward(x)
    assert torch.allclose(y[..., :head.shape[-1]], head, atol=1e-6)   # head (what/where) unchanged


def test_set_op_rejects_unknown():
    # Both unary .when ops validate the selected op eagerly (fail fast).
    for layer in (TenseLayer(), AspectLayer()):
        try:
            layer.set_op("BOGUS")
        except ValueError:
            continue
        raise AssertionError(f"{type(layer).__name__}.set_op accepted an unknown op")


if __name__ == "__main__":
    unittest.main()
