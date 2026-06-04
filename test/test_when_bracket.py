"""Phase 1 -- unit-bracket ``.when`` point convention.

doc/plans/2026-06-03-contextual-bind-preposition-when.md (single-architecture
convergence). A single stamped time ``t`` is the unit window [t-0.5, t+0.5], so
``encode`` is the mutual inverse of the range ``decode`` and every ``.when``
carries a recoverable duration. The locked aspect scheme reads the reference
``r`` from the interval CENTER: SIMPLE=(r-0.5, r+0.5), PERFECT=(r-1.0, r),
PROGRESSIVE=(r-1.0, r+1.0). The present default stamp is encode_range(-0.5, 0.5).
"""

import math, os, sys, unittest
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from Spaces import WhenRangeEncoding


def _enc(): return WhenRangeEncoding(64, 2)


# --- Task 1.1: encode -> unit bracket --------------------------------------
def test_encode_is_unit_bracket_and_inverts_decode():
    enc = _enc()
    for t in (-1.5, -1.0, 0.0, 0.5, 2.0):
        key = enc.encode(t)
        assert torch.allclose(key, enc.encode_range(t - 0.5, t + 0.5), atol=1e-6)
        ds, de = enc.decode(key)
        assert math.isclose(float(ds), t - 0.5, abs_tol=2e-3)
        assert math.isclose(float(de), t + 0.5, abs_tol=2e-3)


def test_encode_tensor_input_round_trips():
    enc = _enc(); ts = torch.tensor([-1.0, 0.0, 1.0])
    ds, de = enc.decode(enc.encode(ts))
    assert torch.allclose((ds + de) / 2.0, ts, atol=2e-3)


# --- Task 1.2: aspect_interval re-derivation + AspectLayer reads center -----
def test_aspect_interval_bracket_shapes():
    enc = _enc()
    assert enc.aspect_interval(0.0, "SIMPLE")      == (-0.5, 0.5)
    assert enc.aspect_interval(0.0, "PERFECT")     == (-1.0, 0.0)
    assert enc.aspect_interval(0.0, "PROGRESSIVE") == (-1.0, 1.0)
    assert enc.aspect_interval(-1.0, "SIMPLE")     == (-1.5, -0.5)


def test_aspect_layer_reads_center_not_end():
    from Language import AspectLayer
    enc = _enc()
    head = torch.randn(1, 1, 6)
    x = torch.cat([head, enc.encode_range(-1.5, -0.5).expand(1, 1, -1)], dim=-1)  # center -1
    a = AspectLayer(); a.set_op("SIMPLE")
    ds, de = enc.decode(a.forward(x)[..., -2:])      # center -1 -> SIMPLE (-1.5, -0.5)
    assert math.isclose(float(ds.reshape(-1)[0]), -1.5, abs_tol=0.05)
    assert math.isclose(float(de.reshape(-1)[0]), -0.5, abs_tol=0.05)


# --- Task 1.3: present-default stamp -> unit bracket ------------------------
def test_forward_stamps_present_unit_bracket():
    enc = _enc(); y = enc.forward(torch.zeros(2, 3, 10))
    ds, de = enc.decode(y[0, 0, enc.resolve(y.shape[-1])])
    assert math.isclose(float(ds), -0.5, abs_tol=2e-3) and math.isclose(float(de), 0.5, abs_tol=2e-3)


if __name__ == "__main__":
    unittest.main()
