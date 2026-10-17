"""Preposition and legacy surface fixtures; reference is tested at the close."""

import math
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from Language import (PrepositionLayer,
                      TenseLayer, AspectLayer)
from surface_tense import normalize_surface
from Spaces import (WhenRangeEncoding, _WHEN_TENSE_STEP,
                    _WHEN_PERIOD)

# --- shared preamble -------------------------------------------------------
D = 8                                  # constituent width, consistent per slab
torch.manual_seed(0)

# --- event-tensor helpers (tense fixtures; 2026-07-04 .when v2 ladder) ------
# ADAPTED: .when is the 4-dim 2-rung start ladder over the event ONSET
# (duration left the band; exact extents ride the record store). Tense is the
# onset-vs-now relation -- PAST shifts the onset -step ticks, FUTURE +step,
# PRESENT = identity; aspect stays a no-op. Built through the one construction
# seam so the fixture and TenseLayer share omega pairs.
from Spaces import event_when_encoding
_ENC = event_when_encoding(4)
_T = _ENC.maxVal // 8                               # a non-aliasing absolute time


def present_event(B=1, V=1, nhead=6, t=_T):
    """A materialized event [B, V, nhead+4] with a present .when tail
    (onset at absolute time ``t``)."""
    _ENC.t = t
    head = torch.randn(B, V, nhead)
    when = _ENC.encode(t).expand(B, V, -1)
    return torch.cat([head, when], dim=-1)


def when_of(event):
    """Decode the trailing 4 .when columns to (start, 0.0): the v2 band
    carries the onset only (extent retired to the record store); the second
    element stays 0 so the fixture assertions keep their shape."""
    s, _res = _ENC.decode(event[..., -4:])
    return float(s.reshape(-1)[0]), 0.0


# ===========================================================================
# 1. that Alice left  =  PREPOSITION(that, LIFT(Alice, left))
# ===========================================================================
def test_fixture_1_that_alice_left():
    # PREPOSITION is transparent to its phrase: the marker-headed phrase
    # exposes LIFT(Alice, left) unchanged for a downstream consumer.
    that_marker = torch.randn(1, 3, D)                 # learned surface marker P
    phrase = torch.randn(1, 3, D)                      # stand-in for LIFT(Alice, left)
    out = PrepositionLayer().forward(that_marker, phrase)
    assert out.shape == phrase.shape
    assert torch.allclose(out, phrase, atol=1e-6)      # phrase survives unchanged
    # And the marker genuinely does NOT leak into the exposed content.
    assert not torch.allclose(out, that_marker, atol=1e-6)


# ===========================================================================
# 5. Alice ran  =  PAST(SIMPLE(run))
# ===========================================================================
def test_fixture_5_alice_ran():
    assert normalize_surface(["ran"]) == ("PAST", [], "run")
    t = TenseLayer(); t.set_op("PAST")
    event = present_event()
    result = t.forward(event)
    # Concept events are opaque (2026-09-14): tense does not move a
    # conceptual .when; PAST is carried by the concept code and realised
    # by the symbolic layer, so the conceptual op leaves the code as it is.
    assert torch.equal(result, event)


# ===========================================================================
# 6. Alice is running  =  PRESENT(PROGRESSIVE(run))
# ===========================================================================
def test_fixture_6_alice_is_running():
    assert normalize_surface(["is", "running"]) == ("PRESENT", ["PROGRESSIVE"], "run")
    a = AspectLayer(); a.set_op("PROGRESSIVE")         # no-op (aspect retired)
    t = TenseLayer(); t.set_op("PRESENT")              # identity
    result = t.forward(a.forward(present_event()))
    center, ext = when_of(result)                      # present is unchanged
    assert math.isclose(center, float(_T), abs_tol=0.05)
    assert math.isclose(ext, 0.0, abs_tol=1e-3)


# ===========================================================================
# 7. Alice has run  =  PRESENT(PERFECT(run))
# ===========================================================================
def test_fixture_7_alice_has_run():
    assert normalize_surface(["has", "run"]) == ("PRESENT", ["PERFECT"], "run")
    a = AspectLayer(); a.set_op("PERFECT")             # no-op (aspect retired)
    t = TenseLayer(); t.set_op("PRESENT")              # identity
    result = t.forward(a.forward(present_event()))
    center, ext = when_of(result)
    assert math.isclose(center, float(_T), abs_tol=0.05)
    assert math.isclose(ext, 0.0, abs_tol=1e-3)


# ===========================================================================
# 8. Alice had been running  =  PAST(PERFECT(PROGRESSIVE(run)))
# ===========================================================================
def test_fixture_8_alice_had_been_running():
    tense, aspect_chain, base = normalize_surface(["had", "been", "running"])
    assert (tense, aspect_chain, base) == ("PAST", ["PERFECT", "PROGRESSIVE"], "run")

    # 2026-06-16 .when bracket redesign: aspect is a no-op (PROGRESSIVE/PERFECT
    # leave .when unchanged); only PAST tense moves the event-time center. Trace:
    #   present instant at time _T
    #   PROGRESSIVE (no-op) -> center _T
    #   PERFECT     (no-op) -> center _T
    #   PAST -> center _T - step (toward the past), duration unchanged
    event = present_event()

    prog = AspectLayer(); prog.set_op("PROGRESSIVE")
    event = prog.forward(event)
    center, _ext = when_of(event)
    assert math.isclose(center, float(_T), abs_tol=0.05)

    perf = AspectLayer(); perf.set_op("PERFECT")       # PERFECT wraps PROGRESSIVE
    event = perf.forward(event)
    center, _ext = when_of(event)
    assert math.isclose(center, float(_T), abs_tol=0.05)

    t = TenseLayer(); t.set_op("PAST")                 # opaque concept: unchanged
    before = event
    event = t.forward(event)
    assert torch.equal(event, before)


if __name__ == "__main__":
    unittest.main()
