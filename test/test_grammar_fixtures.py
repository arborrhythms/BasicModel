"""Phase 5 -- 8 focused grammar fixtures (integration capstone).

doc/plans/2026-06-03-contextual-bind-preposition-when.md. This file is the
integration capstone for Operations 1-3: it does NOT add production code, it
*composes* the Phase 1-4 APIs to structurally model 8 spec sentences, one test
per sentence:

  1. that Alice left            = PREPOSITION(that, LIFT(Alice, left))
  2. Alice wants to run         = LIFT(BIND[Alice], run)
  3. The tired Alice wants to sleep = LIFT(BIND[NP1], sleep), NP1=INTERSECT(Alice, tired)
  4. Alice persuaded Bob to run = LIFT(BIND[Bob], run)  (object-control)
  5. Alice ran                  = PAST(SIMPLE(run))
  6. Alice is running           = PRESENT(PROGRESSIVE(run))
  7. Alice has run              = PRESENT(PERFECT(run))
  8. Alice had been running     = PAST(PERFECT(PROGRESSIVE(run)))

The APIs composed (all already implemented; this file only reads them):
  * Language.PrepositionLayer  -- content-transparent marker packaging.
  * Language.ContextualBindLayer -- nearest-left (slab) / ranked (participants)
    missing-NP resolution.
  * Language.TenseLayer / AspectLayer -- unary .when rewrites on a materialized
    event [B, V, nhead+2].
  * bind_resolver.Participant -- accessible-participant record for the ranked path.
  * surface_tense.normalize_surface -- pure (tense, aspect_chain, base_verb).
  * Spaces.WhenRangeEncoding -- the signed, zero-centered .when range key.

Built-phrase representations (LIFT(Alice, left), the NP/VP constituents in the
BIND slabs) are DISTINCT stand-in tensors, not the live lift/intersect op
output: fixtures 1-4 assert only the NEW ops' behavior (transparency /
nearest-left / ranking), exactly as the existing per-op suites do
(test_contextual_bind.py builds its slabs from plain one-hot tensors). The one
place a constructed-vs-bare distinction is load-bearing -- fixture 3's bind
target NP1 -- is built as a vector provably distinct from the bare ``Alice``
symbol (see ``_intersect_np`` and the fixture's asserts).
"""

import math
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from Language import (PrepositionLayer, ContextualBindLayer,
                      TenseLayer, AspectLayer)
from bind_resolver import Participant
from surface_tense import normalize_surface
from Spaces import (WhenRangeEncoding, _WHEN_TENSE_DEFAULT, _WHEN_TENSE_STEP,
                    _WHEN_PERIOD)

# --- shared preamble -------------------------------------------------------
D = 8                                  # constituent width, consistent per slab
torch.manual_seed(0)

# Distinct symbol vectors for the structural (fixtures 1-4) sentences. One-hot
# rows keep "nearest-left resolved to X" assertions exact (allclose), mirroring
# test_contextual_bind.py's slab construction.
_ALICE = torch.eye(D)[0]
_LEFT = torch.eye(D)[1]
_BIND = torch.eye(D)[2]
_RUN = torch.eye(D)[3]
_TIRED = torch.eye(D)[4]
_SLEEP = torch.eye(D)[5]
_PERSUADED = torch.eye(D)[6]
_BOB = torch.eye(D)[7]


def _slab(*rows):
    """Stack 1-D constituent rows into a [1, N, D] live slab."""
    return torch.stack(rows).unsqueeze(0)


def _intersect_np(a, b):
    """Stand-in for NP1 = INTERSECT(a, b): a vector PROVABLY distinct from
    either bare operand. We do not run the live IntersectionLayer (it operates
    on bivector [B, V, 2] poles via RadMin and needs space_role/activation plumbing
    irrelevant to what fixture 3 asserts); fixture 3 only needs the bind target
    to differ from the bare ``Alice`` symbol. A normalized half-sum of two
    distinct one-hot rows is non-zero on two channels, so it is allclose to
    neither operand -- the asserted distinction."""
    return (a + b) / 2.0


# --- event-tensor helpers (tense fixtures; 2026-06-16 .when bracket redesign) -
# .when is the endpoint-sum bracket over event TIME: ANGLE = event-time center,
# MAGNITUDE = event duration. Tense is the interval-vs-now relation -- PAST shifts
# the center -step ticks, FUTURE +step, PRESENT = identity; aspect is a no-op. The
# default present stamp is an INSTANT at time t (center=t, extent=0).
_ENC = WhenRangeEncoding(_WHEN_PERIOD, 2)
_T = _WHEN_PERIOD // 8                              # a non-aliasing absolute time


def present_event(B=1, V=1, nhead=6, t=_T):
    """A materialized event [B, V, nhead+2] with a present .when tail
    (an instant at absolute time ``t``)."""
    _ENC.t = t
    head = torch.randn(B, V, nhead)
    when = _ENC.encode(t).expand(B, V, -1)
    return torch.cat([head, when], dim=-1)


def when_of(event):
    """Decode the trailing 2 .when columns to (center, extent): event-time center
    (absolute time) and duration."""
    c, ext = _ENC.decode(event[..., -2:])
    return float(c.reshape(-1)[0]), float(ext.reshape(-1)[0])


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
# 2. Alice wants to run  =  VP2 = LIFT(BIND[Alice], run)
# ===========================================================================
def test_fixture_2_alice_wants_to_run():
    layer = ContextualBindLayer()
    slab = _slab(_ALICE, _BIND, _RUN)                  # [1, 3, D]
    layer.set_bind_context(slab=slab)
    out = layer.compose(slab[:, :-1, :], slab[:, 1:, :])   # [1, 2, D], aligned to pairs
    # pairs: 0=(Alice,BIND), 1=(BIND,run). The (BIND, run) pair (index 1)
    # resolves to its nearest-left constituent, Alice.
    assert torch.allclose(out[:, 1, :], _ALICE, atol=1e-6)

    # Ranked path: a lone subject NP under subject-control resolves to Alice.
    layer.set_bind_context(
        participants=[Participant(1, _ALICE.view(1, 1, D), "subject", 0)],
        licensing="subject_control")
    ranked = layer.compose(torch.randn(1, 1, D), torch.randn(1, 1, D))
    assert torch.allclose(ranked, _ALICE.view(1, 1, D).expand_as(ranked), atol=1e-6)


# ===========================================================================
# 3. The tired Alice wants to sleep  =  LIFT(BIND[NP1], sleep),
#    NP1 = INTERSECT(Alice, tired)  -- the bind target is the constructed NP1,
#    NOT the bare Alice symbol.
# ===========================================================================
def test_fixture_3_the_tired_alice_wants_to_sleep():
    np1 = _intersect_np(_ALICE, _TIRED)                # constructed NP, distinct from Alice
    # Guard the premise: NP1 is genuinely not the bare Alice (nor tired) symbol.
    assert not torch.allclose(np1, _ALICE, atol=1e-6)
    assert not torch.allclose(np1, _TIRED, atol=1e-6)

    layer = ContextualBindLayer()
    slab = _slab(np1, _BIND, _SLEEP)                   # NP1 is the constituent before BIND
    layer.set_bind_context(slab=slab)
    out = layer.compose(slab[:, :-1, :], slab[:, 1:, :])   # [1, 2, D]
    # The (BIND, sleep) pair (index 1) resolves to NP1, the constructed NP --
    # and is NOT the bare Alice symbol.
    assert torch.allclose(out[:, 1, :], np1, atol=1e-6)
    assert not torch.allclose(out[:, 1, :], _ALICE, atol=1e-6)

    # Ranked path: subject-control over a participant carrying NP1 returns NP1.
    layer.set_bind_context(
        participants=[Participant(1, np1.view(1, 1, D), "subject", 0)],
        licensing="subject_control")
    ranked = layer.compose(torch.randn(1, 1, D), torch.randn(1, 1, D))
    assert torch.allclose(ranked, np1.view(1, 1, D).expand_as(ranked), atol=1e-6)
    assert not torch.allclose(ranked, _ALICE.view(1, 1, D).expand_as(ranked), atol=1e-6)


# ===========================================================================
# 4. Alice persuaded Bob to run  =  LIFT(BIND[Bob], run)  (object-control)
# ===========================================================================
def test_fixture_4_alice_persuaded_bob_to_run():
    layer = ContextualBindLayer()
    slab = _slab(_ALICE, _PERSUADED, _BOB, _BIND, _RUN)    # [1, 5, D]
    layer.set_bind_context(slab=slab)
    out = layer.compose(slab[:, :-1, :], slab[:, 1:, :])   # [1, 4, D]
    # pairs: 0=(Alice,persuaded) 1=(persuaded,Bob) 2=(Bob,BIND) 3=(BIND,run).
    # The (BIND, run) pair (index 3) resolves to its nearest-left constituent, Bob.
    assert torch.allclose(out[:, 3, :], _BOB, atol=1e-6)
    assert not torch.allclose(out[:, 3, :], _ALICE, atol=1e-6)

    # Ranked path: licensing -- not locality -- is decisive. Alice is the
    # subject at position 0; Bob the object at the MORE-recent position 1, so
    # locality alone would pick Bob in BOTH cases. The discriminator: over the
    # SAME participant list, object-control (persuade) picks Bob while
    # subject-control (want) picks Alice. Identical positions, different result
    # => the pick is driven by licensing, and a licensing-blind resolver could
    # not produce both.
    parts = [Participant(1, _ALICE.view(1, 1, D), "subject", 0),
             Participant(2, _BOB.view(1, 1, D),   "object",  1)]
    layer.set_bind_context(participants=parts, licensing="object_control")
    obj = layer.compose(torch.randn(1, 1, D), torch.randn(1, 1, D))
    assert torch.allclose(obj, _BOB.view(1, 1, D).expand_as(obj), atol=1e-6)        # persuade -> object NP
    layer.set_bind_context(participants=parts, licensing="subject_control")
    subj = layer.compose(torch.randn(1, 1, D), torch.randn(1, 1, D))
    assert torch.allclose(subj, _ALICE.view(1, 1, D).expand_as(subj), atol=1e-6)    # want -> subject NP (same list)
    assert not torch.allclose(obj, subj, atol=1e-6)                                 # locality alone could not yield both


# ===========================================================================
# 5. Alice ran  =  PAST(SIMPLE(run))
# ===========================================================================
def test_fixture_5_alice_ran():
    assert normalize_surface(["ran"]) == ("PAST", [], "run")
    t = TenseLayer(); t.set_op("PAST")
    result = t.forward(present_event())
    # present instant at _T; SIMPLE is a no-op (aspect retired); PAST -> center
    # _T - step (toward the past), duration unchanged (0).
    center, ext = when_of(result)
    assert math.isclose(center, float(_T) - _WHEN_TENSE_STEP, abs_tol=0.05)
    assert math.isclose(ext, 0.0, abs_tol=1e-3)


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

    t = TenseLayer(); t.set_op("PAST")                 # moves the event-time center back
    event = t.forward(event)
    center, ext = when_of(event)
    assert math.isclose(center, float(_T) - _WHEN_TENSE_STEP, abs_tol=0.05)
    assert math.isclose(ext, 0.0, abs_tol=1e-3)


if __name__ == "__main__":
    unittest.main()
