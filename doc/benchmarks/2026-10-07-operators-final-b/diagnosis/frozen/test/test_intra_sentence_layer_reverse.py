"""Dedicated reverse / invertibility pinning for ``IntraSentenceLayer``.

Task 9 (plan §6) of the STM serial/parallel modes plan. The existing
``test/test_intra_sentence_layer.py`` (Task 2) already has ONE parallel
roundtrip assertion; this file is the thorough roundtrip pinning the
plan calls for -- it exercises BOTH the serial and parallel paths, WITH
and WITHOUT routing, and pins the serial path's documented
approximate-reverse behavior rather than only its output shape.

The layer's invertibility contract (from its class docstring and Task 2):

  * PARALLEL / per-slot path -- ``sigma(pi(x))`` applied per slot with
    square invertible sublayers. ``reverse(forward(x, parallel=True))``
    recovers ``x`` TIGHTLY (up to the LDU inverse tolerance). We pin a
    hard absolute bound well inside the repo's ``2e-1`` invertible-layer
    convention (``test_intra_sentence_layer.test_parallel_roundtrip_*``
    asserts ``< 2e-1``; in practice it is ~1e-7, so we additionally pin a
    tight ``1e-4`` bound to catch silent regressions).

  * SERIAL / collapse path -- the forward sum-folds the K lifted slots
    before the Sigma collapse (many-to-one), and the reverse divides the
    recovered fold equally across ``k = stm_capacity - 1`` slots and
    expands. Two honest consequences are pinned:

      (a) On an INPUT whose ``k`` slots are IDENTICAL, the fold is
          exactly ``k * lifted_slot``, so divide-by-``k`` recovers the
          slot EXACTLY -- the serial roundtrip is tight on the equal-slot
          subspace (``< 1e-4``).
      (b) On a GENERIC input the reverse cannot recover the individual
          slots (it returns ``k`` identical slots = the equal-slot
          projection). The correct invertibility statement is
          re-forward closure: ``forward(reverse(pred)) ≈ pred``
          (``< 1e-4``), because the equal-slot projection lands back on
          the same fold. This is the documented APPROXIMATE bound.

  * ROUTING is an additive Sigma bias, subtracted exactly on reverse
    (``sign=-1``). So a routed roundtrip recovers the same pre-bias
    state, and ``reverse(y, routing)`` equals
    ``reverse(y - projected_bias, routing=None)`` bit-for-bit.

TEST-ONLY: uses the REAL ``IntraSentenceLayer`` (no reimplementation).
Pinned to exploit mode (``set_sigma(0)``) + ``eval()`` for determinism,
mirroring ``test_intra_sentence_layer._make_layer``.
"""
import os
import sys
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

from Layers import IntraSentenceLayer
from util import TheDevice

# Repo invertible-layer convention (test_intra_sentence_layer.py,
# test_invertibility.py admit rel-err ~0.1 on [-1,1] data).
ATOL_REPO = 2e-1
# Tight regression guard -- the square PI->Sigma inverse is ~1e-7 here;
# 1e-4 catches a silent inversion regression while leaving float slack.
ATOL_TIGHT = 1e-4


def _make_layer(concept_dim=6, stm_capacity=8, routing_dim=5,
                working_dim=None, seed=0):
    """Build an IntraSentenceLayer pinned to exploit mode (sigma=0) so
    the invertible-roundtrip assertions are deterministic. Mirrors
    test_intra_sentence_layer._make_layer."""
    torch.manual_seed(seed)
    layer = IntraSentenceLayer(
        concept_dim=concept_dim,
        stm_capacity=stm_capacity,
        routing_dim=routing_dim,
        working_dim=working_dim,
    )
    layer.set_sigma(0)
    layer.eval()
    return layer.to(TheDevice.get())


def _rand_slots(B, K, D):
    return (torch.rand(B, K, D, device=TheDevice.get()) * 2 - 1)


class TestParallelRoundtripTight(unittest.TestCase):
    """Per-slot path is a square isomorphism: reverse(forward(x)) ~= x."""

    def test_parallel_roundtrip_no_routing(self):
        B, N, D = 3, 5, 6
        layer = _make_layer(concept_dim=D)
        x = _rand_slots(B, N, D)
        y = layer.forward(x, routing=None, parallel=True)
        x_rec = layer.reverse(y, routing=None, parallel=True)
        self.assertEqual(tuple(x_rec.shape), (B, N, D))
        err = (x - x_rec).abs().max().item()
        self.assertLess(err, ATOL_REPO,
                        f"parallel roundtrip within repo convention; "
                        f"max|err|={err:.2e}")
        self.assertLess(err, ATOL_TIGHT,
                        f"parallel roundtrip is in fact tight; "
                        f"max|err|={err:.2e}")

    def test_parallel_roundtrip_varied_widths(self):
        # working_dim != concept_dim is still per-slot invertible as long
        # as both sublayers stay square in their own width -- but the
        # layer keeps PI: D->W and Sigma: W->D, so the *composition* is
        # D->D. A non-default working_dim exercises a non-trivial lift.
        B, N, D, W = 2, 4, 6, 10
        layer = _make_layer(concept_dim=D, working_dim=W)
        x = _rand_slots(B, N, D)
        y = layer.forward(x, routing=None, parallel=True)
        self.assertEqual(tuple(y.shape), (B, N, D))
        x_rec = layer.reverse(y, routing=None, parallel=True)
        err = (x - x_rec).abs().max().item()
        self.assertLess(err, ATOL_REPO,
                        f"widened-working-dim parallel roundtrip; "
                        f"max|err|={err:.2e}")

    def test_parallel_roundtrip_with_routing(self):
        B, N, D, R = 3, 5, 6, 5
        layer = _make_layer(concept_dim=D, routing_dim=R)
        with torch.no_grad():
            layer.routing_proj.weight.normal_(0.0, 0.5)
            layer.routing_proj.bias.normal_(0.0, 0.5)
        x = _rand_slots(B, N, D)
        routing = torch.randn(B, R, device=TheDevice.get())
        y = layer.forward(x, routing=routing, parallel=True)
        x_rec = layer.reverse(y, routing=routing, parallel=True)
        err = (x - x_rec).abs().max().item()
        self.assertLess(err, ATOL_REPO,
                        f"routed parallel roundtrip recovers x; "
                        f"max|err|={err:.2e}")
        self.assertLess(err, ATOL_TIGHT,
                        f"routed parallel roundtrip is tight; "
                        f"max|err|={err:.2e}")

    def test_routing_bias_subtracted_exactly_on_reverse(self):
        # reverse(y, routing) must equal reverse(y - projected_bias, None)
        # bit-for-bit: the bias is a pure additive term the reverse undoes
        # before inverting the Sigma/PI bodies.
        B, N, D, R = 3, 5, 6, 5
        layer = _make_layer(concept_dim=D, routing_dim=R)
        with torch.no_grad():
            layer.routing_proj.weight.normal_(0.0, 0.7)
            layer.routing_proj.bias.normal_(0.0, 0.7)
        x = _rand_slots(B, N, D)
        routing = torch.randn(B, R, device=TheDevice.get())
        y = layer.forward(x, routing=routing, parallel=True)
        rec_routed = layer.reverse(y, routing=routing, parallel=True)
        bias = layer.routing_proj(routing).unsqueeze(1)   # [B,1,D]
        rec_manual = layer.reverse(y - bias, routing=None, parallel=True)
        self.assertTrue(
            torch.allclose(rec_routed, rec_manual, atol=0.0),
            "reverse must subtract exactly the projected routing bias.")


class TestSerialRoundtripApproximate(unittest.TestCase):
    """Serial collapse is many-to-one: tight only on the equal-slot
    subspace; otherwise pinned by re-forward closure."""

    def test_serial_reverse_shape_and_finiteness(self):
        B, K, D, cap = 4, 7, 6, 8
        layer = _make_layer(concept_dim=D, stm_capacity=cap)
        x = _rand_slots(B, K, D)
        pred = layer.forward(x, routing=None, parallel=False)
        self.assertEqual(tuple(pred.shape), (B, D))
        recon = layer.reverse(pred, routing=None, parallel=False)
        self.assertEqual(tuple(recon.shape), (B, cap - 1, D),
                         "serial reverse fans the fold across k=cap-1 slots")
        self.assertTrue(torch.isfinite(recon).all())

    def test_serial_roundtrip_tight_on_identical_slots(self):
        # When the k input slots are identical, fold = k * lifted_slot,
        # so divide-by-k recovers the slot exactly -> tight roundtrip on
        # the equal-slot subspace.
        B, D, cap = 3, 6, 8
        k = cap - 1
        layer = _make_layer(concept_dim=D, stm_capacity=cap)
        slot = (torch.rand(B, D, device=TheDevice.get()) * 2 - 1)
        x = slot.unsqueeze(1).expand(-1, k, -1).contiguous()   # [B,k,D]
        pred = layer.forward(x, routing=None, parallel=False)
        recon = layer.reverse(pred, routing=None, parallel=False)
        err = (recon - slot.unsqueeze(1)).abs().max().item()
        self.assertLess(err, ATOL_TIGHT,
                        f"serial roundtrip on identical slots is exact; "
                        f"max|err|={err:.2e}")

    def test_serial_roundtrip_tight_on_identical_slots_with_routing(self):
        B, D, cap, R = 3, 6, 8, 5
        k = cap - 1
        layer = _make_layer(concept_dim=D, stm_capacity=cap, routing_dim=R)
        with torch.no_grad():
            layer.routing_proj.weight.normal_(0.0, 0.5)
            layer.routing_proj.bias.normal_(0.0, 0.5)
        slot = (torch.rand(B, D, device=TheDevice.get()) * 2 - 1)
        x = slot.unsqueeze(1).expand(-1, k, -1).contiguous()
        routing = torch.randn(B, R, device=TheDevice.get())
        pred = layer.forward(x, routing=routing, parallel=False)
        recon = layer.reverse(pred, routing=routing, parallel=False)
        err = (recon - slot.unsqueeze(1)).abs().max().item()
        self.assertLess(err, ATOL_TIGHT,
                        f"routed serial roundtrip on identical slots is "
                        f"exact; max|err|={err:.2e}")

    def test_serial_reverse_returns_equal_slots(self):
        # The collapse inverse can only return the equal-slot projection:
        # every recovered slot is identical (fold/k expanded).
        B, K, D, cap = 4, 7, 6, 8
        layer = _make_layer(concept_dim=D, stm_capacity=cap)
        x = _rand_slots(B, K, D)
        pred = layer.forward(x, routing=None, parallel=False)
        recon = layer.reverse(pred, routing=None, parallel=False)
        for j in range(1, cap - 1):
            self.assertTrue(
                torch.allclose(recon[:, 0, :], recon[:, j, :], atol=ATOL_TIGHT),
                "serial reverse must fan one recovered fold across all slots")

    def test_serial_reforward_closure_generic_input(self):
        # The documented APPROXIMATE bound: the serial reverse is a right
        # inverse on the fold, so forward(reverse(pred)) ~= pred even
        # though reverse(pred) != x for a generic (unequal-slot) x.
        B, D, cap = 3, 6, 8
        k = cap - 1
        layer = _make_layer(concept_dim=D, stm_capacity=cap)
        x = _rand_slots(B, k, D)                 # generic, unequal slots
        pred = layer.forward(x, routing=None, parallel=False)
        recon = layer.reverse(pred, routing=None, parallel=False)
        pred_back = layer.forward(recon, routing=None, parallel=False)
        err = (pred_back - pred).abs().max().item()
        self.assertLess(err, ATOL_TIGHT,
                        f"serial reverse is a right inverse on the fold: "
                        f"forward(reverse(pred)) ~= pred; max|err|={err:.2e}")

    def test_serial_reforward_closure_with_routing(self):
        B, D, cap, R = 3, 6, 8, 5
        k = cap - 1
        layer = _make_layer(concept_dim=D, stm_capacity=cap, routing_dim=R)
        with torch.no_grad():
            layer.routing_proj.weight.normal_(0.0, 0.5)
            layer.routing_proj.bias.normal_(0.0, 0.5)
        x = _rand_slots(B, k, D)
        routing = torch.randn(B, R, device=TheDevice.get())
        pred = layer.forward(x, routing=routing, parallel=False)
        recon = layer.reverse(pred, routing=routing, parallel=False)
        pred_back = layer.forward(recon, routing=routing, parallel=False)
        err = (pred_back - pred).abs().max().item()
        self.assertLess(err, ATOL_TIGHT,
                        f"routed serial re-forward closure; "
                        f"max|err|={err:.2e}")

    def test_serial_reverse_honors_explicit_k(self):
        # An explicit k overrides the stm_capacity-1 default fan-out.
        B, D, cap = 2, 6, 8
        layer = _make_layer(concept_dim=D, stm_capacity=cap)
        slot = (torch.rand(B, D, device=TheDevice.get()) * 2 - 1)
        for k in (1, 3, 5):
            x = slot.unsqueeze(1).expand(-1, k, -1).contiguous()
            pred = layer.forward(x, routing=None, parallel=False)
            recon = layer.reverse(pred, routing=None, parallel=False, k=k)
            self.assertEqual(tuple(recon.shape), (B, k, D))
            err = (recon - slot.unsqueeze(1)).abs().max().item()
            self.assertLess(err, ATOL_TIGHT,
                            f"explicit k={k} identical-slot roundtrip; "
                            f"max|err|={err:.2e}")


class TestReverseInputGuards(unittest.TestCase):
    """reverse rejects mis-shaped inputs (parallel wants [B,N,D], serial
    wants [B,D])."""

    def test_parallel_reverse_rejects_2d(self):
        layer = _make_layer()
        bad = torch.randn(3, 6, device=TheDevice.get())
        with self.assertRaises(ValueError):
            layer.reverse(bad, routing=None, parallel=True)

    def test_serial_reverse_rejects_3d(self):
        layer = _make_layer()
        bad = torch.randn(3, 5, 6, device=TheDevice.get())
        with self.assertRaises(ValueError):
            layer.reverse(bad, routing=None, parallel=False)


if __name__ == "__main__":
    unittest.main()
