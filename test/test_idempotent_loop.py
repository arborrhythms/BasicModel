"""Iterative-convergence idempotency for the C↔S round trip without grammar.

Verifies that the codebook's project / project_reverse pair, used as the
intrinsic snap and decode at the C↔S boundary, drives the round trip to a
fixed point on the codebook's row space within one cycle:

  * cycle 0 → cycle 1: projection onto span(W). Non-trivial when input
    has off-span components.
  * cycle 1 → cycle 2 → ...: identity. Per-cycle delta is ~0.
  * orthogonal-to-span input: collapses to ~0 (NEITHER everywhere) and
    stays there.
  * at-prototype input: cycle 0 already at fixed point.

Tests exercise the Codebook directly (not the SymbolicSpace shell), to
keep the focus on the snap loop without grammar entanglement. The
intrinsic snap exposed via ``Codebook.forward(input)`` is
the architectural definition of what naming a symbol means: project an
incoming concept activation onto each codebook prototype, populate the
per-prototype catuskoti bivector ``[B, V_S, 2]``. The decode via
``Codebook.reverse(bivec)`` is the cached SVD-based
pseudo-inverse: it recovers the input modulo span(W).
"""
import sys
import unittest
from pathlib import Path

import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT / "bin"))

from Spaces import ProjectionBasis  # noqa: E402


def _make_codebook(D_C=4, V_S=4, invertible=True):
    """Build a unit-norm projection basis for clean reasoning.

    ``ProjectionBasis`` defaults to identity W via its LDU
    factorization (raw_L = raw_U = 0, d = 1), which is exactly what
    the legacy Codebook tests pinned via ``setW(torch.eye(...))``.
    The ``invertible`` kwarg is retained for back-compat with the
    old signature but is ignored -- ProjectionBasis is structurally
    invertible by construction.
    """
    cb = ProjectionBasis()
    cb.create(nInput=V_S, nVectors=V_S, nDim=D_C)
    return cb


def _cycle(cb, x):
    """One forward (project) + reverse (pseudo-inverse) cycle."""
    proj = cb.forward(x)               # [B, V_S, 1] signed DoT
    x_back = cb.reverse(proj)          # [B, V_in, D_C]
    return x_back, proj


class TestIdempotentLoop(unittest.TestCase):
    def test_at_prototype_is_fixed_point(self):
        """x == row of W: cycle 0 already at fixed point."""
        cb = _make_codebook(D_C=4, V_S=4)
        x0 = cb.getW()[2:3].unsqueeze(0)               # [1, 1, 4] = row 2
        x1, bivec1 = _cycle(cb, x0)
        x2, _ = _cycle(cb, x1)
        torch.testing.assert_close(x1, x0, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(x2, x1, atol=1e-5, rtol=1e-4)
        # Scalar projection (bivector retired): prototype 2 ≈ +1 (signed
        # Degree-of-Truth, the TRUE corner), all other prototypes ≈ 0
        # (orthogonal in this codebook).
        dot = bivec1[0, :, 0]                          # [V_S] signed DoT
        self.assertAlmostEqual(float(dot[2].item()), 1.0, places=5)
        others = torch.cat([dot[:2], dot[3:]])
        self.assertLess(float(others.abs().max().item()), 1e-5)

    def test_noisy_input_converges_in_one_cycle(self):
        """Noisy near-prototype input: cycle 1 strips off-span noise; cycle 2+ identity."""
        cb = _make_codebook(D_C=8, V_S=4)
        cb_W = torch.zeros(4, 8); cb_W[:, :4] = torch.eye(4)
        cb.setW(cb_W)

        # Input: row 1 (in span) + noise on axes 4-7 (orthogonal to span).
        x0 = cb_W[1:2].unsqueeze(0) + torch.cat(
            [torch.zeros(1, 1, 4), 0.1 * torch.randn(1, 1, 4)], dim=-1)
        x1, _ = _cycle(cb, x0)
        x2, _ = _cycle(cb, x1)
        x3, _ = _cycle(cb, x2)
        delta_01 = (x1 - x0).norm().item()
        delta_12 = (x2 - x1).norm().item()
        delta_23 = (x3 - x2).norm().item()
        self.assertGreater(delta_01, 1e-3)             # cycle 1 changed state
        self.assertLess(delta_12, 1e-5)                # cycle 2 is fixed
        self.assertLess(delta_23, 1e-5)                # subsequent identity

    def test_iterative_snap_distance_reduces_to_zero(self):
        """Across 5 iterations on noisy input, per-cycle delta reduces to ~0.

        This is the convergence guarantee: forward → reverse projects
        onto span(W). After the first projection the state is in
        span(W); subsequent projections are identity.
        """
        cb = _make_codebook(D_C=8, V_S=4)
        cb_W = torch.zeros(4, 8); cb_W[:, :4] = torch.eye(4)
        cb.setW(cb_W)

        x = 0.3 * cb_W[3:4].unsqueeze(0) + 0.1 * torch.randn(1, 1, 8)
        deltas = []
        for _ in range(5):
            x_next, _ = _cycle(cb, x)
            deltas.append((x_next - x).norm().item())
            x = x_next
        # Cycle 1 has the largest delta (the projection step).
        # Cycles 2+ are ~0 (already on the lattice).
        self.assertGreaterEqual(deltas[0], deltas[1])
        for d in deltas[1:]:
            self.assertLess(d, 1e-5)

    def test_orthogonal_input_collapses_to_neither(self):
        """x orthogonal to span(W): bivector ~ 0, decoded x' ~ 0."""
        cb = _make_codebook(D_C=8, V_S=4)
        cb_W = torch.zeros(4, 8); cb_W[:, :4] = torch.eye(4)
        cb.setW(cb_W)
        x = torch.zeros(1, 1, 8); x[0, 0, 5] = 1.0       # orthogonal to span
        x_back, bivec = _cycle(cb, x)
        self.assertLess(x_back.norm().item(), 1e-5)
        self.assertLess(bivec.abs().max().item(), 1e-5)


def _widen_roundtrip_perslot(cb, slab):
    """The H2 widened symbolic-space loop, mirrored from
    ``BasicModel._stm_symbolic_roundtrip``: fold the ``cap`` axis into the
    batch dim (one STM slot per row), ``forward`` then ``reverse(V=1)``,
    unfold. Generalizes the retired single-vector ``V=1`` round-trip to
    ``V = STM capacity`` while preserving every slot (drops the legacy
    ``idea_back.sum(dim=1)`` single-slot collapse; does NOT replace it
    with a V-axis mean collapse).
    """
    B, cap, D = slab.shape
    flat = slab.reshape(B * cap, 1, D)        # [B*cap, 1, D]
    snap = cb.forward(flat)                   # [B*cap, V_S, 1]
    back = cb.reverse(snap, V=1)              # [B*cap, 1, D]
    return back.reshape(B, cap, D)            # [B, cap, D]


class TestWidenedIdempotentLoopV7(unittest.TestCase):
    """H2: the C↔S round-trip widened from the single-vector ``V=1``
    regime to ``V = STM capacity = 7`` is still a fixed point -- per slot
    -- so the ``[B, 7, concept_dim]`` STM slab can pass through symbolic
    space and back idempotently. ADDED alongside (does NOT weaken) the
    ``V=1`` assertions above: the widening must preserve idempotency.
    """

    def test_v7_perslot_roundtrip_is_fixed_point(self):
        """Non-trivial codebook, ``[B, 7, 1024]`` slab (post-H3 dims):
        round-tripping an already-round-tripped slab returns it
        bit-identically (the same fixed-point property as ``V=1``, here
        per-slot over the 7 STM slots)."""
        torch.manual_seed(1234)
        B, cap, D = 4, 7, 1024
        cb = _make_codebook(D_C=D, V_S=cap)   # span(W) is a proper 7-dim
                                              # subspace of R^1024
        x0 = torch.randn(B, cap, D)           # large off-span component
        x1 = _widen_roundtrip_perslot(cb, x0)  # cycle(x): projects onto span(W)
        x2 = _widen_roundtrip_perslot(cb, x1)  # cycle(cycle(x))
        # Projection actually happened (input had off-span energy).
        self.assertGreater((x1 - x0).abs().max().item(), 1e-3)
        # Fixed point at width 7, bit-identical (strict; the V=1
        # assertions use atol=1e-5 -- the widened per-slot path is exact).
        self.assertTrue(torch.equal(x2, x1),
                        f"V=7 not a fixed point: "
                        f"max|Δ|={ (x2 - x1).abs().max().item():.3e}")
        torch.testing.assert_close(x2, x1, atol=1e-5, rtol=1e-4)
        # All 7 slots stay DISTINCT (the slab is a real per-slot working
        # memory, not collapsed to one summary -- the failure mode of a
        # naive ``cb.reverse(snap, V=cap)`` V-axis mean collapse).
        self.assertGreater((x1 - x1[:, :1, :]).abs().max().item(), 1e-3)

    def test_v7_at_prototype_slab_is_fixed_point_cycle0(self):
        """A slab whose every slot is a codebook prototype row is already
        a fixed point at cycle 0 (per-slot generalization of
        ``test_at_prototype_is_fixed_point``)."""
        cap = 7
        cb = _make_codebook(D_C=cap, V_S=cap)  # identity W (unit rows)
        W = cb.getW()                          # [cap, cap]
        slab0 = W.unsqueeze(0).expand(3, cap, cap).contiguous()  # [3,7,7]
        slab1 = _widen_roundtrip_perslot(cb, slab0)
        slab2 = _widen_roundtrip_perslot(cb, slab1)
        torch.testing.assert_close(slab1, slab0, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(slab2, slab1, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
