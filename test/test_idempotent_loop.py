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
intrinsic snap exposed via ``Codebook.forward(input, project=True)`` is
the architectural definition of what naming a symbol means: project an
incoming concept activation onto each codebook prototype, populate the
per-prototype catuskoti bivector ``[B, V_S, 2]``. The decode via
``Codebook.reverse(bivec, project=True)`` is the cached SVD-based
pseudo-inverse: it recovers the input modulo span(W).
"""
import sys
import unittest
from pathlib import Path

import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT / "bin"))

from Spaces import Codebook  # noqa: E402


def _make_codebook(D_C=4, V_S=4, invertible=True):
    """Build a unit-norm codebook for clean reasoning."""
    cb = Codebook()
    cb.create(nInput=V_S, nVectors=V_S, nDim=D_C,
              customVQ=False,
              monotonic=True, invertible=invertible)
    cb.setW(torch.eye(V_S, D_C))
    return cb


def _cycle(cb, x):
    """One forward (project) + reverse (pseudo-inverse) cycle."""
    bivec = cb.forward(x, project=True)               # [B, V_S, 2]
    x_back = cb.reverse(bivec, project=True)          # [B, V_in, D_C]
    return x_back, bivec


class TestIdempotentLoop(unittest.TestCase):
    def test_at_prototype_is_fixed_point(self):
        """x == row of W: cycle 0 already at fixed point."""
        cb = _make_codebook(D_C=4, V_S=4)
        x0 = cb.getW()[2:3].unsqueeze(0)               # [1, 1, 4] = row 2
        x1, bivec1 = _cycle(cb, x0)
        x2, _ = _cycle(cb, x1)
        torch.testing.assert_close(x1, x0, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(x2, x1, atol=1e-5, rtol=1e-4)
        # Bivector should be TRUE-corner for prototype 2: pos[2]=1,
        # all other prototypes ≈ 0 (orthogonal in this codebook).
        self.assertAlmostEqual(float(bivec1[0, 2, 0].item()), 1.0, places=5)
        self.assertAlmostEqual(float(bivec1[0, 2, 1].item()), 0.0, places=5)

    def test_noisy_input_converges_in_one_cycle(self):
        """Noisy near-prototype input: cycle 1 strips off-span noise; cycle 2+ identity."""
        cb = _make_codebook(D_C=8, V_S=4)
        cb_W = torch.zeros(4, 8); cb_W[:, :4] = torch.eye(4)
        cb.setW(cb_W)
        torch.manual_seed(0)
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
        torch.manual_seed(42)
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


if __name__ == "__main__":
    unittest.main()
