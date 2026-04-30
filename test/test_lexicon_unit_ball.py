"""Projective unit-ball Lexicon: matmul-form RP^D lookup.

Covers the ``Lexicon`` rewrite documented in
[doc/Spaces.md#lexicon](../doc/Spaces.md): default geometry is the
projective unit ball ``B^D / (x ~ -x)`` realized as ``RP^D``. Lookup
score is ``|<x, w>| - 0.5 * ||w||^2``, expressed as a single matmul
+ elementwise abs + bias subtract.

Verifies:
  * Initialization places rows inside the closed unit ball.
  * ``project_unit_ball_`` is idempotent and never grows row norms.
  * Projective primitives (``rp_distance_sq``, ``rp_similarity``,
    ``rp_pode``, ``rp_wrapped_pode``, ``rp_closer_rep``) match their
    brute-force definitions.
  * ``rp_scores`` argmax matches the projective brute-force argmin.
  * ``topk_rp`` returns exact projective distances.
  * ``topk_rp_chunked`` agrees with ``topk_rp`` to numerical precision.
  * Plain L2 helpers (``l2_scores`` / ``topk_l2`` / ``topk_l2_chunked``)
    still produce non-projective L2 sorts.
  * LEGACY torus mode still works when ``torus=True``.
"""

import sys
import unittest
from pathlib import Path

import torch

_project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project / "bin"))

from Layers import Lexicon, topk_l2_chunked, topk_rp_chunked


class TestUnitBallInitAndProjection(unittest.TestCase):
    def test_uniform_ball_init_inside_ball(self):
        torch.manual_seed(0)
        emb = Lexicon(2048, 6)
        norms = emb.weight.norm(dim=-1)
        self.assertTrue(torch.all(norms <= 1.0 + 1e-5),
                        f"max norm {norms.max().item()}")

    def test_small_normal_init_inside_ball(self):
        torch.manual_seed(0)
        emb = Lexicon(512, 8, init="small_normal")
        norms = emb.weight.norm(dim=-1)
        self.assertTrue(torch.all(norms <= 1.0 + 1e-5))

    def test_project_unit_ball_idempotent(self):
        emb = Lexicon(64, 4)
        # Push some rows outside the ball, then project.
        with torch.no_grad():
            emb.weight.mul_(3.0)
        emb.project_unit_ball_()
        norms_a = emb.weight.norm(dim=-1).clone()
        emb.project_unit_ball_()
        norms_b = emb.weight.norm(dim=-1)
        self.assertTrue(torch.allclose(norms_a, norms_b, atol=1e-6))
        self.assertTrue(torch.all(norms_b <= 1.0 + 1e-5))

    def test_project_never_grows_norms(self):
        emb = Lexicon(64, 4)
        before = emb.weight.norm(dim=-1).clone()
        emb.project_unit_ball_()
        after = emb.weight.norm(dim=-1)
        self.assertTrue(torch.all(after <= before + 1e-6))


class TestL2Scores(unittest.TestCase):
    def test_score_sort_matches_brute_force_distance_sort(self):
        torch.manual_seed(0)
        V, D, B = 256, 6, 32
        emb = Lexicon(V, D)
        W_index, W_norm2 = emb.lookup_index()
        x = Lexicon.project_unit_ball(torch.randn(B, D))

        scores = Lexicon.l2_scores(x, W_index, W_norm2)
        idx_score = scores.argmax(dim=-1)

        # Brute force squared L2 distance.
        dist = (x.unsqueeze(1) - W_index.unsqueeze(0)).square().sum(dim=-1)
        idx_dist = dist.argmin(dim=-1)

        self.assertTrue(torch.equal(idx_score, idx_dist))

    def test_l2_scores_shape_and_dtype(self):
        emb = Lexicon(128, 4)
        W_index, W_norm2 = emb.lookup_index()
        x = torch.randn(3, 5, 4)
        s = Lexicon.l2_scores(x, W_index, W_norm2)
        self.assertEqual(s.shape, (3, 5, 128))
        self.assertEqual(s.dtype, x.dtype)

    def test_l2_scores_dim_mismatch_raises(self):
        emb = Lexicon(64, 4)
        W_index, W_norm2 = emb.lookup_index()
        with self.assertRaises(ValueError):
            Lexicon.l2_scores(torch.randn(2, 8), W_index, W_norm2)


class TestTopkL2(unittest.TestCase):
    def test_topk_returns_exact_distances(self):
        torch.manual_seed(0)
        V, D, B = 256, 6, 16
        emb = Lexicon(V, D)
        W_index, W_norm2 = emb.lookup_index()
        x = Lexicon.project_unit_ball(torch.randn(B, D))

        idx, dist_sq, scores = Lexicon.topk_l2(x, W_index, W_norm2, k=8)
        self.assertEqual(idx.shape, (B, 8))
        self.assertEqual(dist_sq.shape, (B, 8))

        # Within each row, dist_sq must be sorted ascending and match
        # the brute-force values.
        brute = (x.unsqueeze(1) - W_index.unsqueeze(0)).square().sum(dim=-1)
        gathered = brute.gather(1, idx)
        self.assertTrue(torch.allclose(gathered, dist_sq, atol=1e-4))
        # Distances strictly nondecreasing along k.
        diffs = dist_sq[:, 1:] - dist_sq[:, :-1]
        self.assertTrue(torch.all(diffs >= -1e-5))

    def test_topk_k_capped_to_V(self):
        emb = Lexicon(8, 4)
        W_index, W_norm2 = emb.lookup_index()
        x = torch.randn(2, 4)
        idx, dist_sq, scores = Lexicon.topk_l2(x, W_index, W_norm2, k=64)
        self.assertEqual(idx.shape[-1], 8)

    def test_topk_zero_k_raises(self):
        emb = Lexicon(8, 4)
        W_index, W_norm2 = emb.lookup_index()
        with self.assertRaises(ValueError):
            Lexicon.topk_l2(torch.randn(1, 4), W_index, W_norm2, k=0)


class TestTopkL2Chunked(unittest.TestCase):
    def test_chunked_matches_unchunked(self):
        torch.manual_seed(0)
        V, D, B = 4096, 6, 8
        emb = Lexicon(V, D)
        W_index, W_norm2 = emb.lookup_index()
        x = Lexicon.project_unit_ball(torch.randn(B, D))

        idx_full, dist_full, _ = Lexicon.topk_l2(x, W_index, W_norm2, k=16)
        idx_chunk, dist_chunk, _ = topk_l2_chunked(
            x, W_index, W_norm2, k=16, chunk_size=512)

        self.assertTrue(torch.equal(idx_full, idx_chunk))
        self.assertTrue(torch.allclose(dist_full, dist_chunk, atol=1e-4))

    def test_chunk_size_larger_than_V(self):
        torch.manual_seed(0)
        emb = Lexicon(64, 4)
        W_index, W_norm2 = emb.lookup_index()
        x = torch.randn(2, 4)
        idx_a, _, _ = Lexicon.topk_l2(x, W_index, W_norm2, k=4)
        idx_b, _, _ = topk_l2_chunked(
            x, W_index, W_norm2, k=4, chunk_size=10_000)
        self.assertTrue(torch.equal(idx_a, idx_b))


class TestProjectivePrimitives(unittest.TestCase):
    """``rp_distance_sq`` / ``rp_similarity`` / ``rp_pode`` /
    ``rp_wrapped_pode`` / ``rp_closer_rep`` agree with their
    closed-form definitions on random inputs."""

    def test_rp_distance_min_of_pode_and_wrapped(self):
        torch.manual_seed(0)
        a = torch.rand(64, 6) * 2 - 1
        b = torch.rand(64, 6) * 2 - 1
        a = Lexicon.project_unit_ball(a)
        b = Lexicon.project_unit_ball(b)

        d_pode = (a - b).square().sum(dim=-1)
        d_wrapped = (a + b).square().sum(dim=-1)
        d_rp = Lexicon.rp_distance_sq(a, b)
        d_min = torch.minimum(d_pode, d_wrapped)
        self.assertTrue(torch.allclose(d_rp, d_min, atol=1e-5))

    def test_rp_similarity_negates_distance(self):
        a = torch.rand(8, 4) * 2 - 1
        b = torch.rand(8, 4) * 2 - 1
        sim = Lexicon.rp_similarity(a, b)
        d = Lexicon.rp_distance_sq(a, b)
        self.assertTrue(torch.allclose(sim, -d, atol=1e-6))

    def test_rp_distance_zero_at_b_or_minus_b(self):
        a = torch.rand(16, 4) * 2 - 1
        a = Lexicon.project_unit_ball(a)
        d_self = Lexicon.rp_distance_sq(a, a)
        d_anti = Lexicon.rp_distance_sq(a, -a)
        self.assertTrue(torch.allclose(d_self, torch.zeros(16), atol=1e-6))
        self.assertTrue(torch.allclose(d_anti, torch.zeros(16), atol=1e-6))

    def test_pode_and_wrapped_pode_midpoints(self):
        a = torch.tensor([0.6, 0.0])
        b = torch.tensor([-0.4, 0.2])
        pode = Lexicon.rp_pode(a, b)
        wrapped = Lexicon.rp_wrapped_pode(a, b)
        self.assertTrue(torch.allclose(pode, 0.5 * (a + b)))
        self.assertTrue(torch.allclose(wrapped, 0.5 * (a - b)))

    def test_rp_distance_is_twice_min_pode_distance(self):
        # The user-facing definition: d_RP(a, b) is twice the smaller of
        # ||a - pode(a, b)|| and ||a - wrapped_pode(a, b)||.
        a = torch.rand(32, 4) * 2 - 1
        b = torch.rand(32, 4) * 2 - 1
        d_a_pode = (a - Lexicon.rp_pode(a, b)).square().sum(dim=-1)
        d_a_wpode = (a - Lexicon.rp_wrapped_pode(a, b)).square().sum(dim=-1)
        twice_min_sq = 4.0 * torch.minimum(d_a_pode, d_a_wpode)
        self.assertTrue(
            torch.allclose(twice_min_sq, Lexicon.rp_distance_sq(a, b),
                           atol=1e-5))

    def test_closer_rep_picks_sign_of_inner_product(self):
        a = torch.tensor([[1.0, 0.0], [-0.5, 0.2]])
        b = torch.tensor([[0.5, 0.5], [0.3, -0.1]])
        closer = Lexicon.rp_closer_rep(a, b)
        # First row: <a, b> > 0, closer = +b
        self.assertTrue(torch.allclose(closer[0], b[0]))
        # Second row: <a, b> < 0, closer = -b
        self.assertTrue(torch.allclose(closer[1], -b[1]))


class TestRpScores(unittest.TestCase):
    def test_rp_score_sort_matches_projective_distance_sort(self):
        torch.manual_seed(0)
        V, D, B = 256, 6, 32
        emb = Lexicon(V, D)
        W_index, W_norm2 = emb.lookup_index()
        x = Lexicon.project_unit_ball(torch.randn(B, D))

        scores = Lexicon.rp_scores(x, W_index, W_norm2)
        idx_score = scores.argmax(dim=-1)

        # Brute-force projective distance.
        d_pode = (x.unsqueeze(1) - W_index.unsqueeze(0)).square().sum(dim=-1)
        d_wrap = (x.unsqueeze(1) + W_index.unsqueeze(0)).square().sum(dim=-1)
        d_rp = torch.minimum(d_pode, d_wrap)
        idx_rp = d_rp.argmin(dim=-1)
        self.assertTrue(torch.equal(idx_score, idx_rp))

    def test_rp_score_treats_w_and_minus_w_equivalent(self):
        # The score depends only on |<x, w>|, so flipping the codebook
        # row should not change the rank.
        torch.manual_seed(0)
        V, D, B = 64, 4, 8
        cb = torch.randn(V, D) * 0.5
        cb = Lexicon.project_unit_ball(cb)
        W_norm2 = cb.square().sum(dim=-1)
        x = Lexicon.project_unit_ball(torch.randn(B, D))

        idx_a = Lexicon.rp_scores(x, cb, W_norm2).argmax(dim=-1)
        idx_b = Lexicon.rp_scores(x, -cb, W_norm2).argmax(dim=-1)
        self.assertTrue(torch.equal(idx_a, idx_b))


class TestTopkRp(unittest.TestCase):
    def test_topk_rp_returns_exact_projective_distances(self):
        torch.manual_seed(0)
        V, D, B = 256, 6, 16
        emb = Lexicon(V, D)
        W_index, W_norm2 = emb.lookup_index()
        x = Lexicon.project_unit_ball(torch.randn(B, D))

        idx, dist_sq, scores = Lexicon.topk_rp(x, W_index, W_norm2, k=8)
        self.assertEqual(idx.shape, (B, 8))

        d_pode = (x.unsqueeze(1) - W_index.unsqueeze(0)).square().sum(dim=-1)
        d_wrap = (x.unsqueeze(1) + W_index.unsqueeze(0)).square().sum(dim=-1)
        brute = torch.minimum(d_pode, d_wrap)
        gathered = brute.gather(1, idx)
        self.assertTrue(torch.allclose(gathered, dist_sq, atol=1e-4))
        diffs = dist_sq[:, 1:] - dist_sq[:, :-1]
        self.assertTrue(torch.all(diffs >= -1e-5))

    def test_topk_rp_chunked_matches_unchunked(self):
        torch.manual_seed(0)
        V, D, B = 4096, 6, 8
        emb = Lexicon(V, D)
        W_index, W_norm2 = emb.lookup_index()
        x = Lexicon.project_unit_ball(torch.randn(B, D))

        idx_full, dist_full, _ = Lexicon.topk_rp(x, W_index, W_norm2, k=16)
        idx_chunk, dist_chunk, _ = topk_rp_chunked(
            x, W_index, W_norm2, k=16, chunk_size=512)
        self.assertTrue(torch.equal(idx_full, idx_chunk))
        self.assertTrue(torch.allclose(dist_full, dist_chunk, atol=1e-4))


class TestTorusLegacyMode(unittest.TestCase):
    """``torus=True`` selects the legacy flat-torus geometry. The
    static torus primitives must still operate correctly so call sites
    that depend on them keep working."""

    def test_torus_init_inside_canonical_cell(self):
        torch.manual_seed(0)
        emb = Lexicon(256, 6, torus=True)
        self.assertTrue(torch.all(emb.weight >= -1.0))
        self.assertTrue(torch.all(emb.weight < 1.0))

    def test_torus_normalize_wraps(self):
        emb = Lexicon(64, 4, torus=True)
        with torch.no_grad():
            emb.weight.add_(3.0)
        emb.normalize()
        self.assertTrue(torch.all(emb.weight >= -1.0))
        self.assertTrue(torch.all(emb.weight < 1.0))

    def test_torus_distance_primitives_still_work(self):
        a = torch.tensor([0.4, -0.7])
        b = torch.tensor([0.0, 0.0])
        d_sq = Lexicon.distance_sq(a, b)
        sim = Lexicon.similarity(a, b)
        self.assertTrue(torch.isfinite(d_sq))
        self.assertTrue(torch.allclose(sim, -d_sq))

    def test_unit_ball_normalize_projects_into_ball(self):
        emb = Lexicon(64, 4)  # default unit-ball
        with torch.no_grad():
            emb.weight.mul_(5.0)
        emb.normalize()
        self.assertTrue(torch.all(emb.weight.norm(dim=-1) <= 1.0 + 1e-5))


if __name__ == "__main__":
    unittest.main()
