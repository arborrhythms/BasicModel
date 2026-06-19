"""Phase 1 & 2 tests: Taxonomy derived heat helpers and candidate union.

Tests 1-4 from the plan
``doc/plans/2026-06-06-symbolic-heat-retrieval.md`` §Tests:

  1. ``Taxonomy.heat_mask`` returns ``_priming - 1`` clipped at zero.
  2. ``Taxonomy.topk_heat`` returns only live rows and respects row
     restriction.
  3. ``build_semantic_heat`` equals dense reference ``A_S^T r_S``.
  4. Low-rank outer product computes ``Cq`` equal to dense ``C @ q``.

Tests 5-7 (Phase 2) cover ``SymbolicSubSpace.retrieval_candidates_for_slot``
(plan §Candidate generation / §Recommender changes):

  5. Candidate union includes content-nearest rows even when cold.
  6. Candidate union includes hot rows even when content similarity is weak.
  7. Typed masks exclude hot but inadmissible rows; priming is 1.0 for
     non-candidates and >1.0 for at least one hot/near candidate.
"""

import sys
from pathlib import Path

import pytest
import torch

# conftest.py already adds bin/ to sys.path, but be defensive.
_project = Path(__file__).resolve().parent.parent
_bin = str(_project / "bin")
if _bin not in sys.path:
    sys.path.insert(0, _bin)

from Language import Taxonomy, SymbolicSubSpace  # noqa: E402
from Layers import Ops  # noqa: E402
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_tax(B=2, V=8, L=6):
    """Return a Taxonomy with _priming allocated at [B, V], live=L."""
    tax = Taxonomy()
    tax.allocate_priming(batch_size=B, capacity=V, live=L)
    return tax


# ---------------------------------------------------------------------------
# Test 1: heat_mask clips priming at 0 and respects live slice
# ---------------------------------------------------------------------------

class TestHeatMask:
    def test_cold_rows_give_zero(self):
        """All-neutral priming => heat_mask is all zeros."""
        tax = _make_tax(B=2, V=8, L=6)
        r = tax.heat_mask(batch=0)
        assert r is not None
        assert r.shape == (6,)
        assert r.eq(0).all(), f"Expected zeros, got {r}"

    def test_hot_rows_give_excess(self):
        """Rows primed above 1.0 produce positive heat."""
        tax = _make_tax(B=2, V=8, L=6)
        tax._priming[0, 2] = 1.5
        tax._priming[0, 4] = 2.0
        r = tax.heat_mask(batch=0)
        assert abs(float(r[2]) - 0.5) < 1e-6
        assert abs(float(r[4]) - 1.0) < 1e-6
        assert r[0].item() == 0.0
        assert r[5].item() == 0.0

    def test_matches_priming_mask_formula(self):
        """heat_mask(b) == (priming_mask(b) - 1).clamp(min=0)."""
        tax = _make_tax(B=2, V=8, L=6)
        tax._priming[1, 0] = 1.3
        tax._priming[1, 3] = 0.8   # sub-neutral; must clip to 0
        r = tax.heat_mask(batch=1)
        ref = (tax.priming_mask(batch=1) - 1.0).clamp(min=0.0)
        assert torch.allclose(r, ref), f"{r} != {ref}"

    def test_batch_none_returns_all_rows(self):
        """heat_mask(batch=None) returns [B, V_live]."""
        tax = _make_tax(B=3, V=10, L=7)
        tax._priming[2, 5] = 2.5
        r = tax.heat_mask(batch=None)
        assert r.shape == (3, 7)
        assert abs(float(r[2, 5]) - 1.5) < 1e-6

    def test_returns_only_live_slice(self):
        """Slack columns (index >= L) must not appear in the result."""
        tax = _make_tax(B=1, V=8, L=4)
        # Prime a slack column directly.
        tax._priming[0, 6] = 5.0
        r = tax.heat_mask(batch=0)
        assert r.shape == (4,), f"Expected live=4 slice, got shape {r.shape}"

    def test_returns_none_when_no_buffer(self):
        """heat_mask returns None when _priming is None."""
        tax = Taxonomy()
        assert tax.heat_mask(batch=0) is None
        assert tax.heat_mask(batch=None) is None


# ---------------------------------------------------------------------------
# Test 2: topk_heat live-slice and row restriction
# ---------------------------------------------------------------------------

class TestTopkHeat:
    def test_excludes_slack_rows(self):
        """A hot row beyond live must be excluded from topk_heat."""
        tax = _make_tax(B=2, V=8, L=4)
        # Hot live row at index 2.
        tax._priming[0, 2] = 2.0
        # Hot SLACK row at index 6 (>= L=4).
        tax._priming[0, 6] = 9.9
        ids = tax.topk_heat(k=5, batch=0)
        assert 6 not in ids.tolist(), "Slack row must be excluded"
        assert 2 in ids.tolist(), "Live hot row must be included"

    def test_sorted_by_heat_descending(self):
        """topk_heat returns ids with highest heat first."""
        tax = _make_tax(B=1, V=8, L=6)
        tax._priming[0, 0] = 1.1
        tax._priming[0, 3] = 1.8   # hottest
        tax._priming[0, 5] = 1.5   # second
        ids = tax.topk_heat(k=3, batch=0)
        assert ids[0].item() == 3
        assert ids[1].item() == 5

    def test_returns_fewer_than_k_when_few_hot(self):
        """If only 2 rows are hot, topk_heat(k=10) returns 2."""
        tax = _make_tax(B=1, V=8, L=6)
        tax._priming[0, 1] = 1.5
        tax._priming[0, 4] = 2.0
        ids = tax.topk_heat(k=10, batch=0)
        assert ids.numel() == 2

    def test_rows_restriction(self):
        """rows= restricts candidates; only hot restricted ids returned."""
        tax = _make_tax(B=1, V=8, L=6)
        tax._priming[0, 0] = 1.5
        tax._priming[0, 2] = 2.0
        tax._priming[0, 4] = 1.9
        restrict = torch.tensor([0, 4])
        ids = tax.topk_heat(k=5, batch=0, rows=restrict)
        assert set(ids.tolist()).issubset({0, 4}), f"Unexpected ids: {ids}"
        assert 2 not in ids.tolist()

    def test_rows_restriction_excludes_slack(self):
        """rows= restriction that includes a slack id is silently dropped."""
        tax = _make_tax(B=1, V=8, L=4)
        tax._priming[0, 2] = 2.0
        tax._priming[0, 7] = 9.0   # slack
        restrict = torch.tensor([2, 7])
        ids = tax.topk_heat(k=5, batch=0, rows=restrict)
        assert 7 not in ids.tolist()
        assert 2 in ids.tolist()

    def test_empty_when_nothing_hot(self):
        """If no rows are hot (all priming==1), return empty tensor."""
        tax = _make_tax(B=1, V=8, L=6)
        ids = tax.topk_heat(k=4, batch=0)
        assert ids.numel() == 0

    def test_empty_when_k_zero(self):
        """topk_heat(k=0) returns empty tensor."""
        tax = _make_tax(B=1, V=8, L=6)
        tax._priming[0, 0] = 2.0
        ids = tax.topk_heat(k=0, batch=0)
        assert ids.numel() == 0

    def test_empty_when_no_buffer(self):
        """topk_heat returns empty LongTensor when _priming is None."""
        tax = Taxonomy()
        ids = tax.topk_heat(k=5, batch=0)
        assert isinstance(ids, torch.Tensor)
        assert ids.numel() == 0


# ---------------------------------------------------------------------------
# Test 3: build_semantic_heat matches dense reference
# ---------------------------------------------------------------------------

class TestBuildSemanticHeat:
    def _setup(self, B=2, V=8, L=6, D=5):
        tax = _make_tax(B=B, V=V, L=L)
        A = torch.randn(V, D)
        return tax, A, D, L

    def test_matches_dense_reference(self):
        """build_semantic_heat(A, batch=b) == r_live @ A[:L]."""
        tax, A, D, L = self._setup()
        b = 0
        tax._priming[b, 0] = 1.8
        tax._priming[b, 3] = 2.2
        z = tax.build_semantic_heat(A, batch=b)
        r_live = (tax._priming[b, :L] - 1.0).clamp(min=0.0)
        ref = r_live @ A[:L]
        assert z.shape == (D,)
        assert torch.allclose(z, ref, atol=1e-5), f"max diff {(z-ref).abs().max()}"

    def test_zeros_when_nothing_hot(self):
        """All-cold priming -> zero semantic heat."""
        tax, A, D, L = self._setup()
        z = tax.build_semantic_heat(A, batch=0)
        assert torch.allclose(z, torch.zeros(D)), f"Expected zeros, got {z}"

    def test_rows_restriction(self):
        """rows= restricts which live rows contribute."""
        tax, A, D, L = self._setup()
        b = 0
        tax._priming[b, 0] = 2.0
        tax._priming[b, 2] = 1.5
        tax._priming[b, 4] = 3.0
        # Restrict to rows {0, 4} only.
        restrict = torch.tensor([0, 4])
        z_r = tax.build_semantic_heat(A, batch=b, rows=restrict)
        # Build reference manually.
        r_live = (tax._priming[b, :L] - 1.0).clamp(min=0.0)
        r_sub = r_live[restrict]
        ref = r_sub @ A[restrict]
        assert torch.allclose(z_r, ref, atol=1e-5)

    def test_topk_restriction(self):
        """topk= limits to the k hottest rows."""
        tax, A, D, L = self._setup()
        b = 1
        tax._priming[b, 1] = 1.3
        tax._priming[b, 3] = 2.5   # hottest
        tax._priming[b, 5] = 1.9   # second
        z_k = tax.build_semantic_heat(A, batch=b, topk=2)
        # Only rows 3 and 5 should contribute.
        ids = torch.tensor([3, 5])
        r_live = (tax._priming[b, :L] - 1.0).clamp(min=0.0)
        ref = r_live[ids] @ A[ids]
        assert torch.allclose(z_k, ref, atol=1e-5)
        # The topk=2 result must exclude row 1 (the third-hottest).  Verify by
        # checking it differs from the untrimmed full result (which includes all
        # three hot rows).  The codebook is random, so equality would be a fluke.
        z_full = tax.build_semantic_heat(A, batch=b)
        assert not torch.allclose(z_k, z_full, atol=1e-6), (
            "topk=2 result should exclude the third-hottest row but matched "
            "the full (untrimmed) result — topk likely not trimming"
        )

    def test_returns_zeros_when_no_buffer(self):
        """Returns zeros([D]) when _priming is None."""
        tax = Taxonomy()
        A = torch.randn(8, 5)
        z = tax.build_semantic_heat(A, batch=0)
        assert z.shape == (5,)
        assert z.eq(0).all()

    def test_dtype_matches_codebook(self):
        """Result dtype matches codebook_rows dtype."""
        tax, _, D, L = self._setup()
        A_f16 = torch.randn(8, D, dtype=torch.float16)
        tax._priming[0, 0] = 1.5
        z = tax.build_semantic_heat(A_f16, batch=0)
        assert z.dtype == torch.float16


# ---------------------------------------------------------------------------
# Test 4: build_outer_heat low-rank equals dense C
# ---------------------------------------------------------------------------

class TestBuildOuterHeat:
    def _setup(self, B=2, V=8, L=5, D=4):
        tax = _make_tax(B=B, V=V, L=L)
        A = torch.randn(V, D)
        return tax, A, D, L

    def test_lowrank_U_times_q_equals_dense_C_times_q(self):
        """U^T (U q) must equal C q == (U^T U) q for random q."""
        tax, A, D, L = self._setup()
        b = 0
        tax._priming[b, 0] = 1.4
        tax._priming[b, 2] = 2.1
        tax._priming[b, 3] = 1.7

        U = tax.build_outer_heat(A, batch=b, low_rank=True)
        C = tax.build_outer_heat(A, batch=b, low_rank=False)

        assert U.shape[1] == D
        assert C.shape == (D, D)

        q = torch.randn(D)
        Cq_from_U = U.t() @ (U @ q)
        Cq_from_C = C @ q
        assert torch.allclose(Cq_from_U, Cq_from_C, atol=1e-5), (
            f"max diff {(Cq_from_U - Cq_from_C).abs().max()}"
        )

    def test_dense_C_equals_UtU(self):
        """Dense C must equal U^T U."""
        tax, A, D, L = self._setup()
        b = 1
        tax._priming[b, 1] = 1.9
        tax._priming[b, 4] = 2.5

        U = tax.build_outer_heat(A, batch=b, low_rank=True)
        C = tax.build_outer_heat(A, batch=b, low_rank=False)
        assert torch.allclose(C, U.t() @ U, atol=1e-5)

    def test_empty_S_low_rank_shape(self):
        """Empty active set => U has shape [0, D]."""
        tax, A, D, L = self._setup()
        U = tax.build_outer_heat(A, batch=0, low_rank=True)
        assert U.shape == (0, D)

    def test_empty_S_dense_is_zeros(self):
        """Empty active set => dense C is zeros([D, D])."""
        tax, A, D, L = self._setup()
        C = tax.build_outer_heat(A, batch=0, low_rank=False)
        assert C.shape == (D, D)
        assert C.eq(0).all()

    def test_rows_restriction(self):
        """rows= restricts which live rows appear in U."""
        tax, A, D, L = self._setup()
        b = 0
        tax._priming[b, 0] = 2.0
        tax._priming[b, 2] = 1.5
        tax._priming[b, 3] = 1.8
        restrict = torch.tensor([0, 2])
        U_r = tax.build_outer_heat(A, batch=b, rows=restrict, low_rank=True)
        C_r = tax.build_outer_heat(A, batch=b, rows=restrict, low_rank=False)
        assert torch.allclose(C_r, U_r.t() @ U_r, atol=1e-5)
        # Recompute the expected restricted C directly from the active set
        # {0, 2} to verify the implementation used exactly those rows.
        r_live = (tax._priming[b, :L] - 1.0).clamp(min=0.0)
        ids_ref = restrict  # both in-range and hot after priming above
        r_ref = r_live[ids_ref]
        A_ref = A[ids_ref]                          # [2, D]
        sqrt_r_ref = r_ref.sqrt().unsqueeze(1)      # [2, 1]
        U_ref = sqrt_r_ref * A_ref                  # [2, D]
        C_ref = U_ref.t() @ U_ref                   # [D, D]
        assert torch.allclose(C_r, C_ref, atol=1e-5), (
            f"C_r does not match restricted-set reference; "
            f"max diff {(C_r - C_ref).abs().max()}"
        )

    def test_dtype_matches_codebook(self):
        """Result dtype matches codebook_rows dtype."""
        tax, _, D, L = self._setup()
        A_f16 = torch.randn(8, D, dtype=torch.float16)
        tax._priming[0, 0] = 1.5
        U = tax.build_outer_heat(A_f16, batch=0, low_rank=True)
        assert U.dtype == torch.float16


# ---------------------------------------------------------------------------
# Helpers shared across Phase-2 tests
# ---------------------------------------------------------------------------

class FakeView:
    """Duck-typed KnowledgeView for unit tests.

    Stores pre-built ref-id tensors per category name and per integer order.
    Mirrors the interface ``SymbolicSubSpace.retrieval_candidates_for_slot``
    requires (plan §Candidate generation).
    """

    def __init__(self, by_cat, by_order):
        self._c = by_cat    # dict[str, LongTensor]
        self._o = by_order  # dict[int, LongTensor]

    def refs_by_category(self, name):
        return self._c.get(name, torch.empty(0, dtype=torch.long))

    def refs_by_order(self, o):
        return self._o.get(int(o), torch.empty(0, dtype=torch.long))


def _call_retrieval(stub, A, query, category, order, **kw):
    """Invoke ``SymbolicSubSpace.retrieval_candidates_for_slot`` via the class
    (unbound call) so no full ``SymbolicSubSpace`` construction is needed."""
    basis = SimpleNamespace(getW=lambda: A)
    return SymbolicSubSpace.retrieval_candidates_for_slot(
        stub, query=query, basis=basis,
        category=category, order=order, **kw)


# ---------------------------------------------------------------------------
# Test 5: content-nearest row appears in candidate union even when cold
# ---------------------------------------------------------------------------

class TestCandidateUnionIncludesContentNearest:
    """Plan test 5: cold but content-nearest row must reach ``out['rows']``.

    Construct a deterministic codebook ``A`` of shape ``[K, D]`` where row 0
    is the closest to ``q`` by cosine similarity, while its priming stays at
    the neutral 1.0 (cold).  The typed admissible set either admits row 0 or
    is left empty (so no typed filter applies).  Assert that row 0 ends up in
    ``out['rows']``.
    """

    def _setup(self):
        torch.manual_seed(42)
        K, D = 10, 8
        # Build codebook: all rows random except row 0 which we set == q.
        A = torch.randn(K, D)
        A = A / A.norm(dim=1, keepdim=True).clamp(min=1e-6)
        q = torch.randn(D)
        q = q / q.norm()
        # Point row 0 directly at q -> cosine similarity == 1.0
        A[0] = q.clone()
        return A, q, K, D

    def test_cold_nearest_in_union_no_typed_filter(self):
        """Cold nearest row is included when C_typed is empty (no filter)."""
        A, q, K, D = self._setup()
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=K, live=K)
        # Row 0 stays cold (priming == 1.0).
        # No typed filter: FakeView returns empty tensors for this category.
        view = FakeView(by_cat={}, by_order={})
        stub = SimpleNamespace(knowledge=view, taxonomy=tax)

        out = _call_retrieval(stub, A, q, category='N', order=0,
                              topk_content=5, topk_heat=5)

        assert 'rows' in out, "Expected 'rows' key in output"
        assert 0 in out['rows'].tolist(), (
            "Content-nearest row 0 (cold) must appear in candidate union")

    def test_cold_nearest_in_union_typed_admits_it(self):
        """Cold nearest row is included when C_typed admits its ref-id."""
        A, q, K, D = self._setup()
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=K, live=K)
        # C_typed admits row 0 and a few others.
        admitted = torch.tensor([0, 3, 7], dtype=torch.long)
        view = FakeView(by_cat={'N': admitted},
                        by_order={0: admitted})
        stub = SimpleNamespace(knowledge=view, taxonomy=tax)

        out = _call_retrieval(stub, A, q, category='N', order=0,
                              topk_content=5, topk_heat=5)

        assert 0 in out['rows'].tolist(), (
            "Content-nearest row 0 must survive typed intersection")


# ---------------------------------------------------------------------------
# Test 6: hot-but-distant row appears in candidate union
# ---------------------------------------------------------------------------

class TestCandidateUnionIncludesHot:
    """Plan test 6: a hot row with low cosine to ``q`` must reach
    ``out['rows']`` via the heat branch of the union.

    We construct a codebook where row 5 is orthogonal (or even anti-parallel)
    to ``q`` but is strongly primed.  The typed set either admits it or is
    empty.  Assert row 5 ∈ ``out['rows']``.
    """

    def _setup(self):
        torch.manual_seed(7)
        K, D = 12, 6
        A = torch.randn(K, D)
        A = A / A.norm(dim=1, keepdim=True).clamp(min=1e-6)
        # q aligns with row 0 only.
        q = A[0].clone()
        # Make row 5 nearly anti-parallel to q (low / negative cosine).
        A[5] = -q.clone()
        A[5] = A[5] / A[5].norm()
        return A, q, K, D

    def test_hot_distant_in_union_no_typed_filter(self):
        """Hot distant row reaches candidate union without a typed filter."""
        A, q, K, D = self._setup()
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=K, live=K)
        # Prime row 5 strongly.
        tax.prime([5], batch=0, boost=10.0)
        view = FakeView(by_cat={}, by_order={})
        stub = SimpleNamespace(knowledge=view, taxonomy=tax)

        out = _call_retrieval(stub, A, q, category='N', order=0,
                              topk_content=3, topk_heat=3)

        assert 5 in out['rows'].tolist(), (
            "Hot-but-distant row 5 must appear via the heat branch of union")

    def test_hot_distant_in_union_typed_admits_it(self):
        """Hot distant row reaches candidate union when typed set admits it."""
        A, q, K, D = self._setup()
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=K, live=K)
        tax.prime([5], batch=0, boost=10.0)
        # Typed set includes rows {0, 5} so intersection will keep row 5.
        admitted = torch.tensor([0, 5], dtype=torch.long)
        view = FakeView(by_cat={'V': admitted}, by_order={1: admitted})
        stub = SimpleNamespace(knowledge=view, taxonomy=tax)

        out = _call_retrieval(stub, A, q, category='V', order=1,
                              topk_content=3, topk_heat=3)

        assert 5 in out['rows'].tolist(), (
            "Hot-but-distant row 5 must survive typed intersection")


# ---------------------------------------------------------------------------
# Test 7: typed mask excludes hot-but-inadmissible rows; priming invariants
# ---------------------------------------------------------------------------

class TestTypedMaskExcludesHotInadmissible:
    """Plan test 7: a hot row NOT in ``C_typed`` must be excluded; rows in
    ``C_typed`` ∩ union must be included.  Also verify the priming vector
    invariants:

    * non-candidate rows have ``priming[i] == 1.0``;
    * at least one candidate has ``priming[i] > 1.0``.

    Setup
    -----
    K=10 rows, D=4.
    * Row 3 is HOT (primed to 2.0) but NOT in the typed admissible set.
    * Row 1 is warm (primed to 1.5) AND in the typed admissible set.
    * Row 0 is content-nearest AND in the typed admissible set (via query).
    * All other rows are neutral.
    """

    def _setup(self):
        torch.manual_seed(99)
        K, D = 10, 4
        A = torch.randn(K, D)
        A = A / A.norm(dim=1, keepdim=True).clamp(min=1e-6)
        # q aligns with row 0 -> content nearest.
        q = A[0].clone()
        return A, q, K, D

    def test_inadmissible_hot_excluded_from_rows(self):
        """Hot inadmissible row 3 must not appear in out['rows']."""
        A, q, K, D = self._setup()
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=K, live=K)
        # Row 3: hot but NOT in typed set.
        tax.prime([3], batch=0, boost=5.0)
        # Row 1: warm and in typed set.
        tax.prime([1], batch=0, boost=0.5)
        # Typed set: admits only rows {0, 1, 2, 7} — row 3 excluded.
        admitted = torch.tensor([0, 1, 2, 7], dtype=torch.long)
        view = FakeView(by_cat={'N': admitted}, by_order={0: admitted})
        stub = SimpleNamespace(knowledge=view, taxonomy=tax)

        out = _call_retrieval(stub, A, q, category='N', order=0,
                              topk_content=5, topk_heat=5)

        rows_list = out['rows'].tolist()
        assert 3 not in rows_list, (
            f"Hot-inadmissible row 3 must be excluded; got rows={rows_list}")
        # Admitted row 0 (content-nearest) must be present.
        assert 0 in rows_list, (
            f"Admitted content-nearest row 0 must be in rows; got {rows_list}")

    def test_priming_identity_for_non_candidates(self):
        """Non-candidate rows must have priming[i] == 1.0."""
        A, q, K, D = self._setup()
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=K, live=K)
        tax.prime([3], batch=0, boost=5.0)
        tax.prime([1], batch=0, boost=0.5)
        admitted = torch.tensor([0, 1, 2, 7], dtype=torch.long)
        view = FakeView(by_cat={'N': admitted}, by_order={0: admitted})
        stub = SimpleNamespace(knowledge=view, taxonomy=tax)

        out = _call_retrieval(stub, A, q, category='N', order=0,
                              topk_content=5, topk_heat=5)

        candidate_set = set(out['rows'].tolist())
        priming = out['priming']
        assert priming.shape == (K,), f"Expected shape ({K},), got {priming.shape}"
        for i in range(K):
            if i not in candidate_set:
                assert abs(priming[i].item() - 1.0) < 1e-6, (
                    f"Non-candidate row {i} must have priming==1.0, "
                    f"got {priming[i].item()}")

    def test_priming_above_unity_for_at_least_one_candidate(self):
        """At least one admitted hot/near candidate must have priming > 1.0."""
        A, q, K, D = self._setup()
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=K, live=K)
        tax.prime([3], batch=0, boost=5.0)
        tax.prime([1], batch=0, boost=0.5)
        # Admitted warm row 1 plus content-near row 0.
        admitted = torch.tensor([0, 1, 2, 7], dtype=torch.long)
        view = FakeView(by_cat={'N': admitted}, by_order={0: admitted})
        stub = SimpleNamespace(knowledge=view, taxonomy=tax)

        out = _call_retrieval(stub, A, q, category='N', order=0,
                              topk_content=5, topk_heat=5)

        priming = out['priming']
        candidate_set = set(out['rows'].tolist())
        assert len(candidate_set) > 0, "Expected non-empty candidate set"
        max_cand_priming = max(priming[i].item() for i in candidate_set)
        assert max_cand_priming > 1.0, (
            f"Expected at least one candidate with priming > 1.0, "
            f"got max={max_cand_priming}")


# ---------------------------------------------------------------------------
# Graceful fallback: knowledge=None or basis=None
# ---------------------------------------------------------------------------

class TestRetrievalGracefulFallback:
    """Extra test: ``retrieval_candidates_for_slot`` returns ``{}`` when
    ``knowledge`` is ``None``, ``basis`` is ``None``, or ``getW()`` returns
    ``None``.  Mirrors ``priming_kwargs_for_slots``'s graceful ``{}``."""

    def _make_basis(self, K=8, D=4):
        A = torch.randn(K, D)
        return SimpleNamespace(getW=lambda: A)

    def _make_view(self):
        return FakeView(by_cat={}, by_order={})

    def _make_query(self, D=4):
        return torch.randn(D)

    def test_returns_empty_when_knowledge_none(self):
        """knowledge=None -> {}."""
        stub = SimpleNamespace(knowledge=None, taxonomy=None)
        basis = self._make_basis()
        out = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=self._make_query(), basis=basis,
            category='N', order=0)
        assert out == {}

    def test_returns_empty_when_basis_none(self):
        """basis=None -> {}."""
        tax = Taxonomy()
        stub = SimpleNamespace(knowledge=self._make_view(), taxonomy=tax)
        out = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=self._make_query(), basis=None,
            category='N', order=0)
        assert out == {}

    def test_returns_empty_when_getW_returns_none(self):
        """basis.getW() returning None -> {}."""
        tax = Taxonomy()
        stub = SimpleNamespace(knowledge=self._make_view(), taxonomy=tax)
        basis = SimpleNamespace(getW=lambda: None)
        out = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=self._make_query(), basis=basis,
            category='N', order=0)
        assert out == {}

    def test_returns_empty_when_basis_has_no_getW(self):
        """basis with no getW attribute -> {}."""
        tax = Taxonomy()
        stub = SimpleNamespace(knowledge=self._make_view(), taxonomy=tax)
        basis = SimpleNamespace()   # no getW
        out = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=self._make_query(), basis=basis,
            category='N', order=0)
        assert out == {}


# ---------------------------------------------------------------------------
# Fix 5: typed_only fallback when content/heat union is empty
# ---------------------------------------------------------------------------

class TestTypedOnlyFallback:
    """Fix 5: when topk_content=0 and topk_heat=0, the content/heat union is
    empty.  If C_typed is non-empty the method must fall back to typed_only
    and return those ids as ``out['rows']``."""

    def test_typed_only_fallback(self):
        """topk_content=0, topk_heat=0 with non-empty C_typed -> typed_only."""
        torch.manual_seed(13)
        K, D = 10, 6
        A = torch.randn(K, D)
        A = A / A.norm(dim=1, keepdim=True).clamp(min=1e-6)
        q = torch.randn(D)

        # C_typed is the intersection of category ∩ order sets.
        # FakeView returns the same ids for both so the intersection equals them.
        typed_ids = torch.tensor([1, 4, 7], dtype=torch.long)
        view = FakeView(by_cat={'N': typed_ids}, by_order={0: typed_ids})

        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=K, live=K)
        # Priming is neutral (1.0) — no heat
        stub = SimpleNamespace(knowledge=view, taxonomy=tax)

        basis = SimpleNamespace(getW=lambda: A)
        out = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=q, basis=basis,
            category='N', order=0,
            topk_content=0, topk_heat=0)

        assert 'rows' in out, "Expected 'rows' key in output"
        assert out['diagnostics']['fallback'] == 'typed_only', (
            f"Expected fallback='typed_only', got {out['diagnostics']['fallback']}")
        assert set(out['rows'].tolist()) == set(typed_ids.tolist()), (
            f"Expected rows == typed_ids {typed_ids.tolist()}, "
            f"got {out['rows'].tolist()}")


# ---------------------------------------------------------------------------
# Fix 6: diagnostics consistency on a normal (non-fallback) path
# ---------------------------------------------------------------------------

class TestDiagnosticsConsistency:
    """Fix 6: on a normal (non-fallback) retrieval path, assert that
    ``out['diagnostics']['n_candidates'] == out['rows'].numel()`` and
    ``out['diagnostics']['fallback'] == 'none'``."""

    def test_diagnostics_consistent_on_normal_path(self):
        """n_candidates matches rows.numel() and fallback=='none' normally."""
        torch.manual_seed(55)
        K, D = 12, 8
        A = torch.randn(K, D)
        A = A / A.norm(dim=1, keepdim=True).clamp(min=1e-6)
        q = A[0].clone()  # query aligns with row 0

        # Typed set admits rows {0, 1, 2, 3} so the union∩typed is non-empty
        admitted = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        view = FakeView(by_cat={'N': admitted}, by_order={0: admitted})

        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=K, live=K)
        stub = SimpleNamespace(knowledge=view, taxonomy=tax)

        basis = SimpleNamespace(getW=lambda: A)
        out = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=q, basis=basis,
            category='N', order=0,
            topk_content=4, topk_heat=4)

        diag = out['diagnostics']
        assert diag['fallback'] == 'none', (
            f"Expected fallback='none', got {diag['fallback']}")
        assert diag['n_candidates'] == out['rows'].numel(), (
            f"n_candidates={diag['n_candidates']} != rows.numel()={out['rows'].numel()}")


# ---------------------------------------------------------------------------
# Test 8: unity priming is byte-identical to no priming (Phase 3b)
# ---------------------------------------------------------------------------

class TestUnityPrimingByteIdentical:
    """Plan test 8: passing all-1.0 priming must produce the exact same
    (x1, x2) pair as passing no priming (None).

    Unity is the multiplicative identity for the priming boost, so the
    algorithms are provably equivalent.  This test exercises:
      - ``Ops.disjunctionReverse``
      - ``Ops.conjunctionReverse``
    at the Ops level (the Layer-level wrappers just forward kwargs, so
    Ops-level testing covers the substance).

    This validates the 'default-off' contract of Phase 3b:
    doc/plans/2026-06-06-symbolic-heat-retrieval.md §Phase 3b.
    """

    def _random_W_and_result(self, K=6, D=3, seed=0):
        torch.manual_seed(seed)
        W = torch.rand(K, D) * 0.8 + 0.1   # all positive, well-scaled
        result = torch.rand(1, 1, D) * 0.6 + 0.2   # shape [B=1, N=1, D]
        return W, result

    def test_disjunction_none_equals_unity(self):
        """disjunctionReverse with no priming == all-1.0 priming."""
        W, result = self._random_W_and_result(seed=10)
        K = W.shape[0]
        ones = torch.ones(K)
        x1_none, x2_none = Ops.disjunctionReverse(result, result, W)
        x1_ones, x2_ones = Ops.disjunctionReverse(
            result, result, W,
            left_priming=ones, right_priming=ones)
        assert torch.equal(x1_none, x1_ones), (
            f"x1 differs: {x1_none} vs {x1_ones}")
        assert torch.equal(x2_none, x2_ones), (
            f"x2 differs: {x2_none} vs {x2_ones}")

    def test_conjunction_none_equals_unity(self):
        """conjunctionReverse with no priming == all-1.0 priming."""
        W, result = self._random_W_and_result(seed=20)
        K = W.shape[0]
        ones = torch.ones(K)
        # For conjunction, result elements must be <= 1.0 and W must be >= result.
        # Scale W up to be feasible (>= result).
        W_feas = W.clamp(min=float(result.max().item()))
        x1_none, x2_none = Ops.conjunctionReverse(result, result, W_feas)
        x1_ones, x2_ones = Ops.conjunctionReverse(
            result, result, W_feas,
            left_priming=ones, right_priming=ones)
        assert torch.equal(x1_none, x1_ones), (
            f"x1 differs: {x1_none} vs {x1_ones}")
        assert torch.equal(x2_none, x2_ones), (
            f"x2 differs: {x2_none} vs {x2_ones}")

    def test_disjunction_none_rows_equals_unity_rows(self):
        """None priming with rows restriction == unity priming with same rows."""
        W, result = self._random_W_and_result(seed=30)
        K = W.shape[0]
        rows = torch.tensor([0, 2, 4], dtype=torch.long)
        ones = torch.ones(K)
        x1_none, x2_none = Ops.disjunctionReverse(
            result, result, W, left_rows=rows, right_rows=rows)
        x1_ones, x2_ones = Ops.disjunctionReverse(
            result, result, W,
            left_rows=rows, right_rows=rows,
            left_priming=ones, right_priming=ones)
        assert torch.equal(x1_none, x1_ones)
        assert torch.equal(x2_none, x2_ones)


# ---------------------------------------------------------------------------
# Test 9: heat-boosted priming shifts operand pick toward favored row
# ---------------------------------------------------------------------------

class TestHeatContentPrefersExpectedCandidate:
    """Plan test 9: a boosted priming vector must make disjunctionReverse
    select the primed row (for x1 / the 'argmax' direction), rather than
    the un-primed winner selected without priming.

    Construction mirrors ``test_priming_lifts_argmax_choice_union_x1`` in
    test_primed_reverse_combined.py:
      - W has two rows that are both feasible (both <= result element-wise).
      - Row 0 has a slightly smaller norm → loses the unprimed argmax.
      - Row 1 has a slightly larger norm → wins unprimed.
      - Boost row 0 by factor 2 so its primed score > row 1's un-primed score.
      - Assert x1_unprimed == W[1] and x1_primed == W[0].

    This validates the path
      retrieval_candidates_for_slot → priming vector → disjunctionReverse
    returns the hot candidate (the simulated Phase-4 output of forward heat).

    doc/plans/2026-06-06-symbolic-heat-retrieval.md §Phase 3b test 9.
    """

    def test_primed_row_wins_over_unprimed_in_disjunction(self):
        """Hot-primed row 0 beats the otherwise-winning row 1 in disjunctionReverse."""
        # Row 0: norm ≈ 0.50, slightly smaller than row 1 but still <= result.
        # Row 1: norm ≈ 0.515, slightly larger, unprimed winner.
        W = torch.tensor([
            [0.40, 0.30],   # norm ≈ 0.500
            [0.41, 0.31],   # norm ≈ 0.515
        ])
        result = torch.tensor([[[0.5, 0.4]]])

        # Without priming: row 1's larger norm wins the argmax-≤-result step.
        x1_un, _ = Ops.disjunctionReverse(result, result, W)
        assert torch.allclose(x1_un[0, 0], W[1]), (
            "Sanity: unprimed pick should be row 1 (larger norm)")

        # Build priming that boosts row 0 to score 0.50 * 2.0 = 1.00,
        # which beats row 1's 0.515 * 1.0 = 0.515.
        left_priming = torch.tensor([2.0, 1.0])
        x1_pr, _ = Ops.disjunctionReverse(
            result, result, W, left_priming=left_priming)
        assert torch.allclose(x1_pr[0, 0], W[0]), (
            f"Expected primed pick to be W[0]={W[0]}, got {x1_pr[0,0]}")

    def test_primed_row_wins_over_unprimed_in_conjunction(self):
        """Hot-primed row 0 beats otherwise-winning row 1 in conjunctionReverse.

        For conjunction the inverse is argmin-≥-result (smallest feasible row);
        priming reduces effective norm by dividing, so a boosted row with
        a slightly larger raw norm can be selected over the un-primed winner.
        """
        W = torch.tensor([
            [0.80, 0.70],   # norm ≈ 1.063, slightly larger
            [0.75, 0.65],   # norm ≈ 0.991, smaller → wins unprimed
        ])
        result = torch.tensor([[[0.7, 0.6]]])

        # Without priming: row 1 (smaller norm) wins.
        x1_un, _ = Ops.conjunctionReverse(result, result, W)
        assert torch.allclose(x1_un[0, 0], W[1]), (
            "Sanity: unprimed pick should be row 1 (smaller norm)")

        # Priming row 0 with factor 2: effective score = 1.063 / 2.0 = 0.5315
        # vs row 1's 0.991 / 1.0 = 0.991 → primed row 0 now has smaller
        # effective norm → argmin selects it.
        left_priming = torch.tensor([2.0, 1.0])
        x1_pr, _ = Ops.conjunctionReverse(
            result, result, W, left_priming=left_priming)
        assert torch.allclose(x1_pr[0, 0], W[0]), (
            f"Expected primed pick to be W[0]={W[0]}, got {x1_pr[0,0]}")

    def test_candidate_from_retrieval_boosts_expected_row(self):
        """End-to-end: retrieval_candidates_for_slot produces a priming
        vector that, when fed to disjunctionReverse, selects the hot row
        over the content-nearest row.

        Setup (deterministic):
          - Codebook W: 3 rows, D=3.
          - Row 0: content-nearest (query = row 0); slightly larger norm →
            unprimed argmax winner.
          - Row 1: cold in taxonomy, slightly smaller norm → unprimed loser.
          - Row 2: very small norm, unrelated.
          - taxonomy: row 1 is hot (boost=1.0, so priming=2.0+).
          - retrieval_candidates_for_slot: topk_content=1 (row 0),
            topk_heat=1 (row 1).
          - The returned priming vector assigns row 1 > row 0 (heat boost
            outweighs content).
          - With priming: row 1's effective score > row 0's → row 1 wins.
          - Assert unprimed pick == row 0, primed pick == row 1.
        """
        K, D = 3, 3
        # Row 0: norm ≈ 0.500 (largest feasible), content-nearest.
        # Row 1: norm ≈ 0.475 (slightly smaller), heat candidate.
        # Row 2: small, cold, irrelevant.
        W = torch.tensor([
            [0.40, 0.30, 0.0],   # norm ≈ 0.500
            [0.38, 0.28, 0.0],   # norm ≈ 0.475
            [0.10, 0.10, 0.0],   # norm ≈ 0.141
        ])
        q = W[0].clone()   # query aligns exactly with row 0

        # result is above all rows element-wise → all feasible for x1.
        result = torch.tensor([[[0.5, 0.4, 0.1]]])

        # Verify sanity: unprimed picks row 0 (largest norm <= result).
        x1_un, _ = Ops.disjunctionReverse(result, result, W)
        assert torch.allclose(x1_un[0, 0], W[0], atol=1e-5), (
            "Sanity: unprimed pick must be row 0 (largest norm)")

        # Set up taxonomy: row 1 hot (priming → 2.0).
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=K, live=K)
        tax.prime([1], batch=0, boost=1.0)   # priming[1] = 2.0

        view = FakeView(by_cat={}, by_order={})
        stub = SimpleNamespace(knowledge=view, taxonomy=tax)

        out = _call_retrieval(stub, W, q, category='N', order=0,
                              topk_content=1, topk_heat=1)
        priming_K = out['priming']   # shape [K]
        assert priming_K.shape == (K,), f"Expected shape ({K},), got {priming_K.shape}"
        # Hot row 1 must have priming > row 0's priming so it wins x1.
        assert priming_K[1].item() > priming_K[0].item(), (
            f"Expected priming[1] > priming[0]; got "
            f"priming={priming_K.tolist()}")

        # With the retrieved priming, row 1 should win (score = norm * priming).
        x1_pr, _ = Ops.disjunctionReverse(result, result, W, left_priming=priming_K)
        assert torch.allclose(x1_pr[0, 0], W[1], atol=1e-5), (
            f"Expected primed pick to be W[1]={W[1].tolist()}, "
            f"got {x1_pr[0, 0].tolist()} (priming={priming_K.tolist()})")


# ---------------------------------------------------------------------------
# Phase 4: forward-commit heat updates (plan §Forward-path responsibilities,
# plan tests 8 byte-identical-OFF and 11 sentence-reset).
#
# The forward commit site is ``SymbolicSubSpace.push`` (word/percept commit) and
# ``LanguageLayer.reduce`` (C-tier grammar reduction). Both delegate the gated
# heat update to ``SymbolicSubSpace._commit_priming(b, ref_id)`` so the
# ``priming_enabled``-first guard lives in exactly one place. These tests
# exercise that helper unbound on a duck-typed stub (same pattern as
# ``_call_retrieval`` above) so no full ``SymbolicSubSpace`` construction is needed.
# ---------------------------------------------------------------------------

class AdjacencyView:
    """Duck-typed KnowledgeView exposing only the parent/children adjacency
    that ``Taxonomy.propagate`` walks (``parent_of`` / ``children_of`` /
    ``n_refs_live``). A tiny two-level taxonomy: ``parent`` maps child->parent,
    children is derived. Mirrors ``embed.KnowledgeView``'s propagation surface.
    """

    def __init__(self, parent, n_refs_live):
        self._parent = dict(parent)        # dict[int, int] child -> parent
        self._n = int(n_refs_live)
        self._children = {}
        for c, p in self._parent.items():
            self._children.setdefault(int(p), []).append(int(c))

    @property
    def n_refs_live(self):
        return self._n

    def parent_of(self, ref_id):
        p = self._parent.get(int(ref_id))
        return None if p is None else int(p)

    def children_of(self, ref_id):
        kids = self._children.get(int(ref_id), [])
        return torch.as_tensor(kids, dtype=torch.long)


def _commit(stub, b, ref_id):
    """Invoke ``SymbolicSubSpace._commit_priming`` unbound on a duck-typed stub."""
    return SymbolicSubSpace._commit_priming(stub, b, ref_id)


class TestSymbolicPrimingOffIsNoop:
    """Plan test 8 (byte-identical OFF) for the forward commit path.

    Exercises the off-path of ``_commit_priming``: committing a ref when
    ``priming_enabled=False`` must NOT touch ``_priming`` — it stays at unity
    (all 1.0), so ``heat_mask`` is all-zero.  This mirrors what a production
    SymbolicSubSpace taxonomy looks like: ``attach_knowledge`` calls
    ``configure_priming(priming_enabled=<symbolicPriming>)`` (default False),
    so priming is off unless the XML flag is set.  A bare ``Taxonomy()``
    class default is True (for the retrieval helpers), so the tests call
    ``configure_priming(priming_enabled=False)`` explicitly to exercise this
    off-path.  The training-path zero-cost guarantee: ``prime``/``propagate``
    (the expensive host-side walk) never run.
    """

    def test_off_leaves_priming_at_unity(self):
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=8, live=6)
        # Bare Taxonomy() class default is True; explicitly disable to mirror
        # what attach_knowledge sets from <symbolicPriming> absent/false.
        tax.configure_priming(priming_enabled=False)
        view = AdjacencyView(parent={0: 4, 1: 4}, n_refs_live=6)
        tax.attach_view(view)
        stub = SimpleNamespace(taxonomy=tax)

        before = tax._priming.clone()
        _commit(stub, b=0, ref_id=0)

        assert torch.equal(tax._priming, before), (
            "OFF path must not mutate _priming")
        assert tax._priming.eq(1.0).all(), "Priming must stay at unity (1.0)"
        hm = tax.heat_mask(batch=0)
        assert hm.eq(0).all(), f"heat_mask must be all-zero when OFF, got {hm}"

    def test_off_no_op_even_for_valid_ref(self):
        """A perfectly valid ref id is still a no-op when the flag is off."""
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=8, live=6)
        tax.configure_priming(priming_enabled=False)
        stub = SimpleNamespace(taxonomy=tax)
        _commit(stub, b=0, ref_id=3)
        assert tax.heat_mask(batch=0).eq(0).all()


class TestSymbolicPrimingOnMakesRefHot:
    """Plan §Word/percept commit: with priming ON, committing ref ``r`` makes
    ``heat_mask()[r] > 0`` and a taxonomic NEIGHBOR of ``r`` also becomes hot
    via ``propagate``.
    """

    def test_committed_ref_is_hot(self):
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=8, live=6)
        tax.configure_priming(priming_enabled=True)
        stub = SimpleNamespace(taxonomy=tax)
        _commit(stub, b=0, ref_id=2)
        hm = tax.heat_mask(batch=0)
        assert hm[2].item() > 0.0, f"Committed ref must be hot, heat={hm}"

    def test_neighbor_becomes_hot_via_propagate(self):
        # Two-level taxonomy: refs 0 and 1 are children of parent ref 4.
        # Committing ref 0 should prime 0 (direct) and spread to its parent 4
        # (one hop) and to sibling 1 (two hops, via the shared parent).
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=8, live=6)
        tax.configure_priming(priming_enabled=True)
        view = AdjacencyView(parent={0: 4, 1: 4}, n_refs_live=6)
        tax.attach_view(view)
        stub = SimpleNamespace(taxonomy=tax)

        _commit(stub, b=0, ref_id=0)
        hm = tax.heat_mask(batch=0)
        assert hm[0].item() > 0.0, "Seed ref 0 must be hot"
        assert hm[4].item() > 0.0, (
            f"Parent ref 4 must be hot via propagate, heat={hm}")

    def test_negative_ref_is_noop_when_on(self):
        """ref_id < 0 (byte fallback / no codebook id) must short-circuit even
        when priming is enabled."""
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=8, live=6)
        tax.configure_priming(priming_enabled=True)
        stub = SimpleNamespace(taxonomy=tax)
        _commit(stub, b=0, ref_id=-1)
        assert tax.heat_mask(batch=0).eq(0).all(), (
            "Negative ref must not prime anything")


class TestSentenceResetClearsHeat:
    """Plan test 11: after priming several refs, ``taxonomy.reset()`` returns
    ``_priming`` to all-1.0 (heat_mask all-zero) and the derived
    ``build_semantic_heat`` then returns zeros.
    """

    def test_reset_clears_row_heat_and_semantic(self):
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=8, live=6)
        tax.configure_priming(priming_enabled=True)
        view = AdjacencyView(parent={0: 4, 1: 4}, n_refs_live=6)
        tax.attach_view(view)
        stub = SimpleNamespace(taxonomy=tax)

        # Prime several refs.
        for rid in (0, 1, 2):
            _commit(stub, b=0, ref_id=rid)
        assert tax.heat_mask(batch=0).gt(0).any(), "Pre-reset: something hot"

        # A codebook to derive semantic heat from.
        torch.manual_seed(0)
        A = torch.randn(6, 5)
        z_pre = tax.build_semantic_heat(A, batch=0)
        assert z_pre.abs().sum().item() > 0.0, "Pre-reset semantic heat nonzero"

        tax.reset()

        assert tax._priming.eq(1.0).all(), "reset must restore unity"
        assert tax.heat_mask(batch=0).eq(0).all(), "Post-reset heat all-zero"
        z_post = tax.build_semantic_heat(A, batch=0)
        assert torch.equal(z_post, torch.zeros_like(z_post)), (
            f"Post-reset semantic heat must be zeros, got {z_post}")


class TestConfigurePrimingFromSymbolicPriming:
    """Plan §Configuration: ``configure_priming(priming_enabled=...)`` flips the
    instance flag, and the forward gate respects it. Models the
    ``<symbolicPriming>`` true/false wiring at the buffer level.
    """

    def test_flag_flips_and_gate_respects_it(self):
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=8, live=6)
        stub = SimpleNamespace(taxonomy=tax)

        # OFF: no heat.
        tax.configure_priming(priming_enabled=False)
        assert tax.priming_enabled is False
        _commit(stub, b=0, ref_id=2)
        assert tax.heat_mask(batch=0).eq(0).all(), "OFF gate must be no-op"

        # ON: heat appears.
        tax.configure_priming(priming_enabled=True)
        assert tax.priming_enabled is True
        _commit(stub, b=0, ref_id=2)
        assert tax.heat_mask(batch=0)[2].item() > 0.0, "ON gate must prime"

    def test_missing_taxonomy_is_safe(self):
        """The gate tolerates a SymbolicSubSpace with no taxonomy / no buffer."""
        # No 'taxonomy' attribute at all.
        _commit(SimpleNamespace(), b=0, ref_id=2)
        # taxonomy present but priming unallocated (_priming is None).
        tax = Taxonomy()
        _commit(SimpleNamespace(taxonomy=tax), b=0, ref_id=2)


class TestReduceCommitPrimesChildRefs:
    """Plan §Grammar reduction (best-effort): when the C-tier reduction is
    given the source child ref_ids, the same gated prime+propagate runs for
    those committed child refs when priming is enabled, and is a no-op when
    disabled. The ref_ids are supplied by the caller (the reduce site has no
    ref ids in the dense ``.what`` stack itself — see report).
    """

    def test_child_refs_primed_when_on(self):
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=8, live=6)
        tax.configure_priming(priming_enabled=True)
        stub = SimpleNamespace(taxonomy=tax)
        # Simulate the reduce site committing two child refs.
        for rid in (1, 3):
            _commit(stub, b=0, ref_id=rid)
        hm = tax.heat_mask(batch=0)
        assert hm[1].item() > 0.0 and hm[3].item() > 0.0, (
            f"Both committed child refs must be hot, heat={hm}")

    def test_child_refs_noop_when_off(self):
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=8, live=6)
        tax.configure_priming(priming_enabled=False)
        stub = SimpleNamespace(taxonomy=tax)
        for rid in (1, 3):
            _commit(stub, b=0, ref_id=rid)
        assert tax.heat_mask(batch=0).eq(0).all(), (
            "Reduce-site priming must be a no-op when OFF")


# ===========================================================================
# Phase 5 tests (plan §Phase 5 / §Tests items 12, and new carrier tests)
# Plan: doc/plans/2026-06-06-symbolic-heat-retrieval.md
# ===========================================================================


# ---------------------------------------------------------------------------
# Test D1: primer mode is byte-identical to pre-Phase-5 (no carrier terms)
# ---------------------------------------------------------------------------

class TestPrimerModeByteIdentical:
    """Plan Phase-5 test D1: retrieval_candidates_for_slot(..., mode='primer')
    and the bare default call must produce a ``priming`` vector element-equal
    to ``exp(alpha*sim + beta*log1p(r))`` (carrier OFF).  Verified against:
    (a) an explicit reference constructed from the formula, and
    (b) the default (no mode kwarg) call — both must be bit-for-bit equal.
    """

    def _setup(self):
        torch.manual_seed(2025)
        K, D = 12, 8
        A = torch.randn(K, D)
        A = A / A.norm(dim=1, keepdim=True).clamp(min=1e-6)
        q = A[2].clone()
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=K, live=K)
        tax.configure_priming(priming_enabled=True)
        tax.prime([0, 3, 7], batch=0, boost=1.5)
        view = FakeView(by_cat={}, by_order={})
        stub = SimpleNamespace(knowledge=view, taxonomy=tax)
        basis = SimpleNamespace(getW=lambda: A)
        return stub, A, q, K, D, basis

    def test_primer_equals_reference_formula(self):
        """mode='primer' priming == exp(alpha*sim + beta*log1p(r)) exactly."""
        stub, A, q, K, D, basis = self._setup()
        import torch.nn.functional as F
        alpha, beta = 1.0, 0.5
        out = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=q, basis=basis,
            category='N', order=0,
            alpha=alpha, beta=beta,
            mode='primer')
        priming_primer = out['priming']
        rows = out['rows']
        # Build the reference priming for candidates only
        sim = F.cosine_similarity(q.reshape(1, -1), A, dim=1)
        heat = stub.taxonomy.heat_mask(batch=0)
        for idx in rows.tolist():
            if idx >= K:
                continue
            r_i = heat[idx].item() if idx < heat.numel() else 0.0
            expected = torch.exp(
                torch.tensor(alpha * float(sim[idx]) + beta * float(torch.log1p(torch.tensor(r_i)))))
            assert abs(priming_primer[idx].item() - expected.item()) < 1e-5, (
                f"Row {idx}: primer priming={priming_primer[idx].item():.6f} "
                f"!= formula={expected.item():.6f}")

    def test_primer_equals_default_call(self):
        """Default call (no mode kwarg) is element-equal to mode='primer'."""
        stub, A, q, K, D, basis = self._setup()
        out_default = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=q, basis=basis, category='N', order=0)
        out_primer = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=q, basis=basis, category='N', order=0, mode='primer')
        assert torch.equal(out_default['priming'], out_primer['priming']), (
            "Default call priming must be element-equal to mode='primer'")

    def test_primer_mode_with_nonzero_gamma_delta_ignored(self):
        """mode='primer' must ignore gamma/delta even if passed (byte-identical)."""
        stub, A, q, K, D, basis = self._setup()
        out_primer = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=q, basis=basis, category='N', order=0,
            mode='primer', gamma=5.0, delta=5.0)
        out_default = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=q, basis=basis, category='N', order=0)
        # mode='primer' must clamp gamma/delta to zero internally (spec)
        assert torch.equal(out_primer['priming'], out_default['priming']), (
            "primer mode must ignore passed gamma/delta — priming must match default")


# ---------------------------------------------------------------------------
# Test D2: low-rank carrier == dense carrier (mirrors Phase-1 test 4)
# ---------------------------------------------------------------------------

class TestLowRankCarrierMatchesDense:
    """Plan Phase-5 test D2: the low-rank delta contribution
    ``A[C] @ (U^T @ (U @ q))`` must equal the second-order dense
    contribution ``A[C] @ (C_mat @ q)`` within tolerance.
    Mirrors Phase-1 test 4 (TestBuildOuterHeat.test_lowrank_U_times_q_equals_dense_C_times_q)
    at the retrieval level.
    """

    def test_lowrank_delta_equals_dense_delta(self):
        """low-rank and second-order delta contributions agree.

        Both paths build the same carrier contribution ``A[C] @ (C @ q)`` --
        low-rank factors ``C = U^T U`` so it computes ``A[C] @ (U^T (U q))`` --
        and feed it through the SAME ``exp(...)`` weighting. The pre-``exp``
        carrier dot-products agree to ~float32 eps; the comparison below is on
        the post-``exp`` ``priming`` weights, which are exp-AMPLIFIED (here they
        span ~7e-3 .. ~8e2). A fixed *absolute* tolerance is therefore the wrong
        gauge -- a float32-eps relative disagreement on a weight of ~8e2 is an
        absolute diff of ~4e-4, which would spuriously fail a 1e-4 atol while the
        relative error stays ~5e-7. Compare with a RELATIVE tolerance instead so
        the check tracks the genuine (relative) low-rank==dense agreement and is
        independent of the weights' magnitude. Seeded so it is order-independent.
        """
        torch.manual_seed(31415)
        K, D = 8, 5
        A = torch.randn(K, D)
        q = torch.randn(D)
        q = q / q.norm()
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=K, live=K)
        tax.configure_priming(priming_enabled=True)
        tax.prime([0, 2, 4], batch=0, boost=1.2)
        view = FakeView(by_cat={}, by_order={})
        stub = SimpleNamespace(knowledge=view, taxonomy=tax)
        basis = SimpleNamespace(getW=lambda: A)

        out_lr = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=q, basis=basis, category='N', order=0,
            alpha=0.0, beta=0.0,  # isolate delta contribution
            mode='low-rank', gamma=0.0, delta=1.0, outer_topk=8)
        out_so = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=q, basis=basis, category='N', order=0,
            alpha=0.0, beta=0.0,
            mode='second-order', gamma=0.0, delta=1.0, outer_topk=8)

        # Compare only the rows that appear in both outputs
        rows_lr = set(out_lr['rows'].tolist())
        rows_so = set(out_so['rows'].tolist())
        common = sorted(rows_lr & rows_so)
        assert len(common) > 0, "No common rows between low-rank and second-order"
        idx_t = torch.tensor(common, dtype=torch.long)
        lr_vals = out_lr['priming'][idx_t]
        so_vals = out_so['priming'][idx_t]
        # RELATIVE tolerance (rtol=1e-4) with a small atol floor for the
        # near-zero weights: this is the right gauge for exp-amplified values
        # (see docstring). The genuine relative agreement is ~5e-7..9e-7, so
        # rtol=1e-4 keeps ~100x of headroom while still being a real
        # low-rank==dense check, not a vacuous one.
        assert torch.allclose(lr_vals, so_vals, rtol=1e-4, atol=1e-5), (
            "low-rank vs second-order priming disagree beyond relative "
            "tolerance:\n"
            + "\n".join(
                f"  row {i}: low-rank={lv:.6f} second-order={sv:.6f} "
                f"reldiff={abs(lv - sv) / max(abs(sv), 1e-12):.3e}"
                for i, lv, sv in zip(common,
                                     lr_vals.tolist(), so_vals.tolist())))


# ---------------------------------------------------------------------------
# Test D3: carrier actually changes weights (non-vacuous)
# ---------------------------------------------------------------------------

class TestCarrierChangesWeight:
    """Plan Phase-5 test D3: with gamma/delta>0 and non-trivial heat, at
    least one candidate's weight differs from the primer-mode weight.
    This confirms the carrier terms are not dead code.
    """

    def test_gamma_changes_weight(self):
        """A non-zero gamma causes at least one candidate to differ from primer."""
        torch.manual_seed(7777)
        K, D = 10, 6
        A = torch.randn(K, D)
        q = torch.randn(D); q = q / q.norm()
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=K, live=K)
        tax.configure_priming(priming_enabled=True)
        tax.prime([0, 1, 2], batch=0, boost=2.0)
        view = FakeView(by_cat={}, by_order={})
        stub = SimpleNamespace(knowledge=view, taxonomy=tax)
        basis = SimpleNamespace(getW=lambda: A)

        out_primer = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=q, basis=basis, category='N', order=0,
            mode='primer', gamma=0.0, delta=0.0)
        out_carrier = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=q, basis=basis, category='N', order=0,
            mode='low-rank', gamma=2.0, delta=0.0, outer_topk=K)

        rows = out_primer['rows'].tolist()
        changed = False
        for idx in rows:
            if idx >= K:
                continue
            if abs(out_primer['priming'][idx].item()
                   - out_carrier['priming'][idx].item()) > 1e-5:
                changed = True
                break
        assert changed, (
            "Expected at least one candidate priming to change with gamma>0")

    def test_delta_changes_weight(self):
        """A non-zero delta causes at least one candidate to differ from primer."""
        torch.manual_seed(8888)
        K, D = 10, 6
        A = torch.randn(K, D)
        q = torch.randn(D); q = q / q.norm()
        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=K, live=K)
        tax.configure_priming(priming_enabled=True)
        tax.prime([0, 1, 2], batch=0, boost=2.0)
        view = FakeView(by_cat={}, by_order={})
        stub = SimpleNamespace(knowledge=view, taxonomy=tax)
        basis = SimpleNamespace(getW=lambda: A)

        out_primer = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=q, basis=basis, category='N', order=0,
            mode='primer', gamma=0.0, delta=0.0)
        out_carrier = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=q, basis=basis, category='N', order=0,
            mode='low-rank', gamma=0.0, delta=2.0, outer_topk=K)

        rows = out_primer['rows'].tolist()
        changed = False
        for idx in rows:
            if idx >= K:
                continue
            if abs(out_primer['priming'][idx].item()
                   - out_carrier['priming'][idx].item()) > 1e-5:
                changed = True
                break
        assert changed, (
            "Expected at least one candidate priming to change with delta>0")


# ---------------------------------------------------------------------------
# Test D4: million-row smoke test — no [V,V] allocation
# ---------------------------------------------------------------------------

class TestMillionRowNoVVSmoke:
    """Plan Phase-5 test D4 (plan test 12): build a large codebook, a
    Taxonomy with a few primed rows, and call retrieval_candidates_for_slot
    in both 'primer' and 'low-rank' modes.  The fact that this completes
    without OOM proves no [V,V] tensor was allocated (a [1e6,1e6] fp32
    tensor would be ~4TB and crash immediately).

    D=64 keeps the codebook ~256MB fp32.  If 1_000_000 rows is too slow,
    we fall back to 300_000 with a comment (see below).

    Note: this test is deliberately placed last so it does not slow down
    the rest of the suite.  Mark with pytest.mark.slow if needed.
    """

    def _make_large_codebook(self, V, D):
        """Build a [V, D] random codebook without materializing a [V,V] matrix."""
        # Use chunked randn to avoid a single enormous allocation.
        chunk = 50_000
        rows = []
        for start in range(0, V, chunk):
            end = min(start + chunk, V)
            rows.append(torch.randn(end - start, D))
        A = torch.cat(rows, dim=0)
        # L2-normalize
        norms = A.norm(dim=1, keepdim=True).clamp(min=1e-6)
        A = A / norms
        return A

    def test_large_codebook_primer_and_lowrank_no_vv(self):
        """primer + low-rank both complete on a large codebook (no [V,V])."""
        # V=1_000_000, D=64.  Codebook is 1M*64*4 bytes = 256 MB in RAM —
        # well within budget on this machine.  A [V,V] fp32 matrix would be
        # 1M*1M*4 ≈ 4 TB and crash immediately; completion proves no [V,V]
        # was attempted.  The iCloud path note in the prior comment was
        # incorrect: the codebook is a bare in-RAM torch.randn tensor and
        # involves no iCloud disk I/O.
        V, D = 1_000_000, 64
        A = self._make_large_codebook(V, D)

        tax = Taxonomy()
        tax.allocate_priming(batch_size=1, capacity=V, live=V)
        tax.configure_priming(priming_enabled=True)
        # Prime a tiny fraction of rows.
        hot_rows = list(range(0, min(50, V)))
        tax.prime(hot_rows, batch=0, boost=2.0)

        q = torch.randn(D)
        q = q / q.norm()

        view = FakeView(by_cat={}, by_order={})
        stub = SimpleNamespace(knowledge=view, taxonomy=tax)
        basis = SimpleNamespace(getW=lambda: A)

        # primer mode
        out_p = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=q, basis=basis, category='N', order=0,
            topk_content=64, topk_heat=64,
            mode='primer')
        assert 'rows' in out_p, "primer: expected 'rows' key"
        assert out_p['rows'].numel() > 0, "primer: expected non-empty rows"
        assert out_p['priming'].shape == (V,), (
            f"primer: expected priming shape ({V},), got {out_p['priming'].shape}")

        # low-rank mode with gamma and delta
        out_lr = SymbolicSubSpace.retrieval_candidates_for_slot(
            stub, query=q, basis=basis, category='N', order=0,
            topk_content=64, topk_heat=64,
            mode='low-rank', gamma=0.5, delta=0.5, outer_topk=32)
        assert 'rows' in out_lr, "low-rank: expected 'rows' key"
        assert out_lr['rows'].numel() > 0, "low-rank: expected non-empty rows"
        assert out_lr['priming'].shape == (V,), (
            f"low-rank: expected priming shape ({V},), got {out_lr['priming'].shape}")

        # Both must produce a sane candidate count (>0, << V)
        n_p = out_p['rows'].numel()
        n_lr = out_lr['rows'].numel()
        assert 0 < n_p <= 256, f"primer: unexpected candidate count {n_p}"
        assert 0 < n_lr <= 256, f"low-rank: unexpected candidate count {n_lr}"
