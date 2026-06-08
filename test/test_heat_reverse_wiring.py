"""Phase 3b CAPSTONE: live-ON wiring of heat-biased candidate retrieval into
the grammar reverse-generation driver (``LanguageLayer.unreduce``).

Plan: doc/plans/2026-06-06-symbolic-heat-retrieval.md §Reverse-path
responsibilities, §Phase 3 (reverse recommender integration).

What this pins
--------------
``unreduce`` decodes the top live stack slot's ``.where`` to a binary rule
(here ``union``), then calls the host layer's ``reverse(parent, basis=...)``.
The CAPSTONE adds a GATED restriction: when the owning space's
``attention_mode != 'off'`` it builds typed+heat candidate ``rows`` / boosted
``priming`` (via ``WordSubSpace.retrieval_candidates_for_slot``) and splats
them into the recommender call so the heat steers the operand pick.

Two assertions, exercising the REAL ``unreduce`` path with the REAL
``UnionLayer`` + REAL ``WordSubSpace.retrieval_candidates_for_slot`` + REAL
``Grammar._rule_order_signature``:

  * ON  (``attention_mode='primer'``): with a specific ADMISSIBLE row primed
    HOT, the recovered left operand is that hot+typed row -- the heat path
    steered the pick away from the un-steered baseline.
  * OFF (``attention_mode='off'``): the SAME fixture reproduces the
    un-steered (baseline) pick -- proving the gate (default-off byte-identity
    on the live generation path).

The ``subspace`` is a thin duck-type over the small surface ``unreduce``
needs (``materialize`` / ``set_*`` / ``what`` / ``wordSubSpace``); everything
that carries the heat semantics (the layer, the retrieval helper, the order
signature) is the real production code.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Language import (  # noqa: E402
    Grammar, LanguageLayer, IntersectionLayer, UnionLayer, WordSubSpace,
    Taxonomy,
)


@pytest.fixture(autouse=True)
def _seed_rng():
    """Make every steering test order-independent w.r.t. the global RNG.

    The decisive fixtures here are hardcoded tensors and the heat steering
    path (``WordSubSpace.retrieval_candidates_for_slot``) consumes no RNG, so
    today these tests are deterministic regardless of suite order. This seed is
    defensive: it pins the global RNG so any future use of ``torch.randn`` in a
    fixture or helper cannot make these capstone assertions flaky under
    full-suite ordering. (The real full-suite failure this file guards against
    was a stale global ``TheXMLConfig`` disabling the heat path -- fixed in
    production code in ``LanguageLayer.unreduce`` / ``XMLConfig.space`` -- not
    an RNG-ordering effect.)
    """
    torch.manual_seed(20260606)
    yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeView:
    """Duck-typed KnowledgeView: just the typed-admissibility surface that
    ``retrieval_candidates_for_slot`` reads (``refs_by_category`` /
    ``refs_by_order``). Same shape as the stub in
    test_symbolic_heat_retrieval.py."""

    def __init__(self, by_cat, by_order):
        self._c = by_cat
        self._o = by_order

    def refs_by_category(self, name):
        return self._c.get(name, torch.empty(0, dtype=torch.long))

    def refs_by_order(self, o):
        return self._o.get(int(o), torch.empty(0, dtype=torch.long))


class _Basis:
    """Minimal tier-local Basis: ``getW()`` returns the symbol codebook
    ``W`` ([K, D]); ``unreduce`` passes the whole Basis (not its W) to the
    layer's ``reverse`` and to the retrieval helper, both of which call
    ``getW()``."""

    def __init__(self, W):
        self._W = W

    def getW(self):
        return self._W


class _StackSubSpace:
    """Thin duck-typed stack-mode SubSpace over the exact surface
    ``LanguageLayer.unreduce`` touches: ``materialize(mode)`` for
    what/where/activation, the three ``set_*`` writers, the ``.what`` Basis,
    and the ``.wordSubSpace`` back-ref.

    Single batch row (B=1), K slots, D-wide payloads -- enough to hold one
    binary parent in the top live slot with an empty slot to its right."""

    def __init__(self, what, where, activation, basis, word_sub_space):
        self._what = what
        self._where = where
        self._activation = activation
        self.what = basis                 # tier-local Basis (has getW())
        self.wordSubSpace = word_sub_space

    def materialize(self, mode):
        return {"what": self._what,
                "where": self._where,
                "activation": self._activation}[mode]

    def set_what(self, t):
        self._what = t

    def set_where(self, t):
        self._where = t

    def set_activation(self, t):
        self._activation = t


def _grammar_with_union():
    """Real Grammar holding exactly one binary rule ``LP = union(NP3, VP1)``.

    ``_rule_order_signature`` then yields rhs_categories=('NP','VP'),
    orders (3, 1). ``symbol_vocab_size`` is set so the ``.where`` rule
    encoding ``where_id_for_rule(0) = symbol_vocab_size + 1 + 0`` decodes
    back to ('rule', 0)."""
    g = Grammar()
    g.rules = [g._parse_rule("LP", "union(NP3, VP1)", tier='C')]
    g.rule_table = {0: g.rules[0].canonical}
    g.symbol_vocab_size = 4          # arbitrary; only the offset matters
    g._configured = True
    return g


def _make_ws(view, *, attention_mode, hot_ref=None, hot_boost=4.0,
             capacity=8, live=8, top_order=3):
    """Real WordSubSpace carrying the REAL ``retrieval_candidates_for_slot``
    and ``_commit_priming`` methods, wired with a real Taxonomy + the typed
    view. ``conceptualSpace`` is a stub exposing only ``attention_mode``
    (tier='C', so ``unreduce`` reads ``ws.conceptualSpace.attention_mode``).

    ``hot_ref`` (optional) is primed HOT so the retrieval helper boosts it.
    ``_order`` slot 0 is set to ``top_order`` (the parent's order, which the
    order-preserving union shares with both operands)."""
    ws = object.__new__(WordSubSpace)
    nn.Module.__init__(ws)
    ws.batch = 1
    tax = Taxonomy()
    tax.allocate_priming(batch_size=1, capacity=capacity, live=live)
    tax.configure_priming(priming_enabled=True)
    if hot_ref is not None:
        tax.prime([int(hot_ref)], batch=0, boost=hot_boost)
    ws.taxonomy = tax
    # ``knowledge`` is a read-only property backed by ``_knowledge``.
    object.__setattr__(ws, '_knowledge', view)
    # Per-slot order buffer: slot 0 holds the top (parent) order.
    ws._order = torch.zeros(1, capacity, dtype=torch.long)
    ws._order[0, 0] = int(top_order)
    # Tier-'C' owning space attention mode (the gate the CAPSTONE reads).
    object.__setattr__(ws, 'conceptualSpace',
                       SimpleNamespace(attention_mode=attention_mode))
    object.__setattr__(ws, 'symbolicSpace', None)
    return ws


def _build_subspace(W, parent, view, *, attention_mode, hot_ref=None,
                    top_order=3):
    """Assemble a B=1, K=3 stack with the binary ``union`` parent stamped in
    the top live slot (slot 0) and an empty slot to its right (slot 1)."""
    D = W.shape[1]
    K = 3
    what = torch.zeros(1, K, D)
    what[0, 0, :] = parent
    where = torch.zeros(1, K, 1)
    # rule 0 -> where_id = symbol_vocab_size + 1 + 0 = 5 (grammar below).
    where[0, 0, 0] = float(4 + 1 + 0)
    activation = torch.zeros(1, K)
    activation[0, 0] = 1.0                     # exactly one live slot
    basis = _Basis(W)
    ws = _make_ws(view, attention_mode=attention_mode, hot_ref=hot_ref,
                  capacity=8, live=8, top_order=top_order)
    sub = _StackSubSpace(what, where, activation, basis, ws)
    return sub, ws


def _register_union_syntactic_layer():
    """Duck-typed SyntacticLayer: tier 'C' (so the gate reads
    ``conceptualSpace.attention_mode``) with the REAL UnionLayer bound to the
    decoded ``method_name`` ('union')."""
    return SimpleNamespace(tier='C', _by_name={'union': UnionLayer()})


# ---------------------------------------------------------------------------
# The decisive scenario
# ---------------------------------------------------------------------------
#
# W has two NP-class rows that are both feasible left operands for union x1
# (both <= parent element-wise). Row 0 has the SMALLER norm (loses the
# unprimed argmax); row 1 has the LARGER norm (wins unprimed). The typed
# admissible set for the left slot (category 'NP', order 3) is {0, 1}; for
# the right slot (category 'VP', order 1) it is {2}.
#
#   * OFF: no heat -> union x1 = argmax(norm <= parent) -> row 1 (W[1]).
#   * ON, row 0 primed HOT: the boosted priming lifts row 0's effective
#     score above row 1's -> union x1 = W[0]. Heat steered the pick.
#
# parent = [0.5, 0.4] dominates both NP rows, so both are feasible parts.

_W = torch.tensor([
    [0.40, 0.30],   # row 0 -- NP, norm ~0.500, HOT in the ON case
    [0.41, 0.31],   # row 1 -- NP, norm ~0.515, unprimed argmax winner
    [0.20, 0.10],   # row 2 -- VP (right-slot admissible)
])
_PARENT = torch.tensor([0.5, 0.4])

# Typed admissibility: NP@3 -> {0,1}; VP@1 -> {2}.
_VIEW = _FakeView(
    by_cat={'NP': torch.tensor([0, 1], dtype=torch.long),
            'VP': torch.tensor([2], dtype=torch.long)},
    by_order={3: torch.tensor([0, 1], dtype=torch.long),
              1: torch.tensor([2], dtype=torch.long)},
)


def _run_unreduce(attention_mode, hot_ref):
    g = _grammar_with_union()
    sub, ws = _build_subspace(
        _W, _PARENT, _VIEW, attention_mode=attention_mode, hot_ref=hot_ref,
        top_order=3)
    syn = _register_union_syntactic_layer()
    lang = LanguageLayer.__new__(LanguageLayer)   # method container; no state
    lang.unreduce(sub, syn, grammar=g)
    # Decoded left operand was written into the (formerly top) slot 0.
    left = sub.materialize(mode="what")[0, 0, :]
    return left, ws


class TestHeatReverseWiringLiveON:
    """The gated CAPSTONE path actually steers the operand pick when ON, and
    is a no-op (baseline pick) when OFF."""

    def test_off_baseline_picks_larger_norm_row(self):
        """attention_mode='off' -> dormant gate -> union x1 = row 1 (the
        unprimed argmax winner, largest norm <= parent)."""
        left, _ = _run_unreduce(attention_mode='off', hot_ref=0)
        assert torch.allclose(left, _W[1], atol=1e-5), (
            f"OFF must reproduce the un-steered pick W[1]={_W[1].tolist()}, "
            f"got {left.tolist()}")
        # Cross-check against the raw recommender baseline.
        from Layers import Ops
        x1_base, _ = Ops.disjunctionReverse(
            _PARENT, _PARENT, _W,
            left_rows=torch.tensor([0, 1]), right_rows=torch.tensor([2]))
        assert torch.allclose(left, x1_base, atol=1e-5), (
            "OFF unreduce pick must equal the typed-only recommender baseline")

    def test_on_primer_steers_pick_to_hot_typed_row(self):
        """attention_mode='primer' with row 0 primed HOT -> union x1 = row 0
        (the hot AND admissible row), flipping the OFF baseline."""
        left, _ = _run_unreduce(attention_mode='primer', hot_ref=0)
        assert torch.allclose(left, _W[0], atol=1e-5), (
            f"ON must steer the pick to the hot+typed row W[0]={_W[0].tolist()}, "
            f"got {left.tolist()}")

    def test_on_and_off_differ(self):
        """The gate is load-bearing: ON and OFF select different operands on
        the identical fixture."""
        left_on, _ = _run_unreduce(attention_mode='primer', hot_ref=0)
        left_off, _ = _run_unreduce(attention_mode='off', hot_ref=0)
        assert not torch.allclose(left_on, left_off, atol=1e-5), (
            "Heat steering must change the pick relative to the OFF baseline")


class TestHeatReverseSelfPriming:
    """Task B: after a heat-steered binary pick, the selected operand rows are
    re-primed so subsequent reverse steps see them hot (plan reverse-flow
    steps 7-8). Row ids are recovered by matching the returned operand vectors
    against the candidate ``rows`` subset of ``W``."""

    def test_selected_left_row_is_reprimed(self):
        """The hot+typed left pick (row 0) is noted + re-primed: its
        selection telemetry bumps and its heat stays positive."""
        left, ws = _run_unreduce(attention_mode='primer', hot_ref=0)
        assert torch.allclose(left, _W[0], atol=1e-5)
        total, boosted = ws.taxonomy.priming_telemetry()
        assert total >= 1, (
            f"reverse self-priming must record >=1 selection, got total={total}")
        # Row 0 was hot at selection time -> the boosted-selection counter
        # must have advanced too.
        assert boosted >= 1, (
            f"the hot selected row must count as a boosted selection, "
            f"got boosted={boosted}")
        # Row 0 remains hot after re-priming.
        hm = ws.taxonomy.heat_mask(batch=0)
        assert hm is not None and float(hm[0].item()) > 0.0, (
            f"selected row 0 must stay hot after reverse self-priming, "
            f"heat_mask={None if hm is None else hm.tolist()}")

    def test_off_path_records_no_selection(self):
        """When OFF (gate dormant), the reverse self-priming block is skipped
        entirely -- no selection telemetry, no heat mutation."""
        _, ws = _run_unreduce(attention_mode='off', hot_ref=None)
        total, boosted = ws.taxonomy.priming_telemetry()
        assert total == 0 and boosted == 0, (
            f"OFF path must not run reverse self-priming; "
            f"got total={total}, boosted={boosted}")
        hm = ws.taxonomy.heat_mask(batch=0)
        assert hm is not None and hm.eq(0).all(), (
            "OFF path must leave taxonomy heat at zero")


# ---------------------------------------------------------------------------
# Intersection (conjunctionReverse) ON-path scenario
# ---------------------------------------------------------------------------
#
# W_INT has two NP-class rows that are both feasible x1 candidates for
# intersection (both >= parent element-wise). Row 0 has the LARGER norm
# (normally loses the argmin for x1); row 1 has the SMALLER norm (argmin
# winner without priming). Row 2 (VP) has element [0] below parent, so it
# is infeasible for x1 in both the restricted and unrestricted case -- this
# ensures the unrestricted OFF baseline also picks from {0, 1} only, making
# the OFF vs ON comparison clean.
#
#   * OFF: no heat, no row restriction -> argmin over feasible rows {0,1}
#     -> intersection x1 = row 1 (norm ~0.500 < 0.781).
#   * ON, row 0 primed HOT (boost=8): effective score = norm(row0)/8 ≈ 0.098
#     < norm(row1) 0.500 -> intersection x1 = row 0. Heat steered the pick.
#
# parent = [0.3, 0.2]; rows 0 and 1 satisfy S >= parent; row 2 does not.

_W_INT = torch.tensor([
    [0.60, 0.50],   # row 0 -- NP, norm ~0.781, HOT in the ON case
    [0.40, 0.30],   # row 1 -- NP, norm ~0.500, unprimed argmin winner
    [0.20, 0.25],   # row 2 -- VP, row2[0]=0.20 < parent[0]=0.30 -> infeasible x1
])
_PARENT_INT = torch.tensor([0.3, 0.2])

# Typed admissibility: NP@3 -> {0,1}; VP@1 -> {2}.
_VIEW_INT = _FakeView(
    by_cat={'NP': torch.tensor([0, 1], dtype=torch.long),
            'VP': torch.tensor([2], dtype=torch.long)},
    by_order={3: torch.tensor([0, 1], dtype=torch.long),
              1: torch.tensor([2], dtype=torch.long)},
)


def _grammar_with_intersection():
    """Real Grammar holding exactly one binary rule ``LP = intersection(NP3, VP1)``."""
    g = Grammar()
    g.rules = [g._parse_rule("LP", "intersection(NP3, VP1)", tier='C')]
    g.rule_table = {0: g.rules[0].canonical}
    g.symbol_vocab_size = 4
    g._configured = True
    return g


def _register_intersection_syntactic_layer():
    """Duck-typed SyntacticLayer with REAL IntersectionLayer."""
    return SimpleNamespace(tier='C', _by_name={'intersection': IntersectionLayer()})


def _run_unreduce_intersection(attention_mode, hot_ref):
    g = _grammar_with_intersection()
    sub, ws = _build_subspace(
        _W_INT, _PARENT_INT, _VIEW_INT,
        attention_mode=attention_mode, hot_ref=hot_ref, top_order=3)
    syn = _register_intersection_syntactic_layer()
    lang = LanguageLayer.__new__(LanguageLayer)
    lang.unreduce(sub, syn, grammar=g)
    # Decoded left operand was written into the (formerly top) slot 0.
    left = sub.materialize(mode="what")[0, 0, :]
    return left, ws


class TestHeatReverseWiringIntersectionON:
    """The gated CAPSTONE heat path steers the intersection/conjunctionReverse
    x1 pick when ON, and is a no-op (baseline argmin) when OFF.

    This exercises IntersectionLayer.reverse -> Ops.conjunctionReverse,
    a distinct code path from the union tests. A bug reachable only on the
    intersection path (e.g. inside the broad ``except`` in the heat block)
    would silently disable steering (ON==OFF) and be caught here.
    """

    def test_off_baseline_picks_smaller_norm_row(self):
        """attention_mode='off' -> dormant gate -> intersection x1 = row 1
        (the unprimed argmin winner, smallest norm >= parent)."""
        left, _ = _run_unreduce_intersection(attention_mode='off', hot_ref=0)
        assert torch.allclose(left, _W_INT[1], atol=1e-5), (
            f"OFF must reproduce the un-steered pick W_INT[1]={_W_INT[1].tolist()}, "
            f"got {left.tolist()}")
        # Cross-check against the raw (unrestricted) recommender baseline --
        # OFF unreduce calls layer.reverse with no left_rows/right_rows.
        from Layers import Ops
        x1_base, _ = Ops.conjunctionReverse(
            _PARENT_INT, _PARENT_INT, _W_INT)
        assert torch.allclose(left, x1_base, atol=1e-5), (
            "OFF unreduce pick must equal the unrestricted conjunctionReverse baseline")

    def test_on_primer_steers_pick_to_hot_typed_row(self):
        """attention_mode='primer' with row 0 primed HOT -> intersection x1 = row 0
        (the hot AND admissible row, flipping the argmin from the OFF baseline)."""
        left, _ = _run_unreduce_intersection(attention_mode='primer', hot_ref=0)
        assert torch.allclose(left, _W_INT[0], atol=1e-5), (
            f"ON must steer the intersection pick to the hot+typed row "
            f"W_INT[0]={_W_INT[0].tolist()}, got {left.tolist()}")

    def test_on_and_off_differ(self):
        """The gate is load-bearing on the intersection path: ON and OFF select
        different x1 operands on the identical fixture. Disabling the heat
        block on this path would collapse this to ON==OFF and fail."""
        left_on, _ = _run_unreduce_intersection(attention_mode='primer', hot_ref=0)
        left_off, _ = _run_unreduce_intersection(attention_mode='off', hot_ref=0)
        assert not torch.allclose(left_on, left_off, atol=1e-5), (
            "Heat steering must change the intersection pick relative to the OFF baseline")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
