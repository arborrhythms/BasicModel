"""Phase 0/1 of truth-grounded reasoning (doc/plans/2026-06-23-truth-grounded-
reasoning.md): QuerySpec framing + the hard tools is_true / is_part_direct /
evaluate. Unit tests, no trained model -- the reasoner reads a hand-built
TernaryTruthStore and an optional model stub.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import torch

import reasoning
from reasoning import (QuerySpec, TruthGroundedReasoner,
                       KIND_IS_TRUE, KIND_IS_PART, KIND_IS_EQUAL,
                       TRUE, FALSE, UNKNOWN, BOTH)
from Layers import TernaryTruthStore


def _v(*xs):
    return torch.tensor(list(xs), dtype=torch.float32)


# Eight-dim idea vectors used across the tests.
IDEA_A = _v(1, 1, 1, 1, 0, 0, 0, 0)      # for is_true rows
IDEA_C = _v(0, 0, 0, 0, 1, 1, 1, 1)
PART = _v(1, 0, 0, 0, 0, 0, 0, 0)        # A small idea, contained by WHOLE
WHOLE = _v(1, 1, 1, 1, 1, 1, 1, 1)
LEFT = _v(1, 1, 0, 0, 0, 0, 0, 0)        # disjoint from RIGHT (no geometric part)
RIGHT = _v(0, 0, 0, 0, 1, 1, 0, 0)


class _ModelStub:
    """Minimal model exposing isTrue (absolute-truth path) and no CS."""
    def __init__(self, dot=0.0):
        self.conceptualSpace = None
        self._dot = float(dot)

    def isTrue(self, activation):
        return self._dot


class TestQuerySpec(unittest.TestCase):
    def test_surface_normalization(self):
        self.assertEqual(QuerySpec.from_surface("exist", PART).predicate,
                         KIND_IS_TRUE)
        self.assertEqual(QuerySpec.from_surface("isTrue", PART).predicate,
                         KIND_IS_TRUE)
        self.assertEqual(QuerySpec.from_surface("queryPart", PART, WHOLE).predicate,
                         KIND_IS_PART)
        self.assertEqual(QuerySpec.from_surface("part", PART, WHOLE).predicate,
                         KIND_IS_PART)
        self.assertEqual(QuerySpec.from_surface("queryEqual", PART, WHOLE).predicate,
                         KIND_IS_EQUAL)

    def test_unknown_surface_raises(self):
        with self.assertRaises(ValueError):
            QuerySpec.from_surface("frobnicate", PART)

    def test_open_variable(self):
        q = QuerySpec.from_surface("isPart", None, WHOLE, variables=("left",))
        self.assertTrue(q.is_open)
        self.assertFalse(QuerySpec.from_surface("isPart", PART, WHOLE).is_open)


def _store(rows_ideas=(), rows_partof=()):
    """Build a TernaryTruthStore. rows_ideas = [(vec, trust)]; rows_partof =
    [(np1, np2, trust)]."""
    s = TernaryTruthStore(nDim=8, capacity=32)
    for vec, trust in rows_ideas:
        s.append_idea(vec, trust=trust)
    for np1, np2, trust in rows_partof:
        s.append_relation(np1, torch.zeros(8), np2,
                          rel_type=s.REL_PARTOF, trust=trust)
    return s


class TestIsTrue(unittest.TestCase):
    def test_positive_trust_single_idea(self):
        r = TruthGroundedReasoner(store=_store(rows_ideas=[(IDEA_A, 0.9)]))
        self.assertAlmostEqual(r.is_true(IDEA_A), 0.9, places=5)

    def test_negative_trust_single_idea(self):
        r = TruthGroundedReasoner(store=_store(rows_ideas=[(IDEA_C, -0.8)]))
        self.assertAlmostEqual(r.is_true(IDEA_C), -0.8, places=5)

    def test_absent_idea_is_unknown(self):
        r = TruthGroundedReasoner(store=_store(rows_ideas=[(IDEA_A, 0.9)]))
        self.assertEqual(r.is_true(IDEA_C), 0.0)

    def test_activation_without_fact_evidence_remains_unknown(self):
        # Concept activation does not establish that its referent exists.
        r = TruthGroundedReasoner(model=_ModelStub(dot=0.6))
        self.assertEqual(r.is_true(IDEA_A), 0.0)


# Discrete, pairwise-disjoint ideas: no geometric parthood holds between any
# two, so only a stored chain can connect them (the syllogism case).
SOCRATES = _v(1, 0, 0, 0, 0, 0, 0, 0)
MAN = _v(0, 1, 0, 0, 0, 0, 0, 0)
MORTAL = _v(0, 0, 1, 0, 0, 0, 0, 0)
ANIMAL = _v(0, 0, 0, 1, 0, 0, 0, 0)


def _taxonomy(*links):
    """Explicit conceptual links; vector row fixtures do not define taxonomy."""
    from types import SimpleNamespace
    from test_cs_symbol_table import _cs
    cs = _cs()
    refs = tuple(("sym", cs.new_concept()) for _ in range(4))
    for left, right in links:
        cs.add_whole(refs[left][1], refs[right])
    return TruthGroundedReasoner(SimpleNamespace(conceptualSpace=cs), store=_store()), refs


class TestEvaluate(unittest.TestCase):
    def test_is_true_true(self):
        r = TruthGroundedReasoner(store=_store(rows_ideas=[(IDEA_A, 0.9)]))
        res = r.evaluate(QuerySpec.from_surface("exist", IDEA_A))
        self.assertEqual(res["posture"], TRUE)

    def test_is_true_false(self):
        r = TruthGroundedReasoner(store=_store(rows_ideas=[(IDEA_C, -0.8)]))
        res = r.evaluate(QuerySpec.from_surface("exist", IDEA_C))
        self.assertEqual(res["posture"], FALSE)

    def test_is_true_unknown(self):
        r = TruthGroundedReasoner(store=_store())
        res = r.evaluate(QuerySpec.from_surface("exist", IDEA_A))
        self.assertEqual(res["posture"], UNKNOWN)

    def test_is_part_direct_true(self):
        r, (a, b, _c, _d) = _taxonomy((0, 1))
        res = r.evaluate(QuerySpec.from_surface("isPart", a, b))
        self.assertEqual(res["posture"], TRUE)
        self.assertEqual(res["kind"], KIND_IS_PART)
        self.assertEqual(len(res["path"]), 1)
        self.assertEqual(res["domain"], "conceptual-taxonomy")

    def test_is_part_unknown(self):
        r = TruthGroundedReasoner(store=_store())
        res = r.evaluate(QuerySpec.from_surface("isPart", LEFT, RIGHT))
        self.assertEqual(res["posture"], UNKNOWN)

    def test_is_equal_shared_parts_and_wholes(self):
        r = TruthGroundedReasoner(store=_store())
        res = r.evaluate(QuerySpec.from_surface("isEqual", WHOLE, WHOLE))
        self.assertEqual(res["posture"], TRUE)
        self.assertEqual(res["kind"], KIND_IS_EQUAL)

    def test_is_equal_distinct_is_unknown(self):
        r = TruthGroundedReasoner(store=_store())
        res = r.evaluate(QuerySpec.from_surface("isEqual", SOCRATES, MORTAL))
        self.assertEqual(res["posture"], UNKNOWN)


class TestGrammarOps(unittest.TestCase):
    """Legacy vector helpers and the current full-description Exist adapter."""

    def test_exist_is_isTrue(self):
        r = TruthGroundedReasoner(store=_store(rows_ideas=[(IDEA_A, 0.9)]))
        self.assertAlmostEqual(r.exist(IDEA_A), 0.9, places=5)
        self.assertEqual(r.exist(IDEA_A), r.is_true(IDEA_A))

    def test_equal_isomorphic_vs_norm(self):
        r = TruthGroundedReasoner()
        self.assertAlmostEqual(r.equal(WHOLE, WHOLE, isomorphic=True), 1.0,
                               places=5)
        self.assertAlmostEqual(r.equal(WHOLE, WHOLE, isomorphic=False), 0.0,
                               places=5)
        # disjoint ideas: isomorphic fraction 0, norm > 0
        self.assertAlmostEqual(r.equal(SOCRATES, MAN, isomorphic=True), 0.0,
                               places=5)
        self.assertGreater(r.equal(SOCRATES, MAN, isomorphic=False), 0.0)


    def test_query_idea_and_relation(self):
        r = TruthGroundedReasoner(store=_store(
            rows_ideas=[(IDEA_A, 0.9)],
            rows_partof=[(SOCRATES, MAN, 0.7)]))
        hit = r.query(IDEA_A)
        self.assertEqual(hit["kind"], "idea")
        self.assertAlmostEqual(hit["trust"], 0.9, places=5)
        rel = r.query(SOCRATES, MAN)
        self.assertEqual(rel["kind"], "relation")
        self.assertAlmostEqual(rel["trust"], 0.7, places=5)

    def test_quantize_snaps_to_nearest_idea(self):
        r = TruthGroundedReasoner(store=_store(rows_ideas=[(IDEA_A, 0.9)]))
        # A noisy near-copy of IDEA_A snaps back onto it.
        noisy = IDEA_A + _v(0, 0, 0, 0, 0.05, 0, 0, 0)
        snapped = r.quantize(noisy)
        self.assertGreaterEqual(r.equal(snapped, IDEA_A), 0.99)

    def test_quantize_noop_without_store(self):
        r = TruthGroundedReasoner()
        out = r.quantize(IDEA_A)
        self.assertAlmostEqual(r.equal(out, IDEA_A), 1.0, places=5)


class TestPostureAndTrace(unittest.TestCase):
    def test_world_refutation_does_not_establish_taxonomic_falsehood(self):
        r = TruthGroundedReasoner(store=_store(rows_partof=[(LEFT, RIGHT, -0.9)]))
        res = r.evaluate(QuerySpec.from_surface("isPart", LEFT, RIGHT))
        self.assertEqual(res["posture"], UNKNOWN)
        self.assertEqual(res["support_false"], 0)

    def test_geometry_and_world_refutation_do_not_establish_taxonomic_conflict(self):
        r = TruthGroundedReasoner(store=_store(rows_partof=[(PART, WHOLE, -0.9)]))
        res = r.evaluate(QuerySpec.from_surface("isPart", PART, WHOLE))
        self.assertEqual(res["posture"], UNKNOWN)
        self.assertEqual((res["support_true"], res["support_false"]), (0, 0))

    def test_chain_trace_rendered(self):
        r, (a, _b, c, _d) = _taxonomy((0, 1), (1, 2))
        res = r.evaluate(QuerySpec.from_surface("isPart", a, c))
        self.assertEqual(res["posture"], TRUE)
        self.assertEqual(len(res["path"]), 2)
        self.assertIn("2 links", res["trace"])

    def test_direct_trace_rendered(self):
        r, (a, b, _c, _d) = _taxonomy((0, 1))
        res = r.evaluate(QuerySpec.from_surface("isPart", a, b))
        self.assertIn("1 links", res["trace"])
        self.assertEqual(res["path"][0].owner, a)


class TestLegacyConsolidation(unittest.TestCase):
    """The chain climb is ONE canonical primitive on ConceptualSpace, shared by
    the reasoner's is_part and ConceptualSpace.reason."""


    def test_reason_open_mode_unchanged_shape(self):
        # Default (no target) still returns the derived/luminosity_gain shape.
        from Spaces import ConceptualSpace
        rows = ConceptualSpace._iter_relation_rows(
            _store(rows_partof=[(SOCRATES, MAN, 0.9)]),
            TernaryTruthStore.REL_PARTOF)
        self.assertEqual(len(list(rows)), 1)


class TestSoftRead(unittest.TestCase):
    """The retained numerical attention reader returns real stored keys."""

    def _spaces(self, store, D=8):
        from Spaces import GlobalAttention as GA
        n = int(store.count.item())
        ltm = (store.slots[:n].mean(dim=1).detach() if n > 0
               else torch.zeros(1, D))
        codebook = torch.stack([IDEA_A, IDEA_C, WHOLE]).detach()
        return [
            {"id": GA.SPACE_LTM, "keys": ltm},
            {"id": GA.SPACE_WHOLE, "keys": codebook, "boosts": torch.ones(3)},
        ]

    def test_where_read_grounds_in_real_keys(self):
        from Spaces import GlobalAttention
        store = _store(rows_ideas=[(IDEA_A, 0.9), (IDEA_C, 0.8)])
        spaces = self._spaces(store)
        read = TruthGroundedReasoner.where_read(
            IDEA_A, spaces, ga=GlobalAttention(), top_k=3)
        self.assertIsNotNone(read)
        self.assertEqual(int(read["idea"].shape[-1]), 8)
        self.assertLessEqual(len(read["candidates"]), 3)
        self.assertIn(read["space_id"],
                      {GlobalAttention.SPACE_LTM, GlobalAttention.SPACE_WHOLE})


if __name__ == "__main__":
    unittest.main()
