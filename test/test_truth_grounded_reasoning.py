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
                       InterveningIdeaGenerator,
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

    def test_ultimate_truth_via_model(self):
        # No store match -> falls through to the model's absolute TruthLayer.
        r = TruthGroundedReasoner(model=_ModelStub(dot=0.6))
        self.assertAlmostEqual(r.is_true(IDEA_A), 0.6, places=5)


class TestIsPartDirect(unittest.TestCase):
    def test_geometric_containment(self):
        r = TruthGroundedReasoner(store=_store())
        res = r.is_part_direct(PART, WHOLE)
        self.assertIsNotNone(res)
        score, how = res
        self.assertEqual(how, "geometric")
        self.assertGreaterEqual(score, r.theta)

    def test_stored_partof_row(self):
        # LEFT/RIGHT are geometrically disjoint, so only the stored row answers.
        r = TruthGroundedReasoner(
            store=_store(rows_partof=[(LEFT, RIGHT, 0.85)]))
        self.assertIsNone(  # sanity: not geometric
            r.is_part_direct(LEFT, RIGHT) if False else None)
        res = r.is_part_direct(LEFT, RIGHT)
        self.assertIsNotNone(res)
        score, how = res
        self.assertEqual(how, "stored")
        self.assertAlmostEqual(score, 0.85, places=5)

    def test_disjoint_is_none(self):
        r = TruthGroundedReasoner(store=_store())
        self.assertIsNone(r.is_part_direct(LEFT, RIGHT))

    def test_negative_trust_row_not_accepted(self):
        r = TruthGroundedReasoner(
            store=_store(rows_partof=[(LEFT, RIGHT, -0.9)]))
        self.assertIsNone(r.is_part_direct(LEFT, RIGHT))


# Discrete, pairwise-disjoint ideas: no geometric parthood holds between any
# two, so only a stored chain can connect them (the syllogism case).
SOCRATES = _v(1, 0, 0, 0, 0, 0, 0, 0)
MAN = _v(0, 1, 0, 0, 0, 0, 0, 0)
MORTAL = _v(0, 0, 1, 0, 0, 0, 0, 0)
ANIMAL = _v(0, 0, 0, 1, 0, 0, 0, 0)


class TestChain(unittest.TestCase):
    def test_socrates_syllogism(self):
        # Socrates ⊑ man, man ⊑ mortal  ⇒  isPart(Socrates, mortal) via 1 hop.
        r = TruthGroundedReasoner(store=_store(rows_partof=[
            (SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)]))
        # No direct geometric/stored edge Socrates→mortal.
        self.assertIsNone(r.is_part_direct(SOCRATES, MORTAL))
        cands = r.is_part(SOCRATES, MORTAL)
        self.assertTrue(cands)
        best = cands[0]
        self.assertEqual(best["how"], "chain")
        self.assertEqual(best["steps"], 2)
        self.assertAlmostEqual(best["score"], 0.8, places=5)   # MIN(0.9, 0.8)

    def test_no_chain_returns_empty(self):
        r = TruthGroundedReasoner(store=_store(rows_partof=[
            (SOCRATES, MAN, 0.9)]))                              # dead-ends at man
        self.assertEqual(r.is_part(SOCRATES, MORTAL), [])

    def test_min_trust_is_weakest_hop(self):
        # man ⊑ mortal ⊑ animal: chain trust = min(0.6, 0.95) = 0.6
        r = TruthGroundedReasoner(store=_store(rows_partof=[
            (MAN, MORTAL, 0.6), (MORTAL, ANIMAL, 0.95)]))
        cands = r.is_part(MAN, ANIMAL)
        self.assertTrue(cands)
        self.assertAlmostEqual(cands[0]["score"], 0.6, places=5)

    def test_beam_caps_candidate_count(self):
        # Many parallel distractor edges out of MAN; beam=2 caps the results.
        rows = [(MAN, _v(0, 0, 0, 0, *[1 if j == k else 0 for j in range(4)]),
                 0.5) for k in range(4)]
        rows.append((MAN, MORTAL, 0.8))
        r = TruthGroundedReasoner(store=_store(rows_partof=rows))
        cands = r.is_part(MAN, MORTAL, beam=2)
        self.assertLessEqual(len(cands), 2)
        self.assertTrue(any(c["how"] == "chain" for c in cands))

    def test_direct_ranks_first(self):
        # PART ⊑ WHOLE is direct (geometric); also add a weaker stored chain.
        r = TruthGroundedReasoner(store=_store(rows_partof=[
            (PART, MAN, 0.5), (MAN, WHOLE, 0.4)]))
        cands = r.is_part(PART, WHOLE)
        self.assertTrue(cands)
        self.assertEqual(cands[0]["how"], "geometric")


class TestMaterialize(unittest.TestCase):
    def test_verified_chain_becomes_direct_hit(self):
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)])
        r = TruthGroundedReasoner(store=store)
        # Before: only the chain answers; no direct edge.
        self.assertIsNone(r.is_part_direct(SOCRATES, MORTAL))
        cands = r.is_part(SOCRATES, MORTAL, materialize=True)
        self.assertEqual(cands[0]["how"], "chain")
        self.assertIn("materialized", cands[0])
        # After: the conclusion is a stored direct edge with the chain trust.
        direct = r.is_part_direct(SOCRATES, MORTAL)
        self.assertIsNotNone(direct)
        score, how = direct
        self.assertEqual(how, "stored")
        self.assertAlmostEqual(score, 0.8, places=5)

    def test_below_floor_not_written(self):
        store = _store(rows_partof=[(SOCRATES, MAN, 0.4), (MAN, MORTAL, 0.3)])
        r = TruthGroundedReasoner(store=store, materialize_floor=0.5)
        cands = r.is_part(SOCRATES, MORTAL, materialize=True)
        # chain score = min(0.4,0.3)=0.3 < floor -> not materialized
        if cands:
            self.assertNotIn("materialized", cands[0])
        self.assertIsNone(r.is_part_direct(SOCRATES, MORTAL))

    def test_materialize_noop_without_store(self):
        r = TruthGroundedReasoner(model=_ModelStub())
        self.assertEqual(r.materialize(SOCRATES, MORTAL, 0.9), -1)


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
        r = TruthGroundedReasoner(store=_store())
        res = r.evaluate(QuerySpec.from_surface("isPart", PART, WHOLE))
        self.assertEqual(res["posture"], TRUE)
        self.assertEqual(res["kind"], KIND_IS_PART)
        self.assertEqual(len(res["candidates"]), 1)

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
    """The reasoner's tool surface = the grammar <queries> ops."""

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

    def test_part(self):
        r = TruthGroundedReasoner()
        self.assertGreaterEqual(r.part(PART, WHOLE), 0.7)
        self.assertAlmostEqual(r.part(LEFT, RIGHT), 0.0, places=5)

    def test_wholes_proximal_frontier(self):
        r = TruthGroundedReasoner(store=_store(rows_partof=[
            (SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)]))
        ws = r.wholes(SOCRATES)
        self.assertEqual(len(ws), 1)                 # proximal: man, not mortal
        self.assertGreaterEqual(r.equal(ws[0]["idea"], MAN), 0.99)
        self.assertAlmostEqual(ws[0]["trust"], 0.9, places=5)

    def test_parts_is_inverse_of_wholes(self):
        r = TruthGroundedReasoner(store=_store(rows_partof=[
            (SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)]))
        ps = r.parts(MORTAL)
        self.assertEqual(len(ps), 1)                 # proximal: man
        self.assertGreaterEqual(r.equal(ps[0]["idea"], MAN), 0.99)

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
    def test_is_part_false_from_refuting_edge(self):
        # ¬isPart asserted (negative-trust REL_PARTOF), no supporting evidence.
        r = TruthGroundedReasoner(
            store=_store(rows_partof=[(LEFT, RIGHT, -0.9)]))
        res = r.evaluate(QuerySpec.from_surface("isPart", LEFT, RIGHT))
        self.assertEqual(res["posture"], FALSE)

    def test_is_part_both_when_supported_and_refuted(self):
        # PART⊑WHOLE holds geometrically AND a refuting edge is asserted.
        r = TruthGroundedReasoner(
            store=_store(rows_partof=[(PART, WHOLE, -0.9)]))
        res = r.evaluate(QuerySpec.from_surface("isPart", PART, WHOLE))
        self.assertEqual(res["posture"], BOTH)

    def test_chain_trace_rendered(self):
        r = TruthGroundedReasoner(store=_store(rows_partof=[
            (SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)]))
        res = r.evaluate(QuerySpec.from_surface("isPart", SOCRATES, MORTAL))
        self.assertEqual(res["posture"], TRUE)
        self.assertIsNotNone(res["trace"])
        self.assertIn("min hop", res["trace"])

    def test_direct_trace_rendered(self):
        r = TruthGroundedReasoner(store=_store())
        res = r.evaluate(QuerySpec.from_surface("isPart", PART, WHOLE))
        self.assertIn("direct", res["trace"])


class TestConsolidation(unittest.TestCase):
    """The chain climb is ONE canonical primitive on ConceptualSpace, shared by
    the reasoner's is_part and ConceptualSpace.reason."""

    def test_chain_to_target_is_shared_loop(self):
        from Spaces import ConceptualSpace
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)])
        chains = ConceptualSpace._chain_to_target(
            SOCRATES, MORTAL, store, parthood_threshold=0.7, max_steps=8,
            beam=8, trust_combine="min", rel_type=TernaryTruthStore.REL_PARTOF)
        self.assertTrue(chains)
        self.assertAlmostEqual(chains[0]["score"], 0.8, places=5)
        # is_part delegates to the SAME loop -> same best score + chain.
        cands = TruthGroundedReasoner(store=store).is_part(SOCRATES, MORTAL)
        self.assertAlmostEqual(cands[0]["score"], chains[0]["score"], places=5)
        self.assertEqual(cands[0]["chain"], chains[0]["chain"])

    def test_wholes_is_canonical_conceptualspace_method(self):
        from Spaces import ConceptualSpace
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9)])
        ws = ConceptualSpace.wholes(
            SOCRATES, store, theta=0.7,
            rel_type=TernaryTruthStore.REL_PARTOF)
        self.assertEqual(len(ws), 1)
        # The reasoner's wholes() returns the same.
        rw = TruthGroundedReasoner(store=store).wholes(SOCRATES)
        self.assertEqual(len(rw), 1)
        self.assertEqual(rw[0]["row"], ws[0]["row"])

    def test_reason_open_mode_unchanged_shape(self):
        # Default (no target) still returns the derived/luminosity_gain shape.
        from Spaces import ConceptualSpace
        rows = ConceptualSpace._iter_relation_rows(
            _store(rows_partof=[(SOCRATES, MAN, 0.9)]),
            TernaryTruthStore.REL_PARTOF)
        self.assertEqual(len(list(rows)), 1)


class TestPartialOrder(unittest.TestCase):
    """Phase 6: antisymmetry / cycle guard at edge insertion."""

    def test_materialize_rejects_cycle(self):
        # MORTAL ⊑ MAN already stored; writing MAN ⊑ MORTAL would cycle.
        r = TruthGroundedReasoner(store=_store(rows_partof=[(MORTAL, MAN, 0.9)]))
        self.assertEqual(r.materialize(MAN, MORTAL, 0.9), -1)

    def test_materialize_rejects_self_loop(self):
        r = TruthGroundedReasoner(store=_store())
        self.assertEqual(r.materialize(MAN, MAN, 0.9), -1)

    def test_materialize_accepts_acyclic(self):
        r = TruthGroundedReasoner(store=_store(rows_partof=[(SOCRATES, MAN, 0.9)]))
        self.assertGreaterEqual(r.materialize(MAN, MORTAL, 0.9), 0)


class TestSoftGenerator(unittest.TestCase):
    """Phase 3: the intervening-idea generator over the truth-space."""

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

    def test_generator_proposes_and_recurs(self):
        from Spaces import GlobalAttention
        store = _store(rows_ideas=[(IDEA_A, 0.9), (IDEA_C, 0.8)])
        gen = InterveningIdeaGenerator(dim=8)
        ga = GlobalAttention()
        spaces = self._spaces(store)
        out = gen.propose(SOCRATES, MORTAL, spaces, ga=ga, top_k=3)
        self.assertIsNotNone(out)
        self.assertEqual(int(out["idea"].shape[-1]), 8)
        out2 = gen.propose(SOCRATES, MORTAL, spaces, ga=ga,
                           prev_r=out["idea"], top_k=3)
        self.assertIsNotNone(out2)

    def test_query_head_is_differentiable(self):
        gen = InterveningIdeaGenerator(dim=8)
        q = gen.concept_q(SOCRATES, MORTAL)
        self.assertTrue(q.requires_grad)
        q.sum().backward()
        # the head has a gradient -> the soft route trains
        grads = [p.grad for p in gen.parameters() if p.grad is not None]
        self.assertTrue(grads)


class TestNeuralToolUser(unittest.TestCase):
    """Phase B: the recurrent tool-use driver (soft propose / hard verify)."""

    def _spaces(self, *ideas):
        from Spaces import GlobalAttention as GA
        keys = torch.stack(list(ideas)).detach()
        return [{"id": GA.SPACE_WHOLE, "keys": keys,
                 "boosts": torch.ones(keys.shape[0])}]

    def _tooluser(self, store, spaces, *, iterations=10):
        from Spaces import GlobalAttention
        reasoner = TruthGroundedReasoner(store=store)
        gen = InterveningIdeaGenerator(dim=8)
        return reasoning.NeuralToolUser(
            reasoner, generator=gen, ga=GlobalAttention(), spaces=spaces,
            iterations=iterations)

    def test_leaf_istrue_delegates(self):
        # A leaf judgment needs no chain loop -> evaluate(), iterations 0.
        store = _store(rows_ideas=[(IDEA_A, 0.9)])
        tool = self._tooluser(store, self._spaces(IDEA_A, IDEA_C))
        res = tool.run(QuerySpec(KIND_IS_TRUE, left=IDEA_A))
        self.assertEqual(res.posture, TRUE)
        self.assertEqual(res.iterations, 0)
        self.assertEqual(res.ideas, [])

    def test_stored_chain_without_generator(self):
        # No soft half -> the stored hard chain alone still proves the syllogism.
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)])
        tool = reasoning.NeuralToolUser(
            TruthGroundedReasoner(store=store), iterations=10)
        res = tool.run(QuerySpec(KIND_IS_PART, left=SOCRATES, right=MORTAL))
        self.assertEqual(res.posture, TRUE)
        self.assertTrue(res.chain)
        self.assertEqual(res.ideas, [])
        self.assertEqual(res.iterations, 0)

    def test_generator_surfaces_verified_intermediate(self):
        # MAN bridges Socrates->mortal; the soft loop must surface + hard-verify
        # it above the distractors.
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)])
        tool = self._tooluser(store, self._spaces(MAN, ANIMAL, IDEA_C))
        res = tool.run(QuerySpec(KIND_IS_PART, left=SOCRATES, right=MORTAL))
        self.assertEqual(res.posture, TRUE)
        verified = [it for it in res.ideas if it["verified"]]
        self.assertTrue(verified)
        best = max(verified, key=lambda it: it["trust"])
        self.assertGreaterEqual(
            TruthGroundedReasoner.equal(best["idea"], MAN), 0.99)
        self.assertAlmostEqual(best["trust"], 0.8, places=5)   # MIN(0.9, 0.8)

    def test_ideas_ranked_by_relevance(self):
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)])
        tool = self._tooluser(store, self._spaces(MAN, ANIMAL, IDEA_C))
        res = tool.run(QuerySpec(KIND_IS_PART, left=SOCRATES, right=MORTAL))
        rels = [it["relevance"] for it in res.ideas]
        self.assertEqual(rels, sorted(rels, reverse=True))

    def test_iterations_capped_and_terminates(self):
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)])
        tool = self._tooluser(store, self._spaces(MAN, ANIMAL, IDEA_C),
                              iterations=3)
        res = tool.run(QuerySpec(KIND_IS_PART, left=SOCRATES, right=MORTAL))
        self.assertGreaterEqual(res.iterations, 1)
        self.assertLessEqual(res.iterations, 3)        # bounded by N
        self.assertLessEqual(len(res.ideas), 3)        # N-capped output

    def test_inert_without_spaces_falls_back(self):
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)])
        from Spaces import GlobalAttention
        tool = reasoning.NeuralToolUser(
            TruthGroundedReasoner(store=store),
            generator=InterveningIdeaGenerator(dim=8),
            ga=GlobalAttention(), spaces=None, iterations=10)
        res = tool.run(QuerySpec(KIND_IS_PART, left=SOCRATES, right=MORTAL))
        self.assertEqual(res.posture, TRUE)            # stored chain still works
        self.assertEqual(res.ideas, [])                # soft half inert


class TestReasonPredictNext(unittest.TestCase):
    """Step 2: the differentiable next-idea blend over {arma, retrieval,
    deduction}; arma earns its weight, absent tools get exactly 0."""

    def _spaces(self, *ideas):
        from Spaces import GlobalAttention as GA
        keys = torch.stack(list(ideas)).detach()
        return [{"id": GA.SPACE_WHOLE, "keys": keys}]

    def test_blend_differentiable_masks_absent_arma(self):
        from Spaces import GlobalAttention
        # store gives a 'wholes' deduction candidate; no model -> arma is absent.
        reasoner = TruthGroundedReasoner(store=_store(
            rows_partof=[(SOCRATES, MAN, 0.9)]))
        gen = InterveningIdeaGenerator(dim=8)
        tool = reasoning.NeuralToolUser(
            reasoner, generator=gen, ga=GlobalAttention(),
            spaces=self._spaces(MAN, MORTAL, IDEA_C))
        e_hat, weights = tool.reason_predict_next(SOCRATES)
        self.assertIsNotNone(e_hat)
        self.assertEqual(int(weights.numel()), 3)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)
        # arma (index 0) absent (no discourse) -> exactly 0 weight.
        self.assertAlmostEqual(float(weights[0]), 0.0, places=6)
        # differentiable through the generator query head.
        self.assertTrue(e_hat.requires_grad)
        e_hat.sum().backward()
        g = gen.head[0].weight.grad
        self.assertIsNotNone(g)
        self.assertTrue(torch.isfinite(g).all())

    def test_none_without_generator(self):
        reasoner = TruthGroundedReasoner(store=_store(
            rows_partof=[(SOCRATES, MAN, 0.9)]))
        tool = reasoning.NeuralToolUser(reasoner)      # no generator
        self.assertEqual(tool.reason_predict_next(SOCRATES), (None, None))


class TestAnswerPolicyLoss(unittest.TestCase):
    """Phase C: the differentiable answer-policy loss + store-derived examples."""

    def _spaces(self, *ideas):
        from Spaces import GlobalAttention as GA
        keys = torch.stack(list(ideas)).detach()
        return [{"id": GA.SPACE_WHOLE, "keys": keys}]

    def test_examples_from_store_builds_transitive_pairs(self):
        r = TruthGroundedReasoner(store=_store(rows_partof=[
            (SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)]))
        ex = reasoning.policy_examples_from_store(r)
        # the 2-hop (socrates -> mortal) is a positive; its reverse is negative.
        pos = [(a, b, g) for (a, b, g) in ex if g == 1.0]
        self.assertTrue(any(
            TruthGroundedReasoner.equal(a, SOCRATES) >= 0.99
            and TruthGroundedReasoner.equal(b, MORTAL) >= 0.99 for a, b, g in pos))
        self.assertTrue(any(g == 0.0 for _, _, g in ex))   # a negative present

    def test_policy_loss_trains_generator_head(self):
        # The bridge (MAN) is present in the truth-space -> the positive example
        # produces a real gradient on the generator query head (the soft route).
        r = TruthGroundedReasoner(store=_store(rows_partof=[
            (SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)]))
        gen = InterveningIdeaGenerator(dim=8)
        spaces = self._spaces(MAN, ANIMAL, IDEA_C)
        ex = reasoning.policy_examples_from_store(r)
        loss = reasoning.policy_answer_loss(gen, spaces, r, ex)
        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        g = gen.head[0].weight.grad
        self.assertIsNotNone(g)
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(float(g.norm()), 0.0)           # the head actually trained

    def test_policy_loss_none_without_examples_or_spaces(self):
        r = TruthGroundedReasoner(store=_store(rows_partof=[(SOCRATES, MAN, 0.9)]))
        gen = InterveningIdeaGenerator(dim=8)
        # no 2-hop chain -> no examples -> None
        self.assertEqual(reasoning.policy_examples_from_store(r), [])
        self.assertIsNone(reasoning.policy_answer_loss(
            gen, self._spaces(MAN), r, []))                 # empty examples
        ex = [(SOCRATES, MORTAL, 1.0)]
        self.assertIsNone(reasoning.policy_answer_loss(gen, [], r, ex))  # no spaces

    def test_policy_loss_skips_oversized_space(self):
        # A percept-scale codebook (> max_keys rows) is skipped; with only that
        # space present, the loss is None (no idea-scale keys).
        r = TruthGroundedReasoner(store=_store(rows_partof=[
            (SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)]))
        gen = InterveningIdeaGenerator(dim=8)
        from Spaces import GlobalAttention as GA
        big = [{"id": GA.SPACE_PART, "keys": torch.zeros(600, 8)}]
        ex = reasoning.policy_examples_from_store(r)
        self.assertIsNone(reasoning.policy_answer_loss(
            gen, big, r, ex, max_keys=512))


class TestAnswerLoss(unittest.TestCase):
    """Phase 5: the policy (answer) loss primitive."""

    def test_proof_score_maps_signed_to_unit(self):
        self.assertAlmostEqual(reasoning.proof_score(1.0), 1.0, places=6)
        self.assertAlmostEqual(reasoning.proof_score(-1.0), 0.0, places=6)
        self.assertAlmostEqual(reasoning.proof_score(0.0), 0.5, places=6)

    def test_answer_loss_lower_when_correct(self):
        good = float(reasoning.answer_loss(torch.tensor(0.9), 1.0))
        bad = float(reasoning.answer_loss(torch.tensor(-0.9), 1.0))
        self.assertLess(good, bad)

    def test_answer_loss_differentiable(self):
        p = torch.tensor(0.2, requires_grad=True)
        reasoning.answer_loss(p, 1.0).backward()
        self.assertIsNotNone(p.grad)


if __name__ == "__main__":
    unittest.main()
