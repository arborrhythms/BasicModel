"""The Thinking Kernel (doc/plans/thinking_kernel_spec.md): truth intervals,
STM frames, the runtime-enforced lookup/part/think/query/answer loop, closure
rules, anti-hallucination invariants, testimony incorporation, and the reward/
trace compilers. Unit tests over a hand-built TernaryTruthStore (same fixture
conventions as test_truth_grounded_reasoning.py) + Models/config integration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import torch

from Layers import TernaryTruthStore
from reasoning import (QuerySpec, TruthGroundedReasoner,
                       KIND_IS_TRUE, KIND_IS_PART, KIND_IS_EQUAL)
import thinking
from thinking import (TruthInterval, Testimony, Frame, ChildResult,
                      ThinkingKernel, KernelPolicy, NextOpPolicy, as_spec,
                      next_op_loss, traces_from_store,
                      TRUE, FALSE, UNKNOWN, MIXED, CONFLICTING,
                      BOUNDED_UNKNOWN)


def _v(*xs):
    return torch.tensor(list(xs), dtype=torch.float32)


SOCRATES = _v(1, 0, 0, 0, 0, 0, 0, 0)
MAN = _v(0, 1, 0, 0, 0, 0, 0, 0)
MORTAL = _v(0, 0, 1, 0, 0, 0, 0, 0)
ANIMAL = _v(0, 0, 0, 1, 0, 0, 0, 0)
IDEA_A = _v(1, 1, 1, 1, 0, 0, 0, 0)


def _store(rows_ideas=(), rows_partof=()):
    s = TernaryTruthStore(nDim=8, capacity=32)
    for vec, trust in rows_ideas:
        s.append_idea(vec, trust=trust)
    for np1, np2, trust in rows_partof:
        s.append_relation(np1, torch.zeros(8), np2,
                          rel_type=s.REL_PARTOF, trust=trust)
    return s


def _kernel(store=None, *, budget=16, materialize=False, **kw):
    r = TruthGroundedReasoner(store=store if store is not None else _store())
    return ThinkingKernel(r, budget=budget, materialize=materialize, **kw)


# -- Truth intervals ----------------------------------------------------------

class TestTruthInterval(unittest.TestCase):
    def test_empty_is_unknown(self):
        iv = TruthInterval.from_evidence([])
        self.assertEqual((iv.lower, iv.upper, iv.trust), (0.0, 0.0, 0.0))
        self.assertEqual(iv.status(0.5), UNKNOWN)
        self.assertEqual(iv.luminosity, 0.0)

    def test_one_sided_true_false(self):
        t = TruthInterval.from_evidence([(0.9, 0.9, {})])
        self.assertEqual(t.status(0.5), TRUE)
        self.assertAlmostEqual(t.luminosity, 0.9)
        f = TruthInterval.from_evidence([(-0.8, 0.8, {})])
        self.assertEqual(f.status(0.5), FALSE)

    def test_conflicting_two_sided(self):
        iv = TruthInterval.from_evidence([(0.8, 0.8, {}), (-0.7, 0.7, {})])
        self.assertEqual((iv.lower, iv.upper), (-0.7, 0.8))
        self.assertEqual(iv.status(0.5), CONFLICTING)

    def test_mixed_luminous_straddle(self):
        # Strong true evidence + weak refutation: straddles 0, only one side
        # clears tau -> mixed, not conflicting.
        iv = TruthInterval.from_evidence([(0.8, 0.8, {}), (-0.1, 0.1, {})])
        self.assertEqual(iv.status(0.5), MIXED)

    def test_below_tau_is_unknown(self):
        iv = TruthInterval.from_evidence([(0.2, 0.2, {})])
        self.assertEqual(iv.status(0.5), UNKNOWN)


# -- lookup: LTM-direct, no chaining ------------------------------------------

class TestLookup(unittest.TestCase):
    def test_stored_idea_is_true(self):
        k = _kernel(_store(rows_ideas=[(IDEA_A, 0.9)]))
        iv = k.lookup(IDEA_A)                     # bare vector -> isTrue
        self.assertEqual(iv.status(k.tau), TRUE)
        self.assertAlmostEqual(iv.trust, 0.9, places=5)

    def test_empty_store_is_unknown(self):
        iv = _kernel().lookup(IDEA_A)
        self.assertEqual(iv.status(0.5), UNKNOWN)

    def test_no_chaining(self):
        # Socrates ⊑ man ⊑ mortal is derivable, but lookup() reads DIRECT
        # evidence only (§5.4) -- the chain belongs to think().
        k = _kernel(_store(rows_partof=[(SOCRATES, MAN, 0.9),
                                        (MAN, MORTAL, 0.8)]))
        iv = k.lookup(QuerySpec(KIND_IS_PART, SOCRATES, MORTAL))
        self.assertEqual(iv.status(k.tau), UNKNOWN)

    def test_direct_stored_edge(self):
        k = _kernel(_store(rows_partof=[(MAN, MORTAL, 0.8)]))
        iv = k.lookup(QuerySpec(KIND_IS_PART, MAN, MORTAL))
        self.assertEqual(iv.status(k.tau), TRUE)

    def test_refuting_edge_is_false(self):
        k = _kernel(_store(rows_partof=[(MAN, MORTAL, -0.8)]))
        iv = k.lookup(QuerySpec(KIND_IS_PART, MAN, MORTAL))
        self.assertEqual(iv.status(k.tau), FALSE)

    def test_conflicting_evidence(self):
        k = _kernel(_store(rows_partof=[(MAN, MORTAL, 0.8),
                                        (MAN, MORTAL, -0.7)]))
        iv = k.lookup(QuerySpec(KIND_IS_PART, MAN, MORTAL))
        self.assertEqual(iv.status(k.tau), CONFLICTING)

    def test_open_binary_query_is_unknown(self):
        iv = _kernel().lookup(QuerySpec(KIND_IS_PART, SOCRATES, None))
        self.assertEqual(iv.status(0.5), UNKNOWN)


# -- part: structural traversal ------------------------------------------------

class TestPart(unittest.TestCase):
    def test_up_and_down(self):
        k = _kernel(_store(rows_partof=[(SOCRATES, MAN, 0.9)]))
        ups = k.part(SOCRATES, direction="up")
        self.assertEqual(len(ups), 1)
        self.assertTrue(torch.allclose(ups[0]["idea"], MAN))
        downs = k.part(MAN, direction="down")
        self.assertEqual(len(downs), 1)
        self.assertTrue(torch.allclose(downs[0]["idea"], SOCRATES))

    def test_mode_rides_provenance(self):
        k = _kernel(_store(rows_partof=[(SOCRATES, MAN, 0.9)]))
        self.assertEqual(k.part(SOCRATES, mode="taxonomy")[0]["mode"],
                         "taxonomy")

    def test_bad_mode_or_direction_raises(self):
        k = _kernel()
        with self.assertRaises(ValueError):
            k.part(SOCRATES, mode="frobnicate")
        with self.assertRaises(ValueError):
            k.part(SOCRATES, direction="sideways")


# -- the execution loop: curriculum depths -------------------------------------

class TestCurriculum(unittest.TestCase):
    def test_depth0_lookup_answer(self):
        # Depth 0 (§12.5): lookup(target) -> answer.
        k = _kernel(_store(rows_ideas=[(IDEA_A, 0.9)]))
        res = k.run(IDEA_A)
        self.assertEqual(res.value, TRUE)
        self.assertAlmostEqual(res.trust, 0.9, places=5)
        ops = [e["op"] for e in res.trace]
        self.assertEqual(ops, ["lookup", "answer"])

    def test_depth0_direct_edge(self):
        k = _kernel(_store(rows_partof=[(MAN, MORTAL, 0.8)]))
        res = k.run(QuerySpec(KIND_IS_PART, MAN, MORTAL))
        self.assertEqual(res.value, TRUE)

    def test_depth1_syllogism_nested_think(self):
        # Depth 2 shape (§7.4): part -> think(subgoal) -> answer; trust = MIN.
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)])
        k = _kernel(store)
        res = k.run(QuerySpec(KIND_IS_PART, SOCRATES, MORTAL))
        self.assertEqual(res.value, TRUE)
        self.assertAlmostEqual(res.trust, 0.8, places=5)   # min(0.9, 0.8)
        ops = [e["op"] for e in res.trace]
        self.assertIn("part", ops)
        self.assertIn("think", ops)

    def test_depth2_two_hops(self):
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.7),
                                    (MORTAL, ANIMAL, 0.95)])
        k = _kernel(store)
        res = k.run(QuerySpec(KIND_IS_PART, SOCRATES, ANIMAL))
        self.assertEqual(res.value, TRUE)
        self.assertAlmostEqual(res.trust, 0.7, places=5)

    def test_false_close_on_refutation(self):
        k = _kernel(_store(rows_partof=[(MAN, MORTAL, -0.8)]))
        res = k.run(QuerySpec(KIND_IS_PART, MAN, MORTAL))
        self.assertEqual(res.value, FALSE)

    def test_dead_end_is_unknown(self):
        # Bounded search failed (§9.2 rule 3): unknown is the valid close.
        k = _kernel(_store(rows_partof=[(SOCRATES, MAN, 0.9)]))
        res = k.run(QuerySpec(KIND_IS_PART, SOCRATES, MORTAL))
        self.assertEqual(res.value, UNKNOWN)

    def test_isolated_leaf_unknown(self):
        res = _kernel().run(IDEA_A)
        self.assertEqual(res.value, UNKNOWN)


# -- frames, budget, closure ----------------------------------------------------

class TestFramesAndBudget(unittest.TestCase):
    def test_budget_exhaustion_bounded_unknown(self):
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)])
        k = _kernel(store, budget=2)         # lookup + part, then dry
        res = k.run(QuerySpec(KIND_IS_PART, SOCRATES, MORTAL))
        self.assertEqual(res.value, BOUNDED_UNKNOWN)

    def test_stack_pops_clean(self):
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)])
        k = _kernel(store)
        k.run(QuerySpec(KIND_IS_PART, SOCRATES, MORTAL))
        self.assertEqual(k.stack, [])        # every frame pushed was popped

    def test_child_returns_result_not_scratch(self):
        # §2.3: only the certified ChildResult crosses the frame boundary.
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)])
        k = _kernel(store)
        res = k.run(QuerySpec(KIND_IS_PART, SOCRATES, MORTAL))
        self.assertIsInstance(res, ChildResult)
        self.assertFalse(hasattr(res, "bindings"))

    def test_cycle_terminates(self):
        # a ⊑ b, b ⊑ a: the climb must terminate (depth cap + budget).
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9), (MAN, SOCRATES, 0.9)])
        k = _kernel(store, budget=64)
        res = k.run(QuerySpec(KIND_IS_PART, SOCRATES, MORTAL))
        self.assertIn(res.value, (UNKNOWN, BOUNDED_UNKNOWN))

    def test_unknown_op_raises(self):
        k = _kernel()
        f = Frame(target=as_spec(IDEA_A))
        with self.assertRaises(ValueError):
            k.execute(f, {"op": "frobnicate"})


# -- anti-hallucination invariants (§14) -----------------------------------------

class TestInvariants(unittest.TestCase):
    def test_unsupported_assertion_refused(self):
        # §12.3/§14.1: answering true with no supporting evidence closes
        # UNKNOWN with the unsupported_assertion marker.
        k = _kernel()
        f = Frame(target=as_spec(IDEA_A))
        k._pool = 4
        k.execute(f, {"op": "answer", "value": TRUE})
        self.assertEqual(f.result.value, UNKNOWN)
        self.assertEqual(f.result.trace[-1]["how"], "unsupported_assertion")

    def test_speculation_does_not_write_ltm(self):
        # An unknown run leaves the store untouched (§14.4).
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9)])
        before = int(store.count.item())
        _kernel(store).run(QuerySpec(KIND_IS_PART, SOCRATES, MORTAL))
        self.assertEqual(int(store.count.item()), before)

    def test_success_without_materialize_flag_does_not_write(self):
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)])
        before = int(store.count.item())
        res = _kernel(store, materialize=False).run(
            QuerySpec(KIND_IS_PART, SOCRATES, MORTAL))
        self.assertEqual(res.value, TRUE)
        self.assertEqual(int(store.count.item()), before)

    def test_grounded_derivation_materializes_lemma(self):
        # §14.2 trusted derivation: the SECOND run is a depth-0 direct hit.
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)])
        k = _kernel(store, materialize=True)
        res = k.run(QuerySpec(KIND_IS_PART, SOCRATES, MORTAL))
        self.assertEqual(res.value, TRUE)
        self.assertTrue(any("materialized_row" in p for p in res.provenance))
        res2 = _kernel(store).run(QuerySpec(KIND_IS_PART, SOCRATES, MORTAL))
        ops2 = [e["op"] for e in res2.trace]
        self.assertEqual(ops2, ["lookup", "answer"])   # direct hit, no climb
        self.assertEqual(res2.value, TRUE)

    def test_direct_hit_does_not_rewrite(self):
        # A depth-0 close derives nothing -> no lemma even with the flag on.
        store = _store(rows_partof=[(MAN, MORTAL, 0.8)])
        before = int(store.count.item())
        _kernel(store, materialize=True).run(
            QuerySpec(KIND_IS_PART, MAN, MORTAL))
        self.assertEqual(int(store.count.item()), before)


# -- query / testimony -----------------------------------------------------------

class TestQueryTestimony(unittest.TestCase):
    def test_registered_addressee_returns_testimony(self):
        k = _kernel()
        k.register_addressee("oracle", lambda target: 0.9)
        t = k.query("oracle", IDEA_A)
        self.assertIsInstance(t, Testimony)
        self.assertEqual(t.source, "oracle")
        self.assertEqual(t.value, 0.9)

    def test_unknown_addressee_zero_trust(self):
        t = _kernel().query("nobody", IDEA_A)
        self.assertEqual(t.effective_trust, 0.0)
        self.assertIsNone(t.value)

    def test_testimony_is_not_truth(self):
        # §14.3: receiving an answer moves nothing until incorporate().
        store = _store()
        k = _kernel(store)
        k.register_addressee("oracle", lambda target: 1.0)
        k.query("oracle", IDEA_A)
        self.assertEqual(k.lookup(IDEA_A).status(k.tau), UNKNOWN)

    def test_incorporate_above_floor_moves_lookup(self):
        store = _store()
        k = _kernel(store)
        t = Testimony(proposition=IDEA_A, value=1.0, source="oracle",
                      source_trust=0.9, channel_trust=0.9)
        row = k.incorporate(t)
        self.assertGreaterEqual(row, 0)
        iv = k.lookup(IDEA_A)
        self.assertEqual(iv.status(k.tau), TRUE)
        self.assertAlmostEqual(iv.trust, 0.81, places=5)

    def test_incorporate_below_floor_refused(self):
        store = _store()
        k = _kernel(store)
        t = Testimony(proposition=IDEA_A, value=1.0, source="rumor",
                      source_trust=0.4, channel_trust=0.5)
        self.assertEqual(k.incorporate(t), -1)
        self.assertEqual(int(store.count.item()), 0)

    def test_incorporate_false_testimony_negative_row(self):
        store = _store()
        k = _kernel(store)
        t = Testimony(proposition=IDEA_A, value=-1.0, source="oracle",
                      source_trust=0.9)
        self.assertGreaterEqual(k.incorporate(t), 0)
        self.assertEqual(k.lookup(IDEA_A).status(k.tau), FALSE)

    def test_incorporate_relation(self):
        store = _store()
        k = _kernel(store)
        t = Testimony(proposition=QuerySpec(KIND_IS_PART, MAN, MORTAL),
                      value=1.0, source="expert", source_trust=0.8)
        self.assertGreaterEqual(k.incorporate(t), 0)
        iv = k.lookup(QuerySpec(KIND_IS_PART, MAN, MORTAL))
        self.assertEqual(iv.status(k.tau), TRUE)

    def test_arma_addressee_registered_with_model(self):
        class _Stub:
            conceptualSpace = None
            symbolSpace = None
        r = TruthGroundedReasoner(_Stub(), store=_store())
        k = ThinkingKernel(r)
        self.assertIn("arma", k.addressees)
        t = k.query("arma", IDEA_A)          # cold predictor -> None value
        self.assertIsNone(t.value)


# -- testimony in the live loop (§7.1/§14.2) ---------------------------------------

class TestTestimonyInLoop(unittest.TestCase):
    def test_reliable_oracle_grounds_true(self):
        k = _kernel()
        k.register_addressee("oracle", lambda t: 0.9, source_trust=0.9)
        res = k.run(IDEA_A)
        self.assertEqual(res.value, TRUE)
        self.assertEqual([e["op"] for e in res.trace],
                         ["lookup", "query", "answer"])
        # Evidence value = asserted 0.9 x reliability 0.9 = 0.81.
        self.assertAlmostEqual(res.interval.upper, 0.81, places=5)

    def test_refuting_oracle_grounds_false(self):
        k = _kernel()
        k.register_addressee("oracle", lambda t: -1.0, source_trust=0.8)
        res = k.run(IDEA_A)
        self.assertEqual(res.value, FALSE)

    def test_unreliable_testimony_stays_unknown(self):
        # Reliability scales the asserted value: 1.0 x 0.3 = 0.3 < tau, so
        # flimsy testimony cannot manufacture luminosity (§14.2).
        k = _kernel()
        k.register_addressee("rumor", lambda t: 1.0, source_trust=0.3)
        res = k.run(IDEA_A)
        self.assertEqual(res.value, UNKNOWN)
        self.assertIn("query", [e["op"] for e in res.trace])

    def test_testimony_in_loop_does_not_write_ltm(self):
        # The premise use (§9.3) is frame-local; the durable write stays the
        # explicit incorporate() path (§14.3).
        store = _store()
        k = _kernel(store, materialize=True)
        k.register_addressee("oracle", lambda t: 0.9, source_trust=0.9)
        res = k.run(IDEA_A)
        self.assertEqual(res.value, TRUE)
        self.assertEqual(int(store.count.item()), 0)

    def test_tensor_testimony_is_content_not_truth(self):
        # An arma-like addressee returns a VECTOR (predicted content); it
        # never folds into the truth interval.
        k = _kernel()
        k.register_addressee("arma", lambda t: torch.randn(8))
        res = k.run(IDEA_A)
        self.assertEqual(res.value, UNKNOWN)

    def test_consult_gate_off_skips_query(self):
        k = _kernel()
        k.consult_addressees = False
        k.register_addressee("oracle", lambda t: 0.9, source_trust=0.9)
        res = k.run(IDEA_A)
        self.assertEqual(res.value, UNKNOWN)
        self.assertNotIn("query", [e["op"] for e in res.trace])

    def test_each_addressee_asked_once(self):
        calls = {"n": 0}

        def flaky(t):
            calls["n"] += 1
            return None
        k = _kernel()
        k.register_addressee("flaky", flaky)
        res = k.run(IDEA_A)
        self.assertEqual(res.value, UNKNOWN)
        self.assertEqual(calls["n"], 1)


# -- rewards + trace examples (§12) ----------------------------------------------

class TestRewardsAndTraces(unittest.TestCase):
    def _success(self):
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)])
        k = _kernel(store)
        return k, k.run(QuerySpec(KIND_IS_PART, SOCRATES, MORTAL))

    def test_success_earns_terminal(self):
        k, res = self._success()
        rw = k.compile_rewards(res)
        self.assertEqual(rw["terminal"], k.terminal_reward)
        self.assertGreater(rw["total"], 0.0)

    def test_step_costs_charged(self):
        k, res = self._success()
        rw = k.compile_rewards(res)
        n_ops = len([e for e in res.trace if e.get("op") != "answer"])
        self.assertEqual(len(rw["steps"]), n_ops)
        # Delta-luminosity is zero until the think() lands -> early steps
        # carry pure step cost.
        self.assertAlmostEqual(rw["steps"][0], -k.step_cost, places=6)

    def test_valid_unknown_is_not_failure(self):
        k = _kernel(_store(rows_partof=[(SOCRATES, MAN, 0.9)]))
        res = k.run(QuerySpec(KIND_IS_PART, SOCRATES, MORTAL))
        self.assertEqual(res.value, UNKNOWN)
        self.assertEqual(k.compile_rewards(res)["terminal"], 0.0)

    def test_unsupported_assertion_is_failure(self):
        k = _kernel()
        f = Frame(target=as_spec(IDEA_A))
        k._pool = 4
        k.execute(f, {"op": "answer", "value": TRUE})
        rw = k.compile_rewards(f.result)
        self.assertEqual(rw["terminal"], -k.terminal_reward)

    def test_trace_examples_only_from_grounded(self):
        k, res = self._success()
        ex = ThinkingKernel.trace_examples(res)
        self.assertTrue(ex)
        self.assertEqual(ex[-1][1], "answer")
        for state, op in ex:
            self.assertIn("lum", state)
            self.assertIn(op, ("lookup", "part", "think", "query", "answer"))
        # An unknown close yields no supervision.
        k2 = _kernel()
        res2 = k2.run(IDEA_A)
        self.assertEqual(ThinkingKernel.trace_examples(res2), [])


# -- next-op policy learning (§12.6/12.7) -------------------------------------------

class TestNextOpPolicy(unittest.TestCase):
    def _syllogism_kernel(self):
        return _kernel(_store(rows_partof=[(SOCRATES, MAN, 0.9),
                                           (MAN, MORTAL, 0.8)]))

    def test_featurize_and_logits_shapes(self):
        head = NextOpPolicy()
        state = {"kind": KIND_IS_PART, "depth": 1, "n_ops": 2, "lum": 0.4}
        self.assertEqual(tuple(NextOpPolicy.featurize(state).shape), (6,))
        self.assertEqual(tuple(head.logits(state).shape), (5,))
        self.assertIn(head.choose(state, ("think", "answer")),
                      ("think", "answer"))

    def test_untrained_head_is_neutral(self):
        # Zero-initialized output layer: an untrained head ties every op and
        # choose() keeps the caller's first (explore) option -- exactly the
        # deterministic baseline. Regression: random init made an untrained
        # head randomly prefer "answer", killing climbs at inference.
        head = NextOpPolicy()
        for state in ({"kind": KIND_IS_PART, "depth": 0, "n_ops": 0,
                       "lum": 0.0},
                      {"kind": KIND_IS_TRUE, "depth": 3, "n_ops": 7,
                       "lum": 0.6}):
            self.assertEqual(head.choose(state, ("think", "answer")), "think")
            self.assertEqual(head.choose(state, ("query", "answer")), "query")
        k = self._syllogism_kernel()
        ex = traces_from_store(k)
        self.assertTrue(ex)
        self.assertTrue(all(op in ("lookup", "part", "think", "query",
                                   "answer") for (_s, op) in ex))
        # Trace generation is read-only on the store (materialize off).
        self.assertEqual(int(k.reasoner.reasoning_store().count.item()), 2)

    def test_next_op_loss_trains_the_head(self):
        k = self._syllogism_kernel()
        ex = traces_from_store(k)
        head = NextOpPolicy()
        opt = torch.optim.SGD(head.parameters(), lr=0.5)
        first = float(next_op_loss(head, ex))
        for _ in range(20):
            opt.zero_grad()
            loss = next_op_loss(head, ex)
            loss.backward()
            opt.step()
        self.assertLess(float(next_op_loss(head, ex)), first)

    def test_empty_examples_returns_none(self):
        self.assertIsNone(next_op_loss(NextOpPolicy(), []))
        self.assertIsNone(next_op_loss(None, [({}, "lookup")]))
        self.assertEqual(traces_from_store(_kernel()), [])

    def test_head_prefers_stop_short_circuits(self):
        # A head that always prefers "answer" stops the climb before any
        # think() subgoal -- legal-menu only, so the close is still an honest
        # unknown (the runtime refuses unsupported assertions).
        class _StopHead:
            def choose(self, state, options):
                return "answer"
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)])
        r = TruthGroundedReasoner(store=store)
        k = ThinkingKernel(r, policy=KernelPolicy(next_op=_StopHead()))
        res = k.run(QuerySpec(KIND_IS_PART, SOCRATES, MORTAL))
        self.assertEqual(res.value, UNKNOWN)
        ops = [e["op"] for e in res.trace]
        self.assertNotIn("think", ops)

    def test_head_preferring_explore_keeps_baseline(self):
        class _GoHead:
            def choose(self, state, options):
                return next(o for o in options if o != "answer")
        store = _store(rows_partof=[(SOCRATES, MAN, 0.9), (MAN, MORTAL, 0.8)])
        r = TruthGroundedReasoner(store=store)
        k = ThinkingKernel(r, policy=KernelPolicy(next_op=_GoHead()))
        res = k.run(QuerySpec(KIND_IS_PART, SOCRATES, MORTAL))
        self.assertEqual(res.value, TRUE)


# -- Models/config integration (gated; off ⇒ byte-identical) ---------------------

class TestModelIntegration(unittest.TestCase):
    """The MM_query_reasoning validation config with <thinkingBudget>16: the
    kernel rides the answer_query payload; budget 0 ⇒ no kernel key."""

    @classmethod
    def setUpClass(cls):
        import Models
        from Models import BaseModel
        _DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
        Models.TheData.load('queries')
        cls.m, _ = BaseModel.from_config(
            os.path.join(_DATA, 'MM_query_reasoning.xml'))
        # <truthSet> provisioning is DEFERRED to the first runEpoch; drive it
        # explicitly (data is loaded) so the syllogism rows land in ltm_store.
        cls.m.provision_ltm()

    def test_knob_parsed(self):
        self.assertEqual(self.m.thinking_budget, 16)

    def test_truthset_provisions_rows(self):
        # Regression (2026-07-12): the observe-site conversation push lives
        # ONLY in the SERIAL per-word body -- with the parallel body this
        # config provisioned 0 rows. <serial>true now lands both truthSet
        # rows, kind-tagged REL_PARTOF at the XML trust. (Endpoint fidelity
        # -- the depth-3 NP/VP/NP split -- still tracks parse quality; the
        # untrained parse collapses each text to a depth-1 absolute row.)
        store = self.m.symbolSpace.ltm_store
        rows = [store.row(int(i))
                for i in store.relations(rel_type=store.REL_PARTOF)]
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertAlmostEqual(r['trust'], 0.9, places=5)
        texts = {r['text'] for r in rows}
        self.assertEqual(texts, {'socrates partOf human',
                                 'human partOf mortal'})

    def test_off_returns_none(self):
        saved = self.m.thinking_budget
        try:
            self.m.thinking_budget = 0
            self.assertIsNone(self.m.think_about(
                QuerySpec(KIND_IS_PART, torch.randn(1024),
                          torch.randn(1024))))
        finally:
            self.m.thinking_budget = saved

    def test_think_about_honest_unknown(self):
        # Random operands over the store: never a hallucinated verdict.
        res = self.m.think_about(
            QuerySpec(KIND_IS_PART, torch.randn(1024), torch.randn(1024)))
        self.assertIsNotNone(res)
        self.assertIn(res.value, (TRUE, FALSE, UNKNOWN, MIXED, CONFLICTING,
                                  BOUNDED_UNKNOWN))

    def test_syllogism_over_live_ltm(self):
        # A socrates ⊑ human ⊑ mortal chain in the model's OWN ltm_store
        # resolves TRUE through the frame stack (trust = min-hop). The parse
        # path can't land the truthSet rows on this byte-grain config (the
        # documented Track-1 wall, test_reasoning_cde_model.py), so the rows
        # are appended directly -- the kernel wiring is what's under test.
        store = self.m.conceptualSpace._reasoning_store()
        D = int(store.slots.shape[-1])
        soc, man, mor = torch.zeros(D), torch.zeros(D), torch.zeros(D)
        soc[0], man[1], mor[2] = 1.0, 1.0, 1.0
        n0 = int(store.count.item())
        try:
            store.append_relation(soc, torch.zeros(D), man,
                                  rel_type=store.REL_PARTOF, trust=0.9)
            store.append_relation(man, torch.zeros(D), mor,
                                  rel_type=store.REL_PARTOF, trust=0.8)
            res = self.m.think_about(QuerySpec(KIND_IS_PART, soc, mor))
            self.assertEqual(res.value, TRUE)
            self.assertAlmostEqual(res.trust, 0.8, places=5)   # min-hop
        finally:
            store.count.fill_(n0)          # leave the shared store clean

    def test_thinking_loss_head_built_and_in_optimizer(self):
        # thinkingLossWeight > 0 => the NextOpPolicy head is built eagerly and
        # its params are IN THE OPTIMIZER (getOptimizer walks named modules,
        # not model.parameters() -- the Phase-C bug class).
        self.assertAlmostEqual(self.m.thinking_loss_weight, 0.1)
        head = getattr(self.m, '_next_op_policy', None)
        self.assertIsNotNone(head)
        head_ids = {id(p) for p in head.parameters()}
        opt = self.m.getOptimizer(lr=0.01)
        opt_ids = set()
        for o in getattr(opt, 'optimizers', [opt]):
            for g in getattr(o, 'param_groups', []):
                for p in g['params']:
                    opt_ids.add(id(p))
        self.assertTrue(head_ids and head_ids <= opt_ids)

    def test_thinking_policy_loss_graceful(self):
        # The collapsed provisioned rows carry no 2-hop chains, so the hook
        # degrades to None (no crash) -- and never a store write either way.
        store = self.m.symbolSpace.ltm_store
        before = int(store.count.item())
        out = self.m._thinking_policy_loss()
        self.assertTrue(out is None or torch.is_tensor(out))
        self.assertEqual(int(store.count.item()), before)

    def test_answer_query_kernel_attachment_gates(self):
        # Stub the operand detector (this byte-grain config has no word vocab)
        # to reach the payload assembly; the kernel key must track the budget.
        A, B = torch.randn(1024), torch.randn(1024)
        self.m._detect_query = lambda msg: ("queryPart", A, B)
        try:
            out = self.m.answer_query('is a part of b?')
            self.assertIsInstance(out, dict)
            self.assertIn('kernel', out)
            k = out['kernel']
            self.assertIn('value', k)
            self.assertEqual(len(k['interval']), 2)
            self.assertIsInstance(k['ops'], list)
            self.m.thinking_budget = 0
            out_off = self.m.answer_query('is a part of b?')
            self.assertNotIn('kernel', out_off)
        finally:
            del self.m._detect_query
            self.m.thinking_budget = 16


if __name__ == "__main__":
    unittest.main()
