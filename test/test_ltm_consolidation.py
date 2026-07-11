"""LTM consolidation -- stages 2-6: the unified TernaryTruthStore wired in
(doc/specs/mereological-order-raising.md "Truth / Ideas processing", "Two
consolidation questions"; Alec 2026-06-18).

The discourse LTM (InterSentenceLayer end-state chain) and the
RelativeTruthStore are COMBINED into ONE ``Layers.TernaryTruthStore`` on
SymbolSubSpace (``ltm_store``) behind the dark gate ``<ltmConsolidation>``
(default OFF -> the legacy two-store path, byte-identical):

  * CONSTRUCTION: gate ON -> ltm_store present, relative_store absent; gate OFF
    -> the reverse (the legacy RTS, no ltm_store).
  * WRITES (observe site): each end-state appends a ternary row -- depth==1 an
    absolute idea, depth>=3 a relation (NP1=idea1, VP=predicate, NP2=idea2)
    with the per-row scalar trust.
  * ROUTING: the ineffable branch of ``_route_learned_relation`` returns the
    ``('idea', -1)`` marker WITHOUT writing a separate store (the row already
    lives in ltm_store from observe).
  * REASONING: ``reason`` / ``verify_relation`` read the ltm_store on the
    CONTENT slice (unscaled vectors + a separate trust column -- no un-baking).
  * SURVIVE-RESET + PERSISTENCE: ltm_store is an attribute (not in self.layers)
    with only a lowercase ``reset()`` -> survives every Reset cascade and rides
    the state_dict.
  * XML PROVISIONING: a ``<truthSet>`` is appended at load (``provision_ltm``).

Flag-off must stay byte-identical (the existing suite covers that); these tests
cover the gate-ON path.
"""

from __future__ import annotations

import os
import sys
import unittest
import warnings

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_DATA_DIR = os.path.join(_PROJECT, "data")
_OFF_CONFIG = os.path.join(_DATA_DIR, "MM_xor_fixture.xml")
_ON_CONFIG = os.path.join(_DATA_DIR, "MM_ltm_consolidation_fixture.xml")
# SERIAL fixture for the FOLLOW-UPS: real-parse
# provisioning (Change 3) needs the per-word serial forward to fire the
# observe-site store-append, and FU3 (Change 2) needs a discourse. Turns ON
# BOTH <ltmConsolidation> AND <training><sentencePrediction>.
_SERIAL_CONFIG = os.path.join(
    _DATA_DIR, "MM_ltm_consolidation_serial_fixture.xml")
# Consolidated but STATEFUL (<stateless>false</stateless>): a checkpoint's
# ORIGIN_USER rows are durable and survive a state_dict reload.
_STATEFUL_CONFIG = os.path.join(
    _DATA_DIR, "MM_ltm_consolidation_stateful_fixture.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")


def _make_model(config):
    import Models
    import Language
    from util import init_config
    init_config(path=config, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(config)
    Models.TheData.load("xor")
    m.eval()
    return m


def _make_model_provisioned(config):
    """Build the model, load data, and TRIGGER provisioning via the real
    forward (provisioning is now lazy at first use, not at construction)."""
    m = _make_model(config)
    m.provision_ltm()
    return m


# -- stage 2: gate + construction ------------------------------------------

class TestConstruction(unittest.TestCase):
    def test_gate_off_builds_relative_store_only(self):
        m = _make_model(_OFF_CONFIG)
        ss = m.symbolSpace
        self.assertFalse(m.ltm_consolidation)
        self.assertIsNotNone(ss.relative_store,
                             "gate OFF -> the legacy RelativeTruthStore")
        self.assertIsNone(ss.ltm_store,
                          "gate OFF -> no unified ltm_store")
        self.assertFalse(getattr(m.conceptualSpace, "_ltm_consolidation",
                                 False))

    def test_gate_on_builds_ltm_store_only(self):
        from Layers import TernaryTruthStore
        m = _make_model(_ON_CONFIG)
        ss = m.symbolSpace
        self.assertTrue(m.ltm_consolidation)
        self.assertIsInstance(ss.ltm_store, TernaryTruthStore)
        self.assertIsNone(ss.relative_store,
                          "gate ON -> RelativeTruthStore is retired")
        self.assertTrue(getattr(m.conceptualSpace, "_ltm_consolidation",
                                False))

    def test_ltm_store_widths(self):
        m = _make_model(_ON_CONFIG)
        store = m.symbolSpace.ltm_store
        # nDim is the FULL idea/event width (muxed); content_width is the
        # content/symbol width reasoning slices to.
        self.assertEqual(int(store.nDim), int(m.symbolSpace.muxedSize))
        self.assertGreater(int(store.content_width), 0)

    def test_ltm_store_registers_as_submodule_not_in_layers(self):
        m = _make_model(_ON_CONFIG)
        ss = m.symbolSpace
        store = ss.ltm_store
        # NOT in self.layers (keeps it OUT of the Reset cascade) ...
        self.assertNotIn(store, list(ss.layers))
        # ... but IS a registered submodule (its buffers ride the state_dict).
        self.assertTrue(any(mod is store for mod in ss.modules()))


# -- Change 3 (FU1+FU2): real-parse XML truthSet provisioning --------------
#
# Provisioning now runs the <truthSet> texts through the REAL forward (so the
# encoding is a real parse, NOT a mean-pool placeholder) and lands one row per
# truth via the Change-1 observe-site store-append. This needs the serial
# forward to fire, so these tests use the SERIAL fixture and trigger
# provisioning explicitly after data load (it is otherwise lazy at first
# runEpoch). The old mean-pool / NP1==VP==NP2 assertions are GONE.

class TestProvisioning(unittest.TestCase):
    def test_provisioning_is_lazy_not_at_construction(self):
        # Real-parse provisioning needs loaded data, so it no longer runs at
        # construction: the store is EMPTY until provision_ltm / first epoch.
        m = _make_model(_SERIAL_CONFIG)
        self.assertEqual(len(m.symbolSpace.ltm_store), 0)
        self.assertFalse(m._ltm_provisioned)

    def test_truthset_rows_land_via_real_forward(self):
        from Layers import TernaryTruthStore as T
        m = _make_model_provisioned(_SERIAL_CONFIG)
        store = m.symbolSpace.ltm_store
        # The fixture provisions 3 truths: one absolute idea ("cat"), one
        # implies ("fire causes smoke"), one partOf ("paw partOf cat"). Each
        # lands exactly one real parsed end-state row, in order.
        self.assertEqual(len(store), 3)
        # rel_type is driven by the entry's KIND tag (the absolute idea has no
        # kind -> REL_NONE; the kinds override whatever the surface parse did).
        self.assertEqual(store.ideas().tolist(), [0])
        self.assertEqual(sorted(store.relations().tolist()), [1, 2])
        self.assertEqual(store.relations(T.REL_IMPLIES).tolist(), [1])
        self.assertEqual(store.relations(T.REL_PARTOF).tolist(), [2])

    def test_provisioned_trust_and_earliest_timestamps(self):
        m = _make_model_provisioned(_SERIAL_CONFIG)
        store = m.symbolSpace.ltm_store
        # Each row's trust is OVERWRITTEN with the XML trust.
        self.assertAlmostEqual(store.row(0)["trust"], 0.9, places=5)
        self.assertAlmostEqual(store.row(1)["trust"], 0.8, places=5)
        self.assertAlmostEqual(store.row(2)["trust"], 0.7, places=5)
        # Provisioned rows take the earliest monotonic ticks; the clock then
        # continues past them.
        self.assertEqual([store.row(i)["timestamp"] for i in range(3)],
                         [0.0, 1.0, 2.0])
        nxt = store.append_idea(torch.ones(store.nDim))
        self.assertEqual(store.row(nxt)["timestamp"], 3.0)

    def test_provisioned_rows_are_real_encodings(self):
        m = _make_model_provisioned(_SERIAL_CONFIG)
        store = m.symbolSpace.ltm_store
        # The REAL parse lands a non-zero NP1 for every row (no all-zero row,
        # and no mean-pool placeholder -- it is the actual parsed end-state).
        for i in range(len(store)):
            self.assertGreater(float(store.row(i)["np1"].norm()), 0.0)

    def test_provision_idempotent_count_via_runepoch_trigger(self):
        # The lazy trigger at first runEpoch provisions exactly once (the
        # _ltm_provisioned guard), then conversation appends continue past it.
        m = _make_model(_SERIAL_CONFIG)
        store = m.symbolSpace.ltm_store
        self.assertEqual(len(store), 0)
        opt = m.getOptimizer(lr=0.01)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m.runEpoch(optimizer=opt, batchSize=1, split="train",
                       max_batches=1)
        # provisioning fired (3 truths) + at least the batch's own append.
        self.assertTrue(m._ltm_provisioned)
        self.assertGreaterEqual(len(store), 3)
        # The first three rows are the provisioned truths (earliest ticks).
        self.assertAlmostEqual(store.row(0)["trust"], 0.9, places=5)

    def test_provision_noop_when_gate_off(self):
        m = _make_model(_OFF_CONFIG)
        # No ltm_store -> provision_ltm is a no-op and returns 0.
        self.assertEqual(m.provision_ltm(), 0)


# -- stage 3: observe-site writes (slot mapping + trust) -------------------

class TestObserveWrites(unittest.TestCase):
    def _drive_observe(self, m, depths, payloads, tetralemmas):
        """Replay the exact observe-site append logic (the host-side block in
        ``_forward_body_per_word``) against the model's ltm_store. The live
        forward needs sentencePrediction + relative end-states; this isolates
        the write contract under test."""
        from Layers import TernaryTruthStore as T
        store = m.symbolSpace.ltm_store
        for b in range(len(payloads)):
            payload = payloads[b]
            if payload is None or payload.shape[0] < 1:
                continue
            d = max(1, min(int(depths[b]), int(payload.shape[0])))
            tet = tetralemmas[b] if tetralemmas is not None else None
            trust = float(tet) if tet is not None else 0.0
            if d >= 3:
                store.append_relation(payload[d - 2], payload[d - 1],
                                      payload[0], rel_type=T.REL_OTHER,
                                      trust=trust)
            else:
                store.append_idea(payload[0], trust=trust)
        return store

    def test_absolute_depth1_appends_idea(self):
        from Layers import TernaryTruthStore as T
        m = _make_model(_ON_CONFIG)
        m.symbolSpace.ltm_store.reset()
        D = m.symbolSpace.ltm_store.nDim
        p = torch.zeros(1, D)
        p[0, 0] = 3.0
        store = self._drive_observe(m, [1], [p], [0.5])
        self.assertEqual(len(store), 1)
        r = store.row(0)
        self.assertEqual(r["rel_type"], T.REL_NONE)
        self.assertAlmostEqual(r["trust"], 0.5, places=5)
        self.assertAlmostEqual(float(r["np1"][0]), 3.0, places=5)

    def test_relative_depth3_slot_mapping(self):
        from Layers import TernaryTruthStore as T
        m = _make_model(_ON_CONFIG)
        m.symbolSpace.ltm_store.reset()
        D = m.symbolSpace.ltm_store.nDim
        # newest-at-slot-0: slot0=idea2, slot1=idea1, slot2=predicate.
        p = torch.zeros(3, D)
        p[0, 0] = 1.0   # idea2 (newest)
        p[1, 1] = 1.0   # idea1
        p[2, 2] = 1.0   # predicate (oldest)
        store = self._drive_observe(m, [3], [p], [0.9])
        self.assertEqual(len(store), 1)
        r = store.row(0)
        self.assertEqual(r["rel_type"], T.REL_OTHER)
        self.assertAlmostEqual(r["trust"], 0.9, places=5)
        # NP1=idea1, VP=predicate, NP2=idea2.
        self.assertAlmostEqual(float(r["np1"][1]), 1.0, places=5)
        self.assertAlmostEqual(float(r["vp"][2]), 1.0, places=5)
        self.assertAlmostEqual(float(r["np2"][0]), 1.0, places=5)

    def test_none_tet_defaults_zero_trust(self):
        m = _make_model(_ON_CONFIG)
        m.symbolSpace.ltm_store.reset()
        D = m.symbolSpace.ltm_store.nDim
        p = torch.zeros(1, D)
        p[0, 0] = 1.0
        store = self._drive_observe(m, [1], [p], None)
        self.assertAlmostEqual(store.row(0)["trust"], 0.0, places=6)


# -- stage 4: routing (ineffable -> marker, no separate store write) -------

class TestRouting(unittest.TestCase):
    def _accept_but_ineffable(self, cs):
        cs._learn_score_children_in_codebook = lambda i1, i2: 0.0
        cs._learn_score_is_truth_obvious = lambda rel: 1.0
        cs._learn_score_resolves_contradiction = lambda rel: 1.0
        cs.truth_criterion = 0.0

    def test_ineffable_returns_marker_no_store_write(self):
        m = _make_model(_ON_CONFIG)
        cs = m.conceptualSpace
        self.assertTrue(getattr(cs, "_ltm_consolidation", False))
        self._accept_but_ineffable(cs)
        cs._tetralemma_trust = lambda rel, truth_set=None: (0.8, 0.1, 0.1, 0.0)
        D = int(cs.nDim)
        predicate = torch.zeros(D); predicate[0] = 1.0
        idea1 = torch.zeros(D); idea1[1] = 1.0
        idea2 = torch.zeros(D); idea2[2] = 1.0
        store = m.symbolSpace.ltm_store
        n_before = len(store)
        out = cs._maybe_learn_relation(predicate, idea1, idea2)
        # ('idea', -1): the explicit-knowing home is the unified ltm_store; the
        # ineffable branch does NOT write a distinct row (the row was already
        # appended at the observe site).
        self.assertIsInstance(out, tuple)
        self.assertEqual(out, ("idea", -1))
        self.assertEqual(len(store), n_before,
                         "ineffable branch must NOT write a separate store")


# -- stage 4/5: reasoning + verification over the ltm_store ----------------

class TestReasonOverLtm(unittest.TestCase):
    def test_reason_selects_ltm_store_when_consolidated(self):
        from Layers import TernaryTruthStore as T
        m = _make_model(_ON_CONFIG)
        cs = m.conceptualSpace
        store = m.symbolSpace.ltm_store
        store.reset()
        D = store.nDim
        cw = min(int(store.content_width), D)
        A = torch.zeros(D); A[0] = 1.0
        P = torch.zeros(D); P[2] = 1.0
        B = torch.zeros(D); B[1] = 1.0
        idx = store.append_relation(A, P, B, rel_type=T.REL_IMPLIES, trust=0.8)
        # No explicit store= -> the consolidation path picks the ltm_store.
        q = torch.zeros(cw); q[0] = 1.0
        res = cs.reason(q, 0.5, parthood_threshold=0.5)
        self.assertEqual(len(res["derived"]), 1)
        d = res["derived"][0]
        self.assertEqual(d["source"], idx)
        self.assertAlmostEqual(d["trust"], 0.4, places=5)   # t1*t2 = 0.8*0.5
        self.assertAlmostEqual(res["luminosity_gain"], 0.4, places=5)

    def test_reason_skips_absolute_idea_rows(self):
        from Layers import TernaryTruthStore as T
        m = _make_model(_ON_CONFIG)
        cs = m.conceptualSpace
        store = m.symbolSpace.ltm_store
        store.reset()
        D = store.nDim
        cw = min(int(store.content_width), D)
        # An absolute idea row covering the query must NOT fire (only relation
        # rows are reasoned over).
        idea = torch.zeros(D); idea[0] = 1.0
        store.append_idea(idea, trust=0.9)
        q = torch.zeros(cw); q[0] = 1.0
        res = cs.reason(q, 1.0, parthood_threshold=0.5)
        self.assertEqual(len(res["derived"]), 0)

    def test_verify_relation_writes_scalar_no_rebake(self):
        from Layers import TernaryTruthStore as T
        m = _make_model(_ON_CONFIG)
        cs = m.conceptualSpace
        store = m.symbolSpace.ltm_store
        store.reset()
        D = store.nDim
        cw = min(int(store.content_width), D)
        A = torch.zeros(D); A[0] = 1.0
        P = torch.zeros(D); P[2] = 1.0
        B = torch.zeros(D); B[1] = 1.0
        idx = store.append_relation(A, P, B, rel_type=T.REL_IMPLIES, trust=0.8)
        np1_before = store.row(idx)["np1"].clone()
        ante = torch.zeros(cw); ante[0] = 1.0
        cons = torch.zeros(cw); cons[1] = 1.0
        new = cs.verify_relation(idx, [(ante, cons)], support_weight=0.5)
        # full support -> nudged toward +1: 0.5*0.8 + 0.5*1.0 = 0.9.
        self.assertAlmostEqual(new, 0.9, places=5)
        self.assertAlmostEqual(store.row(idx)["trust"], 0.9, places=5)
        # The stored vector is UNCHANGED (no magnitude re-bake -- unscaled).
        self.assertTrue(torch.allclose(store.row(idx)["np1"], np1_before,
                                       atol=1e-6))

    def test_verify_absolute_row_is_noop(self):
        m = _make_model(_ON_CONFIG)
        cs = m.conceptualSpace
        store = m.symbolSpace.ltm_store
        store.reset()
        D = store.nDim
        idea = torch.zeros(D); idea[0] = 1.0
        idx = store.append_idea(idea, trust=0.6)
        out = cs.verify_relation(idx, [(idea, idea)], support_weight=0.5)
        self.assertAlmostEqual(out, 0.6, places=5)
        self.assertAlmostEqual(store.row(idx)["trust"], 0.6, places=5)


# -- stage 5: survive-Reset + state_dict round-trip ------------------------

class TestSurviveResetAndPersistence(unittest.TestCase):
    def test_survives_every_reset(self):
        m = _make_model(_ON_CONFIG)
        store = m.symbolSpace.ltm_store
        # Provisioning is lazy now, so seed a couple of rows directly to have
        # something the Reset cascade could (wrongly) clear.
        store.append_idea(torch.ones(store.nDim), trust=0.5)
        store.append_relation(torch.ones(store.nDim), torch.ones(store.nDim),
                              torch.ones(store.nDim), trust=0.3)
        n0 = len(store)
        self.assertGreater(n0, 0)
        m.symbolSpace.Reset(hard=True)
        self.assertEqual(len(store), n0, "SymbolSubSpace.Reset must not clear")
        m.symbolSpace.soft_reset()
        self.assertEqual(len(store), n0, "soft_reset must not clear")
        m.conceptualSpace.Reset(hard=True)
        self.assertEqual(len(store), n0, "CS.Reset must not clear")
        m.dispatch_per_row_reset([True])
        self.assertEqual(len(store), n0, "per-row reset cascade must not clear")
        if m.symbolSpace.discourse is not None:
            m.symbolSpace.discourse.reset()
            self.assertEqual(len(store), n0, "discourse.reset must not clear")

    def test_state_dict_keys_present(self):
        m = _make_model(_ON_CONFIG)
        sd = m.state_dict()
        for suffix in ("slots", "rel_type", "timestamp", "trust", "count",
                       "_next_ts"):
            self.assertIn(
                f"symbolSpace.subspace.ltm_store.{suffix}", sd,
                f"ltm_store.{suffix} must ride the state_dict")

    def test_save_load_roundtrip(self):
        from Layers import TernaryTruthStore as T
        m = _make_model(_ON_CONFIG)
        store = m.symbolSpace.ltm_store
        store.reset()
        D = store.nDim
        A = torch.zeros(D); A[0] = 1.0
        store.append_idea(A, trust=0.3)
        store.append_relation(A, A, A, rel_type=T.REL_IMPLIES, trust=-0.4)
        sd = m.state_dict()
        # Build a fresh model with the same config and load.
        m2 = _make_model(_ON_CONFIG)
        m2.load_state_dict(sd, strict=False)
        s2 = m2.symbolSpace.ltm_store
        self.assertEqual(len(s2), 2)
        self.assertEqual(s2.row(0)["rel_type"], T.REL_NONE)
        self.assertAlmostEqual(s2.row(0)["trust"], 0.3, places=5)
        self.assertEqual(s2.row(1)["rel_type"], T.REL_IMPLIES)
        self.assertAlmostEqual(s2.row(1)["trust"], -0.4, places=5)


# -- Change 1 + Change 2 (FU3): discourse + consolidation, store-backed AR --
#
# These use the SERIAL fixture (sentencePrediction on, ltmConsolidation on) so
# a discourse IS built and the per-word serial forward fires the observe-site
# store-append.

class TestDiscourseConsolidationWiring(unittest.TestCase):
    def test_discourse_is_built(self):
        # sentencePrediction builds symbolSpace.discourse.
        m = _make_model(_SERIAL_CONFIG)
        self.assertIsNotNone(m.symbolSpace.discourse,
                             "sentencePrediction must build the discourse")

    def test_discourse_wired_to_store_when_consolidated(self):
        # FU3 (Change 2): the discourse AR predictor is wired to read the
        # unified store (not its deque) when consolidated.
        m = _make_model(_SERIAL_CONFIG)
        disc = m.symbolSpace.discourse
        self.assertIs(disc._ltm_store, m.symbolSpace.ltm_store)
        self.assertTrue(disc._ltm_consolidation)

    def test_off_path_discourse_uses_deque(self):
        # A discourse WITHOUT consolidation keeps _ltm_store None (legacy
        # deque path, byte-identical). MM_grammar has a serial grammar but no
        # consolidation/sentencePrediction here -> just assert the wiring
        # default on a non-consolidated build (the OFF fixture has no
        # discourse, so check the attribute default on a fresh layer).
        from Layers import InterSentenceLayer
        layer = InterSentenceLayer(n_symbols=4, max_depth=8, n_dim=8,
                                   p=2, q=1, concept_dim=8)
        self.assertIsNone(layer._ltm_store)
        self.assertFalse(layer._ltm_consolidation)


class TestStoreBackedChain(unittest.TestCase):
    def test_get_stm_chain_reads_recent_oldest_first(self):
        # FU3: get_stm_chain reconstructs rows from store.recent() in
        # OLDEST-FIRST time order (recent() is descending timestamp).
        from Layers import TernaryTruthStore as T
        m = _make_model(_SERIAL_CONFIG)
        store = m.symbolSpace.ltm_store
        disc = m.symbolSpace.discourse
        store.reset()
        D = store.nDim
        a = torch.zeros(D); a[0] = 1.0           # oldest (ts 0)
        b = torch.zeros(D); b[1] = 1.0           # newest (ts 1)
        store.append_idea(a, trust=0.5)
        store.append_idea(b, trust=0.6)
        chain = disc.get_stm_chain()
        self.assertEqual(len(chain), 2)
        # oldest-first: chain[0] is 'a', chain[-1] is 'b'.
        d0, p0, t0 = chain[0]
        d1, p1, t1 = chain[-1]
        self.assertEqual(d0, 1)
        self.assertAlmostEqual(float(p0[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(p1[0, 1]), 1.0, places=5)
        self.assertAlmostEqual(t0, 0.5, places=5)
        self.assertAlmostEqual(t1, 0.6, places=5)

    def test_get_stm_chain_relation_slot_reconstruction(self):
        # A relation row -> depth 3, payload = [np1, vp, np2]
        # (INFIX = [idea1, predicate, idea2], the store's native order;
        # Alec 2026-06-18 -- idea1 may be present without a predicate).
        from Layers import TernaryTruthStore as T
        m = _make_model(_SERIAL_CONFIG)
        store = m.symbolSpace.ltm_store
        disc = m.symbolSpace.discourse
        store.reset()
        D = store.nDim
        np1 = torch.zeros(D); np1[0] = 1.0       # idea1
        vp = torch.zeros(D); vp[1] = 1.0         # predicate
        np2 = torch.zeros(D); np2[2] = 1.0       # idea2
        store.append_relation(np1, vp, np2, rel_type=T.REL_IMPLIES, trust=0.7)
        chain = disc.get_stm_chain()
        self.assertEqual(len(chain), 1)
        d, p, t = chain[0]
        self.assertEqual(d, 3)
        self.assertEqual(tuple(p.shape), (3, D))
        # INFIX: slot 0 = np1 (idea1), slot 1 = vp (predicate), slot 2 = np2.
        self.assertAlmostEqual(float(p[0, 0]), 1.0, places=5)   # np1 (idea1)
        self.assertAlmostEqual(float(p[1, 1]), 1.0, places=5)   # vp (predicate)
        self.assertAlmostEqual(float(p[2, 2]), 1.0, places=5)   # np2 (idea2)
        self.assertAlmostEqual(t, 0.7, places=5)

    def test_get_stm_chain_respects_n_window(self):
        m = _make_model(_SERIAL_CONFIG)
        store = m.symbolSpace.ltm_store
        disc = m.symbolSpace.discourse
        store.reset()
        D = store.nDim
        for i in range(5):
            v = torch.zeros(D); v[i % D] = 1.0
            store.append_idea(v, trust=0.1 * i)
        chain = disc.get_stm_chain(n=2)
        self.assertEqual(len(chain), 2)
        # The two MOST RECENT, oldest-first: rows 3 then 4.
        self.assertAlmostEqual(chain[-1][2], 0.4, places=5)
        self.assertAlmostEqual(chain[0][2], 0.3, places=5)


class TestObserveSkipsDequeWhenConsolidated(unittest.TestCase):
    def test_observe_does_not_append_deque_but_runs_cycle(self):
        # FU3: when consolidated, observe_stm_end_state does NOT append to the
        # per-row deque (the store-append is the source), but STILL runs the
        # predict/observe L_inter cycle.
        m = _make_model_provisioned(_SERIAL_CONFIG)
        store = m.symbolSpace.ltm_store
        disc = m.symbolSpace.discourse
        self.assertGreater(len(store), 0)
        deque_before = [len(dq) for dq in disc._stm_end_states]
        D = store.nDim
        # Stage a prediction from the (provisioned) store-backed chain, then
        # observe a fresh end-state.
        disc.predict_next_end_state(0)
        self.assertIsNotNone(disc._inter_last_pred_root[0])
        payload = torch.ones(1, D)
        disc.observe_stm_end_state([1], [payload], None)
        # Deque UNCHANGED (still 0 in the consolidated path).
        deque_after = [len(dq) for dq in disc._stm_end_states]
        self.assertEqual(deque_after, deque_before)
        self.assertEqual(deque_after[0], 0)
        # The L_inter cycle ran: a loss term accumulated.
        self.assertIsNotNone(disc._inter_loss_accum)
        self.assertGreater(disc._inter_loss_count, 0)


# -- user TruthSet -> LTM: the TruthLayer as a view over the store ----------
#
# ``store_truths`` (the serve-layer request-body TruthSet) now lands its rows
# in the unified ltm_store (ORIGIN_USER) alongside the XML-provisioned rows
# (ORIGIN_PROVISIONED) and the observe-site conversation rows
# (ORIGIN_CONVERSATION); the TruthLayer is a compatibility view materialized
# from the provisioned + user rows (``sync_from_ltm``), so the flat-field
# readers (luminosity / falsity penalty / consistency / clarifications /
# assess) read the LTM-backed user truth data.

class TestTruthLayerLtmView(unittest.TestCase):
    def test_truth_layer_attached_when_consolidated(self):
        m = _make_model(_ON_CONFIG)
        tl = m.symbolSpace.truth_layer
        self.assertIs(tl.ltm_backed, m.symbolSpace.ltm_store)

    def test_truth_layer_not_attached_when_gate_off(self):
        m = _make_model(_OFF_CONFIG)
        self.assertIsNone(m.symbolSpace.truth_layer.ltm_backed)

    def test_attach_does_not_duplicate_store_in_state_dict(self):
        m = _make_model(_ON_CONFIG)
        keys = [k for k in m.state_dict() if 'ltm_store' in k]
        self.assertTrue(keys)
        self.assertFalse(
            any('truth_layer' in k for k in keys),
            "attach_ltm must not re-register the store under truth_layer")

    def test_sync_materializes_user_and_provisioned_rows_only(self):
        from Layers import TernaryTruthStore as T
        m = _make_model(_ON_CONFIG)
        store = m.symbolSpace.ltm_store
        tl = m.symbolSpace.truth_layer
        store.reset()
        D = store.nDim
        # the view slices the content band, clamped to the store width
        cw = max(1, min(int(store.content_width), D))
        a = torch.zeros(D); a[0] = 1.0
        b = torch.zeros(D); b[1] = 1.0
        store.append_idea(a, trust=0.5)                     # conversation
        i_prov = store.append_idea(a, trust=0.9)
        store.set_origin(i_prov, T.ORIGIN_PROVISIONED, text="prov")
        i_user = store.append_idea(b, trust=-0.5)
        store.set_origin(i_user, T.ORIGIN_USER, text="user")
        n = tl.sync_from_ltm()
        self.assertEqual(n, 2)
        self.assertEqual(int(tl.count.item()), 2)
        self.assertEqual(tl._sources, ["prov", "user"])
        # flat view = content band * trust (an absolute row reduces to its
        # NP1 band; negative trust flips the sign, no bivector re-bake),
        # zero-padded up to the TruthLayer width when the band is narrower
        exp0 = torch.zeros(tl.nDim); exp0[:cw] = a[:cw] * 0.9
        exp1 = torch.zeros(tl.nDim); exp1[:cw] = b[:cw] * -0.5
        self.assertTrue(torch.allclose(tl.truths[0], exp0, atol=1e-6))
        self.assertTrue(torch.allclose(tl.truths[1], exp1, atol=1e-6))
        self.assertAlmostEqual(tl._trusts[0], 0.9, places=5)
        self.assertAlmostEqual(tl._trusts[1], -0.5, places=5)
        self.assertFalse(tl.is_empty())

    def test_sync_relation_row_flattens_live_slots(self):
        from Layers import TernaryTruthStore as T
        m = _make_model(_ON_CONFIG)
        store = m.symbolSpace.ltm_store
        tl = m.symbolSpace.truth_layer
        store.reset()
        D = store.nDim
        cw = max(1, min(int(store.content_width), D))
        np1 = torch.zeros(D); np1[0] = 1.0
        vp = torch.zeros(D); vp[1] = 1.0
        np2 = torch.zeros(D); np2[2] = 1.0
        i = store.append_relation(np1, vp, np2, rel_type=T.REL_IMPLIES,
                                  trust=0.6)
        store.set_origin(i, T.ORIGIN_USER, text="rel")
        tl.sync_from_ltm()
        exp = torch.zeros(tl.nDim)
        exp[:cw] = (np1[:cw] + vp[:cw] + np2[:cw]) / 3.0 * 0.6
        self.assertTrue(torch.allclose(tl.truths[0], exp, atol=1e-6))

    def test_sync_empty_selection_empties_view(self):
        m = _make_model(_ON_CONFIG)
        store = m.symbolSpace.ltm_store
        tl = m.symbolSpace.truth_layer
        store.reset()
        store.append_idea(torch.ones(store.nDim), trust=0.5)  # conversation
        self.assertEqual(tl.sync_from_ltm(), 0)
        self.assertEqual(int(tl.count.item()), 0)
        self.assertTrue(tl.is_empty())

    def test_sync_without_attachment_is_noop(self):
        from Layers import TruthLayer
        tl = TruthLayer(4, max_truths=8)
        self.assertEqual(tl.sync_from_ltm(), -1)


class TestRuntimeUserIngestion(unittest.TestCase):
    """``store_truths`` end-to-end on the SERIAL fixture: the request-body
    TruthSet lands in the unified LTM via the real forward."""

    def _fresh(self):
        m = _make_model(_SERIAL_CONFIG)
        return m, m.symbolSpace.ltm_store, m.symbolSpace.truth_layer

    def test_user_rows_land_alongside_provisioned(self):
        from Layers import TernaryTruthStore as T
        m, store, tl = self._fresh()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m.store_truths([{"content": "hello world", "trust": 0.9},
                            {"content": "loving there", "trust": -0.5}])
        # lazy provisioning fired first (3 XML truths, earliest ticks) ...
        self.assertTrue(m._ltm_provisioned)
        self.assertEqual(
            store.rows_of_origin(T.ORIGIN_PROVISIONED).tolist(), [0, 1, 2])
        # ... then one real end-state row per user truth, tagged + trusted.
        user_rows = store.rows_of_origin(T.ORIGIN_USER).tolist()
        self.assertEqual(len(user_rows), 2)
        r0 = store.row(user_rows[0])
        r1 = store.row(user_rows[1])
        self.assertAlmostEqual(r0["trust"], 0.9, places=5)
        self.assertAlmostEqual(r1["trust"], -0.5, places=5)
        self.assertEqual(r0["text"], "hello world")
        self.assertEqual(r1["text"], "loving there")
        self.assertGreater(float(r0["np1"].norm()), 0.0,
                           "real parse, not a placeholder")

    def test_view_and_serve_surface_read_ltm_backed_data(self):
        m, store, tl = self._fresh()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m.store_truths([{"content": "hello world", "trust": 0.9},
                            {"content": "loving there", "trust": -0.5}])
        # The view covers provisioned + user rows; sources carry the texts
        # clarifications reference.
        self.assertEqual(int(tl.count.item()), 5)
        self.assertIn("hello world", tl._sources)
        self.assertIn("loving there", tl._sources)
        self.assertIn(0.9, [round(t, 5) for t in tl._trusts if t is not None])
        # Serve surface: score / clarifications / assessment are cached on
        # the model exactly as the legacy path did.
        self.assertIsInstance(m._last_truth_score, float)
        self.assertIsInstance(m._last_clarifications, list)
        assessment = m._last_truth_assessment
        self.assertIsInstance(assessment, dict)
        for k in ("support", "conflict", "ignorance"):
            self.assertIn(k, assessment)
            self.assertGreaterEqual(assessment[k], 0.0)
            self.assertLessEqual(assessment[k], 1.0)
        # The view is non-degenerate: luminosity reads the LTM-backed rows.
        self.assertNotEqual(tl.luminosity(), 0.0)

    def test_resubmit_replaces_user_rows_only(self):
        from Layers import TernaryTruthStore as T
        m, store, tl = self._fresh()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m.store_truths([{"content": "hello world", "trust": 0.9},
                            {"content": "loving there", "trust": -0.5}])
            n_prov = len(store.rows_of_origin(T.ORIGIN_PROVISIONED))
            m.store_truths([{"content": "hello there", "trust": 0.7}])
        user_rows = store.rows_of_origin(T.ORIGIN_USER).tolist()
        self.assertEqual(len(user_rows), 1, "resubmit replaces, not appends")
        self.assertEqual(store.row(user_rows[0])["text"], "hello there")
        self.assertEqual(
            len(store.rows_of_origin(T.ORIGIN_PROVISIONED)), n_prov,
            "provisioned rows persist across user resubmits")
        self.assertEqual(int(tl.count.item()), n_prov + 1)
        self.assertIn("hello there", tl._sources)
        self.assertNotIn("hello world", tl._sources)

    def test_store_truths_after_load_does_not_reprovision(self):
        # A checkpoint provisioned before saving comes back with the
        # transient _ltm_provisioned flag reset while its provisioned rows
        # survive (the load-time revive keeps them). store_truths must NOT
        # re-run provisioning then -- that would duplicate the XML truthSet.
        # Simulate that post-load state by resetting the flag on an already-
        # provisioned model (a raw state_dict round-trip can't be used here:
        # real provisioning grows the dynamic vocab, so the production reload
        # goes through load_weights' vocab restore, not bare load_state_dict).
        from Layers import TernaryTruthStore as T
        m = _make_model_provisioned(_SERIAL_CONFIG)
        store = m.symbolSpace.ltm_store
        n_prov = len(store.rows_of_origin(T.ORIGIN_PROVISIONED))
        self.assertEqual(n_prov, 3)
        m._ltm_provisioned = False               # as after a fresh reload
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m.store_truths([{"content": "hello world", "trust": 0.9}])
        # provisioned rows unchanged (no duplication), one user row added.
        self.assertEqual(
            len(store.rows_of_origin(T.ORIGIN_PROVISIONED)), n_prov)
        self.assertEqual(len(store.rows_of_origin(T.ORIGIN_USER)), 1)
        self.assertTrue(m._ltm_provisioned)

    def test_user_rows_coexist_with_conversation_rows(self):
        from Layers import TernaryTruthStore as T
        m, store, tl = self._fresh()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m.store_truths([{"content": "hello world", "trust": 0.9}])
            opt = m.getOptimizer(lr=0.01)
            m.runEpoch(optimizer=opt, batchSize=1, split="train",
                       max_batches=2)
        conv = store.rows_of_origin(T.ORIGIN_CONVERSATION).tolist()
        user = store.rows_of_origin(T.ORIGIN_USER).tolist()
        prov = store.rows_of_origin(T.ORIGIN_PROVISIONED).tolist()
        self.assertTrue(conv, "the epoch's observe push appends STM rows")
        self.assertEqual(len(user), 1)
        self.assertEqual(len(prov), 3)
        # One persistent store holds all three writers' rows.
        self.assertEqual(len(store), len(conv) + len(user) + len(prov))
        # The view keeps reading only the user TruthSet surface.
        tl.sync_from_ltm()
        self.assertEqual(int(tl.count.item()), len(user) + len(prov))


# -- load-time LTM revive, gated by <stateless> ----------------------------
#
# The server is stateless by default (<stateless> mirrors WikiOracle's
# server.stateless; the shipped deployment runs stateless), so a checkpoint's
# request-scoped ORIGIN_USER rows must NOT be revived on load -- each request
# supplies its own TruthSet. Every load_state_dict rematerializes the
# TruthLayer view; stateless drops the user rows first, stateful keeps them.

class TestStatelessRevive(unittest.TestCase):
    def _seed(self, m):
        """Seed one row per origin (provisioned / user / conversation)."""
        from Layers import TernaryTruthStore as T
        store = m.symbolSpace.ltm_store
        store.reset()
        D = store.nDim
        v = torch.zeros(D); v[0] = 1.0
        i0 = store.append_idea(v, trust=0.9)
        store.set_origin(i0, T.ORIGIN_PROVISIONED, text="cat")
        i1 = store.append_idea(v, trust=0.8)
        store.set_origin(i1, T.ORIGIN_USER, text="hello world")
        store.append_idea(v, trust=0.3)                    # ORIGIN_CONVERSATION
        return store

    def test_default_flag_is_stateless_true(self):
        # Config read: absent <stateless> -> default True (model + subspace).
        m = _make_model(_ON_CONFIG)
        self.assertTrue(m.stateless)
        self.assertTrue(m.symbolSpace._stateless)

    def test_stateful_fixture_reads_false(self):
        # Config read: <stateless>false</stateless> is honored end-to-end.
        m = _make_model(_STATEFUL_CONFIG)
        self.assertFalse(m.stateless)
        self.assertFalse(m.symbolSpace._stateless)

    def test_stateless_load_drops_user_rows(self):
        from Layers import TernaryTruthStore as T
        m = _make_model(_ON_CONFIG)
        self._seed(m)
        sd = m.state_dict()
        m2 = _make_model(_ON_CONFIG)              # stateless (default)
        m2.load_state_dict(sd, strict=False)
        store = m2.symbolSpace.ltm_store
        # user rows dropped; provisioned + conversation survive.
        self.assertEqual(store.rows_of_origin(T.ORIGIN_USER).tolist(), [])
        self.assertEqual(
            store.rows_of_origin(T.ORIGIN_PROVISIONED).tolist(), [0])
        self.assertEqual(
            store.rows_of_origin(T.ORIGIN_CONVERSATION).tolist(), [1])
        self.assertEqual(len(store), 2)
        # the view was rematerialized from the surviving provisioned row only.
        self.assertEqual(int(m2.symbolSpace.truth_layer.count.item()), 1)

    def test_stateful_load_keeps_user_rows(self):
        from Layers import TernaryTruthStore as T
        m = _make_model(_STATEFUL_CONFIG)
        self._seed(m)
        sd = m.state_dict()
        m2 = _make_model(_STATEFUL_CONFIG)        # stateful
        m2.load_state_dict(sd, strict=False)
        store = m2.symbolSpace.ltm_store
        # every row survives; the user row is durable state.
        self.assertEqual(store.rows_of_origin(T.ORIGIN_USER).tolist(), [1])
        self.assertEqual(len(store), 3)
        # the view covers provisioned + user (conversation stays out).
        self.assertEqual(int(m2.symbolSpace.truth_layer.count.item()), 2)

    def test_load_syncs_view_even_with_no_user_rows(self):
        # A checkpoint with only provisioned + conversation rows still
        # rematerializes the view on load (stateless drop is a no-op).
        from Layers import TernaryTruthStore as T
        m = _make_model(_ON_CONFIG)
        store = m.symbolSpace.ltm_store
        store.reset()
        D = store.nDim
        v = torch.zeros(D); v[0] = 1.0
        i0 = store.append_idea(v, trust=0.7)
        store.set_origin(i0, T.ORIGIN_PROVISIONED, text="p")
        store.append_idea(v, trust=0.2)                    # conversation
        sd = m.state_dict()
        m2 = _make_model(_ON_CONFIG)
        m2.load_state_dict(sd, strict=False)
        self.assertEqual(len(m2.symbolSpace.ltm_store), 2)
        self.assertEqual(int(m2.symbolSpace.truth_layer.count.item()), 1)


class TestSerialForwardAppendsStore(unittest.TestCase):
    def test_forward_grows_store_not_deque(self):
        # Change 1 + FU3: a serial discourse+consolidation forward grows the
        # store and leaves the deque empty, running the predict->observe cycle
        # without error.
        m = _make_model(_SERIAL_CONFIG)
        store = m.symbolSpace.ltm_store
        disc = m.symbolSpace.discourse
        opt = m.getOptimizer(lr=0.01)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m.runEpoch(optimizer=opt, batchSize=1, split="train",
                       max_batches=3)
        # Provisioning (3) + per-sentence conversation appends.
        self.assertGreater(len(store), 3)
        # Deque stays empty (the store-append is the single source).
        self.assertEqual(len(disc._stm_end_states[0]), 0)
        # The store-backed chain reflects the store.
        self.assertEqual(len(disc.get_stm_chain()), len(store))


if __name__ == "__main__":
    unittest.main()
