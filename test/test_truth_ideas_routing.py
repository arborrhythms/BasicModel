"""Truth / Ideas processing -- store routing + trust (Workstreams F+G;
doc/specs/mereological-order-raising.md "Truth / Ideas processing"; Alec
2026-06-18).

Stage 2: a learned relative relation is ROUTED by reducibility behind the
``truthIdeas`` gate ("the intuitive and explicit knowings"):

  * REDUCIBLE -- both ENTITY operands snap to existing WS codebook rows (no
    mint) -> stays the WS-META "intuitive knowing" (the legacy reduce path),
    carrying the full tetralemma trust;
  * INEFFABLE -- a composed idea -> the uncollapsed (idea1, predicate, idea2)
    triple lands in the sibling ``RelativeTruthStore`` ("explicit knowing")
    with a SCALAR trust collapsed from the tetralemma (t - f; "inconsistency
    is not a valid object of knowing").

Flag-off is the legacy always-reduce path (byte-identical). The routing is
exercised directly on ``_maybe_learn_relation`` (the factor methods mocked so
the gate is guaranteed to accept), mirroring
``test_relative_sentence_codebook_insertion``.
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
_CONFIG = os.path.join(_DATA_DIR, "MM_xor_fixture.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")


def _make_radix_model(config=_CONFIG):
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


def _accept_all(cs):
    """Force the gate open by mocking the three factors to 1.0 each (so the
    relation is also REDUCIBLE -- children factor 1.0)."""
    cs._learn_score_children_in_codebook = lambda i1, i2: 1.0
    cs._learn_score_is_truth_obvious = lambda rel: 1.0
    cs._learn_score_resolves_contradiction = lambda rel: 1.0


def _accept_but_ineffable(cs):
    """Open the gate (tc=0) but make the relation IRREDUCIBLE: the children
    factor is 0.0 so neither operand 'snaps' to a known code."""
    cs._learn_score_children_in_codebook = lambda i1, i2: 0.0
    cs._learn_score_is_truth_obvious = lambda rel: 1.0
    cs._learn_score_resolves_contradiction = lambda rel: 1.0
    cs.truth_criterion = 0.0


def _three_ideas(D):
    predicate = torch.zeros(D)
    predicate[0] = 1.0
    idea1 = torch.zeros(D)
    idea1[1] = 1.0
    idea2 = torch.zeros(D)
    idea2[2] = 1.0
    return predicate, idea1, idea2


# -- pure helpers ----------------------------------------------------------

class TestTrustCollapse(unittest.TestCase):
    """``_collapse_trust`` = t - f, clamped to [-1, 1] (BOTH/NEITHER drop)."""

    def test_collapse_is_t_minus_f(self):
        from Spaces import ConceptualSpace
        c = ConceptualSpace._collapse_trust
        self.assertAlmostEqual(c((0.8, 0.1, 0.1, 0.0)), 0.7, places=6)
        self.assertAlmostEqual(c((0.1, 0.6, 0.2, 0.1)), -0.5, places=6)
        # BOTH / NEITHER never contribute.
        self.assertAlmostEqual(c((0.0, 0.0, 1.0, 0.0)), 0.0, places=6)
        self.assertAlmostEqual(c((0.0, 0.0, 0.0, 1.0)), 0.0, places=6)

    def test_collapse_clamps(self):
        from Spaces import ConceptualSpace
        c = ConceptualSpace._collapse_trust
        self.assertEqual(c((2.0, 0.0, 0.0, 0.0)), 1.0)
        self.assertEqual(c((0.0, 2.0, 0.0, 0.0)), -1.0)


class TestReducibility(unittest.TestCase):
    """``_relation_is_reducible`` is True iff BOTH operands snap (children
    factor == 1.0)."""

    def test_reducible_only_when_both_snap(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        D = int(cs.nDim)
        _, idea1, idea2 = _three_ideas(D)
        cs._learn_score_children_in_codebook = lambda i1, i2: 1.0
        self.assertTrue(cs._relation_is_reducible(idea1, idea2))
        cs._learn_score_children_in_codebook = lambda i1, i2: 0.5
        self.assertFalse(cs._relation_is_reducible(idea1, idea2))
        cs._learn_score_children_in_codebook = lambda i1, i2: 0.0
        self.assertFalse(cs._relation_is_reducible(idea1, idea2))


# -- routing (the new behaviour) -------------------------------------------

class TestRoutingFlagOff(unittest.TestCase):
    """Flag OFF -> the legacy always-reduce path; the relative store is
    never touched even for an ineffable relation (byte-identical)."""

    def test_flag_off_never_uses_relative_store(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        ws = cs.terminalSymbolSpace_ref
        store = m.symbolSpace.relative_store
        self.assertFalse(getattr(cs, "_truth_ideas", False),
                         "fixture must default truthIdeas OFF")
        cs.truth_criterion = 0.3
        # Even an 'ineffable' (children=0) relation reduces when the flag is
        # off -- mint a fresh row, never the relative store.
        cs._learn_score_children_in_codebook = lambda i1, i2: 0.0
        cs._learn_score_is_truth_obvious = lambda rel: 1.0
        cs._learn_score_resolves_contradiction = lambda rel: 1.0
        cs.truth_criterion = 0.0
        D = int(cs.nDim)
        predicate, idea1, idea2 = _three_ideas(D)
        n_before = len(store)
        out = cs._maybe_learn_relation(predicate, idea1, idea2)
        self.assertIsInstance(out, int, "flag-off reduce path returns an int")
        self.assertGreater(out, 0)
        self.assertTrue(ws.is_meta(out))
        self.assertEqual(len(store), n_before,
                         "flag-off must NOT write the relative store")


class TestRoutingReducible(unittest.TestCase):
    """Flag ON + reducible -> WS META ("intuitive knowing"); relative store
    untouched; full tetralemma stored on the META node."""

    def test_reducible_routes_to_ss_meta(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        ws = cs.terminalSymbolSpace_ref
        store = m.symbolSpace.relative_store
        cs._truth_ideas = True
        cs.truth_criterion = 0.3
        _accept_all(cs)               # children == 1.0 -> reducible
        D = int(cs.nDim)
        predicate, idea1, idea2 = _three_ideas(D)
        n_before = len(store)
        pred_pos = cs._maybe_learn_relation(predicate, idea1, idea2)
        self.assertIsInstance(pred_pos, int,
                              "reducible relation returns the META position")
        self.assertGreater(pred_pos, 0)
        children = ws.taxonomy_children(pred_pos)
        self.assertEqual(len(children), 2)
        self.assertTrue(ws.is_meta(pred_pos))
        trust = ws.meta_trust.get(pred_pos)
        self.assertIsNotNone(trust, "WS META carries the FULL tetralemma")
        self.assertEqual(len(trust), 4)
        self.assertAlmostEqual(sum(trust), 1.0, places=5)
        self.assertEqual(len(store), n_before,
                         "reducible relation must NOT touch the relative store")


class TestRoutingIneffable(unittest.TestCase):
    """Flag ON + ineffable -> RelativeTruthStore ("explicit knowing"); the
    uncollapsed (idea1, predicate, idea2) triple stored with a scalar trust;
    WS taxonomy untouched."""

    def test_ineffable_routes_to_relative_store(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        ws = cs.terminalSymbolSpace_ref
        store = m.symbolSpace.relative_store
        cs._truth_ideas = True
        _accept_but_ineffable(cs)
        # Pin the tetralemma so the collapsed degree is a known non-zero.
        cs._tetralemma_trust = lambda rel, truth_set=None: (0.8, 0.1, 0.1, 0.0)
        D = int(cs.nDim)
        predicate, idea1, idea2 = _three_ideas(D)
        n_before = len(store)
        tax_before = dict(ws.taxonomy)
        trust_before = dict(ws.meta_trust)

        out = cs._maybe_learn_relation(predicate, idea1, idea2)

        # Return shape distinguishes the explicit-knowing home.
        self.assertIsInstance(out, tuple)
        self.assertEqual(out[0], "idea")
        row = out[1]
        self.assertEqual(len(store), n_before + 1,
                         "ineffable relation appends one triple")
        # The uncollapsed triple is (idea1, predicate, idea2) * degree, with
        # degree = t - f = 0.7. Operands are conformed to the store width
        # (leading content slice; zero-padded when the store is wider).
        sd = int(store.nDim)
        k = idea1.numel()
        np1, vp, np2 = store.triple(row)
        self.assertEqual(np1.numel(), sd)
        self.assertTrue(torch.allclose(np1[:k], idea1[:k] * 0.7, atol=1e-5))
        self.assertTrue(torch.allclose(vp[:k], predicate[:k] * 0.7, atol=1e-5))
        self.assertTrue(torch.allclose(np2[:k], idea2[:k] * 0.7, atol=1e-5))
        if sd > k:
            self.assertTrue(torch.allclose(
                np1[k:], torch.zeros(sd - k), atol=1e-6),
                "where/when tail is zero-padded")
        # WS taxonomy is NOT mutated (testimony never grows the codebook).
        self.assertEqual(ws.taxonomy, tax_before,
                         "ineffable relation must not touch the WS taxonomy")
        self.assertEqual(ws.meta_trust, trust_before)

    def test_ineffable_degrades_to_reduce_without_relative_store(self):
        """No relative store reachable -> graceful degrade to the reduce
        path so the relation is never silently dropped."""
        m = _make_radix_model()
        cs = m.conceptualSpace
        ws = cs.terminalSymbolSpace_ref
        cs._truth_ideas = True
        _accept_but_ineffable(cs)
        cs._relative_store_for_learning = lambda: None
        D = int(cs.nDim)
        predicate, idea1, idea2 = _three_ideas(D)
        out = cs._maybe_learn_relation(predicate, idea1, idea2)
        self.assertIsInstance(out, int,
                              "degrade path returns the reduced META position")
        self.assertTrue(ws.is_meta(out))


# -- config -> stamp wiring ------------------------------------------------

class TestFlagStamp(unittest.TestCase):
    """The ``truthIdeas`` config element flows XSD -> parse -> CS stamp."""

    def test_default_off(self):
        m = _make_radix_model()
        self.assertFalse(m.truth_ideas)
        self.assertFalse(getattr(m.conceptualSpace, "_truth_ideas", False))

    def test_config_on_stamps_cs(self):
        # Inject <truthIdeas>true</truthIdeas> into a copy of the fixture
        # (after <symbolicOrder> to honour the XSD sequence) in the data dir
        # so its relative data references still resolve.
        with open(_CONFIG, "r", encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("<symbolicOrder>0</symbolicOrder>", src)
        on = src.replace(
            "<symbolicOrder>0</symbolicOrder>",
            "<symbolicOrder>0</symbolicOrder>\n    "
            "<truthIdeas>true</truthIdeas>")
        tmp = os.path.join(_DATA_DIR, "_tmp_truthideas_on.xml")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(on)
        try:
            m = _make_radix_model(config=tmp)
            self.assertTrue(m.truth_ideas, "config truthIdeas=true -> parsed")
            self.assertTrue(getattr(m.conceptualSpace, "_truth_ideas", False),
                            "config truthIdeas=true -> stamped on CS")
        finally:
            os.remove(tmp)


# -- stage 3: STM -> LTM trust persistence ---------------------------------

class TestStmLtmTrust(unittest.TestCase):
    """``stm_end_state_trust`` collapses each relative end-state's relation
    tetralemma to the stored scalar; the LTM slot persists it."""

    def test_relative_rows_get_scalar_absolute_none(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        cs._truth_ideas = True
        cs._tetralemma_trust = lambda rel, truth_set=None: (0.8, 0.1, 0.1, 0.0)
        D = int(cs.nDim)
        buf = torch.zeros(2, 3, D)
        buf[0, 2, 0] = 1.0          # row 0 predicate (slot depth-1)
        cs.stm._buffer = buf
        cs.stm._depth = torch.tensor([3, 1], dtype=torch.long)
        out = cs.stm_end_state_trust(buf, torch.tensor([True, False]))
        self.assertIsNotNone(out)
        self.assertAlmostEqual(out[0], 0.7, places=6)   # relative -> t - f
        self.assertIsNone(out[1], "absolute row carries no relation trust")
        # The CS stash mirrors the returned trusts (read at observe).
        self.assertEqual(cs._last_end_state_trust, out)

    def test_flag_off_returns_none(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        self.assertFalse(getattr(cs, "_truth_ideas", False))
        D = int(cs.nDim)
        buf = torch.zeros(1, 3, D)
        cs.stm._buffer = buf
        cs.stm._depth = torch.tensor([3], dtype=torch.long)
        self.assertIsNone(
            cs.stm_end_state_trust(buf, torch.tensor([True])),
            "flag off -> no trust -> LTM slot stays None (byte-identical)")

    def test_ltm_slot_persists_scalar_trust(self):
        from Layers import InterSentenceLayer
        disc = InterSentenceLayer(4, 3, 4, concept_dim=4, batch=1)
        payload = torch.randn(3, 4)
        disc.observe_stm_end_state([3], [payload], tetralemmas=[0.7])
        chain = disc.get_stm_chain(b=0)
        self.assertEqual(len(chain), 1)
        depth, stored_payload, tet = chain[0]
        self.assertEqual(depth, 3)
        self.assertEqual(tet, 0.7, "the scalar trust persists in the LTM slot")

    def test_ltm_slot_none_when_no_trust(self):
        from Layers import InterSentenceLayer
        disc = InterSentenceLayer(4, 3, 4, concept_dim=4, batch=1)
        disc.observe_stm_end_state([1], [torch.randn(1, 4)], tetralemmas=None)
        _, _, tet = disc.get_stm_chain(b=0)[0]
        self.assertIsNone(tet, "no trust -> slot stays None (byte-identical)")


# -- stage 4: reasoning engine (modus ponens) ------------------------------

_D2 = 6


def _v(*vals):
    v = torch.zeros(_D2)
    for i, x in enumerate(vals):
        v[i] = x
    return v


def _store():
    from Layers import RelativeTruthStore
    return RelativeTruthStore(_D2, max_triples=16)


_CACHED_CS = []


def _cs():
    """A ConceptualSpace instance (cached) for the pure-logic reason tests --
    they pass a standalone store and never mutate the CS."""
    if not _CACHED_CS:
        _CACHED_CS.append(_make_radix_model().conceptualSpace)
    return _CACHED_CS[0]


class TestParthoodAndIdentity(unittest.TestCase):
    def test_parthood_coverage(self):
        from Spaces import ConceptualSpace
        p = ConceptualSpace._idea_parthood
        self.assertAlmostEqual(p(_v(1, 0, 0), _v(1, 1, 0)), 1.0, places=6)
        self.assertAlmostEqual(p(_v(1, 1, 0), _v(1, 0, 0)), 0.5, places=6)
        self.assertAlmostEqual(p(_v(1, 0), _v(0, 1)), 0.0, places=6)

    def test_identity_jaccard(self):
        from Spaces import ConceptualSpace
        j = ConceptualSpace._idea_identity
        self.assertAlmostEqual(j(_v(1, 1), _v(1, 1)), 1.0, places=6)
        self.assertAlmostEqual(j(_v(1, 0, 0), _v(1, 1, 0)), 0.5, places=6)
        self.assertAlmostEqual(j(_v(1, 0), _v(0, 1)), 0.0, places=6)


class TestReason(unittest.TestCase):
    def test_single_step_modus_ponens(self):
        cs = _cs()
        st = _store()
        st.record_triple(_v(1, 1), _v(0, 1), _v(0, 0, 1), degree=0.8)  # A->B, t1
        res = cs.reason(_v(1, 0), 0.5, parthood_threshold=0.7, store=st)
        self.assertEqual(len(res['derived']), 1)
        d = res['derived'][0]
        self.assertTrue(torch.allclose(d['concept'], _v(0, 0, 1), atol=1e-5),
                        "consequent B recovered unscaled")
        self.assertAlmostEqual(d['trust'], 0.4, places=6)   # t1*t2 = 0.8*0.5
        self.assertAlmostEqual(d['parthood'], 1.0, places=6)
        self.assertEqual(d['source'], 0)
        self.assertEqual(d['step'], 0)
        self.assertAlmostEqual(res['luminosity_gain'], 0.4, places=6)

    def test_no_fire_below_parthood_threshold(self):
        cs = _cs()
        st = _store()
        st.record_triple(_v(1, 1), _v(0, 1), _v(0, 0, 1), degree=0.8)
        res = cs.reason(_v(0, 0, 1), 1.0, parthood_threshold=0.7, store=st)
        self.assertEqual(res['derived'], [])
        self.assertEqual(res['luminosity_gain'], 0.0)

    def test_zero_trust_relation_skipped(self):
        cs = _cs()
        st = _store()
        st.record_triple(_v(1, 1), _v(0, 1), _v(0, 0, 1), degree=0.0)
        res = cs.reason(_v(1, 0), 1.0, parthood_threshold=0.7, store=st)
        self.assertEqual(res['derived'], [],
                         "a ~zero-trust relation carries no knowing to fire")

    def test_negative_trust_lie_not_illuminating(self):
        cs = _cs()
        st = _store()
        st.record_triple(_v(1, 1), _v(0, 1), _v(0, 0, 1), degree=-0.6)
        res = cs.reason(_v(1, 0), 0.5, parthood_threshold=0.7, store=st)
        self.assertEqual(len(res['derived']), 1)
        self.assertAlmostEqual(res['derived'][0]['trust'], -0.3, places=6)
        self.assertEqual(res['luminosity_gain'], 0.0,
                         "a distrusted conclusion adds no illuminated area")

    def test_forward_chaining(self):
        cs = _cs()
        st = _store()
        st.record_triple(_v(1, 0, 0), _v(0, 1, 0), _v(0, 1, 0), degree=1.0)
        st.record_triple(_v(0, 1, 0), _v(0, 0, 1), _v(0, 0, 1), degree=1.0)
        one = cs.reason(_v(1, 0, 0), 1.0, max_steps=1, store=st)
        self.assertEqual(len(one['derived']), 1, "one step -> one hop")
        two = cs.reason(_v(1, 0, 0), 1.0, max_steps=2, store=st)
        self.assertEqual(len(two['derived']), 2, "two steps -> the chain A->B->C")
        self.assertEqual({d['source'] for d in two['derived']}, {0, 1})
        self.assertEqual({d['step'] for d in two['derived']}, {0, 1})

    def test_empty_store(self):
        cs = _cs()
        res = cs.reason(_v(1, 0), 1.0, store=_store())
        self.assertEqual(res, {'derived': [], 'luminosity_gain': 0.0})


# -- stage 5: verification against order-0 episodes ------------------------

class TestVerifyRelation(unittest.TestCase):
    def test_support_raises_trust_and_rebakes(self):
        cs = _cs()
        st = _store()
        st.record_triple(_v(1, 1), _v(0, 1), _v(0, 0, 1), degree=0.5)
        # episodes whose antecedent is part of A=[1,1] and consequent part of
        # B=[0,0,1] -> full support.
        eps = [(_v(1, 0), _v(0, 0, 1)), (_v(0, 1), _v(0, 0, 1))]
        new = cs.verify_relation(0, eps, store=st, support_weight=0.5)
        self.assertAlmostEqual(new, 0.75, places=6)   # 0.5*0.5 + 0.5*1
        # magnitude re-baked: np1 == A * new = [1,1]*0.75.
        np1, _vp, np2 = st.triple(0)
        self.assertTrue(torch.allclose(np1, _v(1, 1) * 0.75, atol=1e-5))
        self.assertTrue(torch.allclose(np2, _v(0, 0, 1) * 0.75, atol=1e-5))

    def test_counterevidence_lowers_trust(self):
        cs = _cs()
        st = _store()
        st.record_triple(_v(1, 1), _v(0, 1), _v(0, 0, 1), degree=0.5)
        # antecedent covered, consequent NOT (it's some other thing) -> all
        # relevant, none supporting.
        eps = [(_v(1, 0), _v(1, 0)), (_v(0, 1), _v(1, 0))]
        new = cs.verify_relation(0, eps, store=st, support_weight=0.5)
        self.assertAlmostEqual(new, -0.25, places=6)  # 0.5*0.5 + 0.5*(-1)

    def test_no_relevant_episode_leaves_trust(self):
        cs = _cs()
        st = _store()
        st.record_triple(_v(1, 1), _v(0, 1), _v(0, 0, 1), degree=0.5)
        # antecedent not covered by A -> no relevant evidence.
        eps = [(_v(0, 0, 1), _v(0, 0, 1))]
        new = cs.verify_relation(0, eps, store=st, support_weight=0.5)
        self.assertAlmostEqual(new, 0.5, places=6)

    def test_zero_trust_relation_unverifiable(self):
        cs = _cs()
        st = _store()
        st.record_triple(_v(1, 1), _v(0, 1), _v(0, 0, 1), degree=0.0)
        self.assertEqual(cs.verify_relation(0, [(_v(1, 0), _v(0, 0, 1))],
                                            store=st), 0.0)


# -- persistence: per-triple trust survives state_dict round-trip ----------

class TestRelativeTrustPersistence(unittest.TestCase):
    """The per-triple relation trust is a REGISTERED BUFFER (``trust``) so a
    checkpoint save/load recovers it. Before the fix it lived in a plain
    Python list outside the state_dict, so a reloaded relation reverted to
    the 1.0 fallback in ``reason`` / ``verify_relation``."""

    def test_trust_in_state_dict_roundtrip(self):
        st = _store()
        idx = st.record_triple(_v(1, 1), _v(0, 1), _v(0, 0, 1), degree=0.7)
        self.assertEqual(idx, 0)
        sd = st.state_dict()
        self.assertIn('trust', sd, "trust must be a serialized buffer")

        fresh = _store()
        self.assertEqual(float(fresh.trust[0]), 0.0)   # zero before load
        fresh.load_state_dict(sd)
        self.assertAlmostEqual(float(fresh.trust[0]), 0.7, places=6)
        # back-compat list view tracks the buffer over live rows.
        self.assertEqual(len(fresh), 1)
        self.assertAlmostEqual(fresh._trusts[0], 0.7, places=6)

    def test_reason_over_reloaded_store_recovers_t1(self):
        cs = _cs()
        st = _store()
        st.record_triple(_v(1, 1), _v(0, 1), _v(0, 0, 1), degree=0.7)
        sd = st.state_dict()

        reloaded = _store()
        reloaded.load_state_dict(sd)
        # query covered by the antecedent A=[1,1]; query_trust=1.0 so the
        # derived trust isolates t1 -> must be 0.7, NOT the 1.0 fallback.
        res = cs.reason(_v(1, 0), 1.0, parthood_threshold=0.7, store=reloaded)
        self.assertEqual(len(res['derived']), 1)
        self.assertAlmostEqual(res['derived'][0]['trust'], 0.7, places=6)
        # B recovered UNSCALED (np2/t1) -> proves t1=0.7 was used to unbake.
        self.assertTrue(torch.allclose(
            res['derived'][0]['concept'], _v(0, 0, 1), atol=1e-5))

    def test_old_checkpoint_without_trust_loads_nonstrict(self):
        st = _store()
        st.record_triple(_v(1, 1), _v(0, 1), _v(0, 0, 1), degree=0.7)
        sd = st.state_dict()
        del sd['trust']   # simulate a pre-fix checkpoint lacking the key
        fresh = _store()
        # non-strict load tolerates the missing key; trust stays zero-init.
        fresh.load_state_dict(sd, strict=False)
        self.assertEqual(float(fresh.trust[0]), 0.0)

    def test_verify_relation_writes_back_to_buffer(self):
        cs = _cs()
        st = _store()
        st.record_triple(_v(1, 1), _v(0, 1), _v(0, 0, 1), degree=0.5)
        eps = [(_v(1, 0), _v(0, 0, 1)), (_v(0, 1), _v(0, 0, 1))]
        new = cs.verify_relation(0, eps, store=st, support_weight=0.5)
        self.assertAlmostEqual(new, 0.75, places=6)
        # the write landed in the buffer (not a discarded list snapshot).
        self.assertAlmostEqual(float(st.trust[0]), 0.75, places=6)


if __name__ == "__main__":
    unittest.main()
