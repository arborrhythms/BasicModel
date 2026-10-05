"""External provenance, independent evidence and relational readers (item 7)."""

from __future__ import annotations

import os
import sys
import tempfile
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


# -- config -> stamp wiring ------------------------------------------------

class TestTrustStamp(unittest.TestCase):
    """The ``trust`` config element flows XSD -> parse -> CS stamp."""

    def test_default_one(self):
        m = _make_radix_model()
        self.assertAlmostEqual(m.trust, 1.0, places=6)
        self.assertAlmostEqual(
            getattr(m.conceptualSpace, "_trust", 1.0), 1.0, places=6)
        self.assertAlmostEqual(
            m._effective_incoming_trust(0.8), 0.8, places=6)

    def test_config_trust_stamps_cs_and_scales_incoming(self):
        with open(_CONFIG, "r", encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("<conceptLayers>1</conceptLayers>", src)
        on = src.replace(
            "<conceptLayers>1</conceptLayers>",
            "<conceptLayers>1</conceptLayers>\n    "
            "<trust>0.25</trust>")
        # Scratch configurations must not enter the source snapshot.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml",
                                         encoding="utf-8", delete=False) as fh:
            fh.write(on)
            tmp = fh.name
        try:
            m = _make_radix_model(config=tmp)
            self.assertAlmostEqual(m.trust, 0.25, places=6)
            self.assertAlmostEqual(
                getattr(m.conceptualSpace, "_trust", 1.0), 0.25, places=6)
            self.assertAlmostEqual(
                m._effective_incoming_trust(0.8), 0.2, places=6)
            self.assertAlmostEqual(
                m._effective_incoming_trust(-0.8), -0.2, places=6)
        finally:
            os.remove(tmp)


# -- stage 3: STM -> LTM trust persistence ---------------------------------

class TestStmLtmTrust(unittest.TestCase):
    """``stm_end_state_trust`` stores event trust for absolute rows and
    relation trust for relative rows; the LTM slot persists it."""

    def test_relative_rows_get_scalar_absolute_gets_event_trust(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        cs._tetralemma_trust = lambda rel, truth_set=None: (0.8, 0.1, 0.1, 0.0)
        D = int(cs.nDim)
        buf = torch.zeros(2, 3, D)
        buf[0, 2, 0] = 1.0          # row 0 predicate (slot depth-1)
        cs.stm._buffer = buf
        cs.stm._depth = torch.tensor([3, 1], dtype=torch.long)
        out = cs.stm_end_state_trust(buf, torch.tensor([True, False]))
        self.assertIsNotNone(out)
        self.assertAlmostEqual(out[0], 1.0, places=6)   # provenance, not predicate
        self.assertAlmostEqual(
            out[1], 1.0, places=6,
            msg="absolute row carries trust that the event description refers")
        # The CS stash mirrors the returned trusts (read at observe).
        self.assertEqual(cs._last_end_state_trust, out)

    def test_model_trust_scales_relative_and_absolute_rows(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        cs._trust = 0.5
        cs._tetralemma_trust = lambda rel, truth_set=None: (0.8, 0.1, 0.1, 0.0)
        D = int(cs.nDim)
        buf = torch.zeros(2, 3, D)
        buf[0, 2, 0] = 1.0
        cs.stm._buffer = buf
        cs.stm._depth = torch.tensor([3, 1], dtype=torch.long)
        out = cs.stm_end_state_trust(buf, torch.tensor([True, False]))
        self.assertAlmostEqual(out[0], 0.5, places=6)
        self.assertAlmostEqual(out[1], 0.5, places=6)

    def test_ltm_slot_persists_scalar_trust(self):
        from Layers import BracketExpectation
        disc = BracketExpectation(4, 3, 4, concept_dim=4, batch=1)
        payload = torch.randn(3, 4)
        disc.observe_stm_end_state([3], [payload], tetralemmas=[0.7])
        chain = disc.get_stm_chain(b=0)
        self.assertEqual(len(chain), 1)
        depth, stored_payload, tet = chain[0]
        self.assertEqual(depth, 3)
        self.assertEqual(tet, 0.7, "the scalar trust persists in the LTM slot")

    def test_ltm_slot_none_when_no_trust(self):
        from Layers import BracketExpectation
        disc = BracketExpectation(4, 3, 4, concept_dim=4, batch=1)
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
    from Layers import TernaryTruthStore
    return TernaryTruthStore(_D2, capacity=16)


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
        st.append_relation(_v(1, 1), _v(0, 1), _v(0, 0, 1), trust=0.8)  # A->B, t1
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
        st.append_relation(_v(1, 1), _v(0, 1), _v(0, 0, 1), trust=0.8)
        res = cs.reason(_v(0, 0, 1), 1.0, parthood_threshold=0.7, store=st)
        self.assertEqual(res['derived'], [])
        self.assertEqual(res['luminosity_gain'], 0.0)

    def test_zero_trust_relation_skipped(self):
        cs = _cs()
        st = _store()
        st.append_relation(_v(1, 1), _v(0, 1), _v(0, 0, 1), trust=0.0)
        res = cs.reason(_v(1, 0), 1.0, parthood_threshold=0.7, store=st)
        self.assertEqual(res['derived'], [],
                         "a ~zero-trust relation carries no knowing to fire")

    def test_negative_trust_lie_not_illuminating(self):
        cs = _cs()
        st = _store()
        st.append_relation(_v(1, 1), _v(0, 1), _v(0, 0, 1), trust=-0.6)
        res = cs.reason(_v(1, 0), 0.5, parthood_threshold=0.7, store=st)
        self.assertEqual(len(res['derived']), 1)
        self.assertAlmostEqual(res['derived'][0]['trust'], -0.3, places=6)
        self.assertEqual(res['luminosity_gain'], 0.0,
                         "a distrusted conclusion adds no illuminated area")

    def test_forward_chaining(self):
        cs = _cs()
        st = _store()
        st.append_relation(_v(1, 0, 0), _v(0, 1, 0), _v(0, 1, 0), trust=1.0)
        st.append_relation(_v(0, 1, 0), _v(0, 0, 1), _v(0, 0, 1), trust=1.0)
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
    def test_support_joins_evidence_without_rebaking(self):
        cs = _cs()
        st = _store()
        st.append_relation(_v(1, 1), _v(0, 1), _v(0, 0, 1), trust=0.5)
        # episodes whose antecedent is part of A=[1,1] and consequent part of
        # B=[0,0,1] -> full support.
        eps = [(_v(1, 0), _v(0, 0, 1)), (_v(0, 1), _v(0, 0, 1))]
        new = cs.verify_relation(0, eps, store=st, support_weight=0.5)
        self.assertAlmostEqual(new, 0.5, places=6)   # observed positive evidence
        # Evidence updates leave the stored vectors unscaled.
        np1, _vp, np2 = st.slots[0].unbind()
        self.assertTrue(torch.allclose(np1, _v(1, 1), atol=1e-5))
        self.assertTrue(torch.allclose(np2, _v(0, 0, 1), atol=1e-5))

    def test_counterevidence_joins_the_negative_pole(self):
        cs = _cs()
        st = _store()
        st.append_relation(_v(1, 1), _v(0, 1), _v(0, 0, 1), trust=0.5,
                           evidence=(.5, 0.))
        # antecedent covered, consequent NOT (it's some other thing) -> all
        # relevant, none supporting.
        eps = [(_v(1, 0), _v(1, 0)), (_v(0, 1), _v(1, 0))]
        new = cs.verify_relation(0, eps, store=st, support_weight=0.5)
        self.assertAlmostEqual(new, 0., places=6)  # equal independent poles
        self.assertAlmostEqual(float(st.trust[0]), .5, places=6)

    def test_no_relevant_episode_leaves_trust(self):
        cs = _cs()
        st = _store()
        st.append_relation(_v(1, 1), _v(0, 1), _v(0, 0, 1), trust=0.5)
        # antecedent not covered by A -> no relevant evidence.
        eps = [(_v(0, 0, 1), _v(0, 0, 1))]
        new = cs.verify_relation(0, eps, store=st, support_weight=0.5)
        self.assertAlmostEqual(new, 0.5, places=6)

    def test_zero_trust_relation_can_acquire_evidence(self):
        cs = _cs()
        st = _store()
        st.append_relation(_v(1, 1), _v(0, 1), _v(0, 0, 1), trust=0.0)
        self.assertEqual(cs.verify_relation(0, [(_v(1, 0), _v(0, 0, 1))],
                                            store=st), 0.5)


# -- persistence: per-triple trust survives state_dict round-trip ----------

class TestRelativeTrustPersistence(unittest.TestCase):
    """The per-triple relation trust is a REGISTERED BUFFER (``trust``) so a
    checkpoint save/load recovers it. Before the fix it lived in a plain
    Python list outside the state_dict, so a reloaded relation reverted to
    the 1.0 fallback in ``reason`` / ``verify_relation``."""

    def test_trust_in_state_dict_roundtrip(self):
        st = _store()
        idx = st.append_relation(_v(1, 1), _v(0, 1), _v(0, 0, 1), trust=0.7)
        self.assertEqual(idx, 0)
        sd = st.state_dict()
        self.assertIn('trust', sd, "trust must be a serialized buffer")

        fresh = _store()
        self.assertEqual(float(fresh.trust[0]), 0.0)   # zero before load
        fresh.load_state_dict(sd)
        self.assertAlmostEqual(float(fresh.trust[0]), 0.7, places=6)
        # back-compat list view tracks the buffer over live rows.
        self.assertEqual(len(fresh), 1)
        self.assertAlmostEqual(fresh.trust[0], 0.7, places=6)

    def test_reason_over_reloaded_store_recovers_t1(self):
        cs = _cs()
        st = _store()
        st.append_relation(_v(1, 1), _v(0, 1), _v(0, 0, 1), trust=0.7)
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
        st.append_relation(_v(1, 1), _v(0, 1), _v(0, 0, 1), trust=0.7)
        sd = st.state_dict()
        del sd['trust']   # simulate a pre-fix checkpoint lacking the key
        fresh = _store()
        # non-strict load tolerates the missing key; trust stays zero-init.
        fresh.load_state_dict(sd, strict=False)
        self.assertEqual(float(fresh.trust[0]), 0.0)

    def test_verify_relation_writes_back_to_buffer(self):
        cs = _cs()
        st = _store()
        st.append_relation(_v(1, 1), _v(0, 1), _v(0, 0, 1), trust=0.5)
        eps = [(_v(1, 0), _v(0, 0, 1)), (_v(0, 1), _v(0, 0, 1))]
        new = cs.verify_relation(0, eps, store=st, support_weight=0.5)
        self.assertAlmostEqual(new, 0.5, places=6)
        # the write landed in the buffer (not a discarded list snapshot).
        self.assertAlmostEqual(float(st.c_plus[0]), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
