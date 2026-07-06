"""Task 6c (doc/plans/2026-05-29-stm-serial-parallel-modes.md §7c).

After ``_maybe_learn_relation`` accepts a relation clearing
``truth_criterion``:

  * ``ws.taxonomy_children(predicate_pos)`` includes the two idea
    positions (the predicate is the META PARENT; the ideas are its
    children); and
  * the META node carries a tetralemma 4-tuple ``(t, f, b, n)`` summing
    to 1, retrievable on ``ws.meta_trust[predicate_pos]``.

Per the plan, ``_maybe_learn_relation`` is called DIRECTLY with hand-set
operand vectors (the full forward wiring is best-effort / separate); the
factor methods are mocked so the gate is guaranteed to accept and the
test focuses on the insertion shape + trust storage.

Also covers the WholeSpace insertion primitive (``insert_relation``)
in isolation, the idempotency of a re-derived relation, and the
``insert_meta(trust=...)`` storage path.
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


def _make_radix_model():
    import Models
    import Language
    from util import init_config
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    m.eval()
    return m


def _accept_all(cs):
    """Force the gate open by mocking the factors to 1.0 each."""
    cs._learn_score_children_in_codebook = lambda i1, i2: 1.0
    cs._learn_score_is_truth_obvious = lambda rel: 1.0
    cs._learn_score_resolves_contradiction = lambda rel: 1.0


class TestMaybeLearnRelationInsertion(unittest.TestCase):
    """``_maybe_learn_relation`` on an accepted relation produces the
    predicate-parent / two-children taxonomy + a tetralemma tuple."""

    def test_accepted_relation_taxonomy_and_trust(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        ws = cs.terminalSymbolSpace_ref
        cs.truth_criterion = 0.3
        _accept_all(cs)
        D = int(cs.nDim)
        predicate = torch.zeros(D)
        predicate[0] = 1.0
        idea1 = torch.zeros(D)
        idea1[1] = 1.0
        idea2 = torch.zeros(D)
        idea2[2] = 1.0

        pred_pos = cs._maybe_learn_relation(predicate, idea1, idea2)
        self.assertIsNotNone(pred_pos, "score 1.0 >= tc 0.3 must accept")
        self.assertIsInstance(pred_pos, int)
        self.assertGreater(pred_pos, 0)

        # Predicate is the META PARENT; the two ideas are its children.
        children = ws.taxonomy_children(pred_pos)
        self.assertEqual(
            len(children), 2,
            f"predicate must parent exactly the two ideas; got "
            f"{children!r}")
        for c in children:
            self.assertEqual(
                ws.taxonomy_parent(c), pred_pos,
                f"child {c} must point back to predicate {pred_pos}")
        # The predicate node is tagged META.
        self.assertTrue(ws.is_meta(pred_pos),
                        "predicate relation node must be tagged META")

        # Tetralemma 4-tuple recorded + retrievable + sums to 1.
        trust = ws.meta_trust.get(pred_pos)
        self.assertIsNotNone(
            trust, "accepted relation must record a tetralemma tuple")
        self.assertEqual(len(trust), 4,
                         f"trust must be a 4-tuple (t,f,b,n); got {trust!r}")
        self.assertAlmostEqual(sum(trust), 1.0, places=5)
        for w in trust:
            self.assertGreaterEqual(w, 0.0)

    def test_rejected_relation_writes_nothing(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        ws = cs.terminalSymbolSpace_ref
        cs.truth_criterion = 0.5
        # Product 0.0 < 0.5 -> reject.
        cs._learn_score_children_in_codebook = lambda i1, i2: 0.0
        cs._learn_score_is_truth_obvious = lambda rel: 1.0
        cs._learn_score_resolves_contradiction = lambda rel: 1.0
        D = int(cs.nDim)
        predicate = torch.zeros(D)
        predicate[3] = 1.0
        idea1 = torch.zeros(D)
        idea1[4] = 1.0
        idea2 = torch.zeros(D)
        idea2[5] = 1.0
        tax_before = dict(ws.taxonomy)
        trust_before = dict(ws.meta_trust)
        out = cs._maybe_learn_relation(predicate, idea1, idea2)
        self.assertIsNone(out)
        self.assertEqual(ws.taxonomy, tax_before,
                         "a rejected relation must not touch the taxonomy")
        self.assertEqual(ws.meta_trust, trust_before,
                         "a rejected relation must not record trust")


class TestInsertRelationPrimitive(unittest.TestCase):
    """``WholeSpace.insert_relation`` builds the predicate-parent /
    two-children edges + stores trust; idempotent on the triple."""

    def test_insert_relation_children_and_trust(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        D = int(ws.nDim)
        pred_pos = ws.insert_whole(init_vec=torch.zeros(D))
        idea1_pos = ws.insert_whole(init_vec=torch.zeros(D))
        idea2_pos = ws.insert_whole(init_vec=torch.zeros(D))
        trust = (0.5, 0.1, 0.3, 0.1)
        ret = ws.insert_relation(pred_pos, idea1_pos, idea2_pos,
                                 trust=trust)
        self.assertEqual(ret, pred_pos,
                         "insert_relation returns the predicate node")
        self.assertEqual(
            ws.taxonomy_children(pred_pos), [idea1_pos, idea2_pos],
            "children must be [idea1, idea2] in order")
        self.assertEqual(ws.taxonomy_parent(idea1_pos), pred_pos)
        self.assertEqual(ws.taxonomy_parent(idea2_pos), pred_pos)
        self.assertTrue(ws.is_meta(pred_pos))
        stored = ws.meta_trust.get(pred_pos)
        # Normalised: input already sums to 1.0 so it round-trips.
        self.assertAlmostEqual(sum(stored), 1.0, places=5)
        for got, exp in zip(stored, trust):
            self.assertAlmostEqual(got, exp, places=5)

    def test_insert_relation_idempotent(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        D = int(ws.nDim)
        pred_pos = ws.insert_whole(init_vec=torch.zeros(D))
        idea1_pos = ws.insert_whole(init_vec=torch.zeros(D))
        idea2_pos = ws.insert_whole(init_vec=torch.zeros(D))
        ws.insert_relation(pred_pos, idea1_pos, idea2_pos,
                           trust=(1, 0, 0, 0))
        # Re-insert the same triple: children must not duplicate; trust
        # overwrites with the new posture.
        ws.insert_relation(pred_pos, idea1_pos, idea2_pos,
                           trust=(0, 0, 1, 0))
        self.assertEqual(
            ws.taxonomy_children(pred_pos), [idea1_pos, idea2_pos],
            "re-insert must not duplicate children")
        self.assertAlmostEqual(ws.meta_trust[pred_pos][2], 1.0, places=5,
                               msg="re-insert overwrites trust (BOTH=1)")

    def test_insert_relation_rejects_nonpositive(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        D = int(ws.nDim)
        good = ws.insert_whole(init_vec=torch.zeros(D))
        with self.assertRaises(ValueError):
            ws.insert_relation(0, good, good)
        with self.assertRaises(ValueError):
            ws.insert_relation(good, -1, good)


class TestInsertMetaTrustKwarg(unittest.TestCase):
    """``insert_meta(trust=...)`` stores the tuple retrievably without
    breaking the existing (trust-absent) callers."""

    def test_insert_meta_without_trust_unchanged(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        pid = ws.insert_percept(b"no_trust")
        sid = ws.insert_whole()
        meta = ws.insert_meta(pid, sid)
        # No trust kwarg -> nothing recorded (autobind path is byte-equal).
        self.assertNotIn(meta, ws.meta_trust)

    def test_insert_meta_with_trust_stores(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        pid = ws.insert_percept(b"with_trust")
        sid = ws.insert_whole()
        meta = ws.insert_meta(pid, sid, trust=(2, 0, 1, 1))
        stored = ws.meta_trust.get(meta)
        self.assertIsNotNone(stored)
        self.assertAlmostEqual(sum(stored), 1.0, places=5)
        # 2/4, 0, 1/4, 1/4 after normalisation.
        self.assertAlmostEqual(stored[0], 0.5, places=5)

    def test_insert_meta_trust_survives_vocab_extras_roundtrip(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        pid = ws.insert_percept(b"persist_trust")
        sid = ws.insert_whole()
        meta = ws.insert_meta(pid, sid, trust=(1, 0, 0, 0))
        extras = ws.vocab_extras()
        self.assertIn("meta_trust", extras)
        from Spaces import WholeSpace
        ss2 = WholeSpace(
            list(ws.inputShape), list(ws.spaceShape), list(ws.outputShape))
        cb2 = ss2.subspace.what
        cb1 = ws.subspace.what
        if cb2.nVectors < cb1.nVectors:
            cb2.grow_to(int(cb1.nVectors))
        ss2.load_vocab_extras(extras)
        self.assertIn(meta, ss2.meta_trust)
        self.assertAlmostEqual(sum(ss2.meta_trust[meta]), 1.0, places=5)
        self.assertAlmostEqual(ss2.meta_trust[meta][0], 1.0, places=5)


class TestBoundaryHookForwardWiring(unittest.TestCase):
    """The sentence-boundary hook ``learn_relations_from_stm`` reads the
    depth-3 relative end-state from STM slots 0/1/2 and routes each
    relative row through the gate. Exercised on the REAL relative grammar
    (``complete.grammar`` via MentalModel.xml), mirroring the Task 6a
    relative-end-state test's STM + current_rules seeding.
    """

    @staticmethod
    def _build_mentalmodel():
        import Models
        import Language
        from util import init_config
        init_config(
            path=os.path.join(_DATA_DIR, "MentalModel.xml"),
            defaults_path=_DEFAULTS)
        Language.TheGrammar._configured = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m, _ = Models.BasicModel.from_config(
                os.path.join(_DATA_DIR, "MentalModel.xml"))
        return m

    def _seed_relative_stm(self, m, B):
        """Hand-set a depth-3 relative STM end-state per row (slots
        0/1/2 = predicate, idea1, idea2).

        Each slot is seeded NEAR a DISTINCT existing SS-codebook row so
        that -- on the MentalModel's fixed-width (``codebook_mode='none'``)
        codebook where new rows cannot be minted -- the three ideas
        resolve to three DISTINCT positions (predicate + 2 children),
        not all collapsing onto one nearest row. Slots 0/1/2 of row b map
        to codebook rows ``3b, 3b+1, 3b+2`` plus a tiny jitter."""
        stm = m.conceptualSpace.stm
        ws = m.conceptualSpace.terminalSymbolSpace_ref
        W = ws.subspace.what.getW()
        dim = int(stm.concept_dim)
        stm.ensure_batch(B)
        stm.ensure_capacity(8)
        # The MentalModel ``codebook_mode='none'`` SS codebook is an
        # untrained ZERO tensor (every row identical -> every idea would
        # snap to row 0). Pin the B*3 targeted rows to distinct one-hot
        # vectors so a realistic (post-training, populated) codebook is
        # simulated: each idea then resolves to its OWN row. This is the
        # configuration under which a learned relation is meaningful.
        with torch.no_grad():
            for b in range(B):
                for s in range(3):
                    row = (3 * b + s) % int(W.shape[0])
                    W.data[row, :].zero_()
                    W.data[row, row] = 1.0
        buf = torch.zeros(B, int(stm.capacity), dim)
        for b in range(B):
            for s in range(3):
                row = (3 * b + s) % int(W.shape[0])
                # Seed near codebook row `row` (jitter << inter-row gap)
                # so nearest_ws_row snaps each idea to its own row. "6+2+2": the
                # STM buffer is event-width (concept_dim), but the SS .what
                # codebook W is the bare content width; place the content in the
                # leading .what slots and leave the .where/.when tail zero.
                buf[b, s, :W.shape[1]] = W[row].detach().to(buf.dtype) + 1e-4
        stm._buffer = buf
        stm._depth = torch.full((B,), 3, dtype=torch.long)
        return dim

    def test_boundary_hook_inserts_relative_rows_only(self):
        m = self._build_mentalmodel()
        import Language
        g = Language.TheGrammar
        g._ensure_configured()
        # A forward relative rule id, found grammar-agnostically via the
        # relative detection set (REL_T in complete.grammar; an isEqual/isPart
        # output-role rule in the role-collapsed default).
        rel_ids = [
            rid for rid in sorted(g._relative_rule_id_set())
            if g.rules[rid].method_name in g._RELATIVE_OP_NAMES
            and ".reverse" not in (g.rules[rid].canonical or "")]
        if not rel_ids:
            # ADAPTED 2026-07-05: relation family relocated to <Queries>
            # (integration design pending) -- no grammar-level producer.
            self.skipTest("no relative parse rule in the configured "
                          "grammar -- relation family lives in <Queries>")
        rel_id = rel_ids[0]

        cs = m.conceptualSpace
        cs.truth_criterion = 0.3
        # Force the gate open so any relative row inserts.
        _accept_all(cs)
        # Generous "known concept" threshold so each idea (seeded 1e-4
        # from a distinct codebook row) resolves to its own existing row.
        cs._learn_children_dist_threshold = 10.0

        B = 3
        self._seed_relative_stm(m, B)
        # Per-row current_rules: rows 0 and 2 relative, row 1 absolute
        # (empty inner list -> not relative).
        m.symbolSpace.current_rules = {"SS": [[rel_id], [], [rel_id]]}

        rel_mask = m._sentence_relative_mask(B)
        self.assertEqual(rel_mask.tolist(), [True, False, True],
                         f"expected rows 0,2 relative; got {rel_mask.tolist()}")

        ws = cs.terminalSymbolSpace_ref
        accepted = cs.learn_relations_from_stm(rel_mask)
        # Two relative rows -> two inserted predicate METAs.
        self.assertEqual(
            len(accepted), 2,
            f"two relative rows must each insert a relation; got "
            f"{accepted!r}")
        for pred_pos in accepted:
            children = ws.taxonomy_children(pred_pos)
            self.assertEqual(len(children), 2,
                             f"each inserted relation parents two ideas; "
                             f"got {children!r}")
            self.assertTrue(ws.is_meta(pred_pos))
            self.assertIn(pred_pos, ws.meta_trust)
            self.assertAlmostEqual(sum(ws.meta_trust[pred_pos]), 1.0,
                                   places=5)

    def test_boundary_hook_noop_when_nothing_relative(self):
        """Absolute-only current_rules -> empty mask -> hook no-ops, no
        taxonomy mutation."""
        m = self._build_mentalmodel()
        cs = m.conceptualSpace
        ws = cs.terminalSymbolSpace_ref
        _accept_all(cs)
        B = 2
        self._seed_relative_stm(m, B)
        m.symbolSpace.current_rules = {"SS": [[], []]}
        rel_mask = m._sentence_relative_mask(B)
        self.assertFalse(bool(rel_mask.any()))
        tax_before = dict(ws.taxonomy)
        accepted = cs.learn_relations_from_stm(rel_mask)
        self.assertEqual(accepted, [])
        self.assertEqual(ws.taxonomy, tax_before,
                         "no relative row -> taxonomy untouched")


if __name__ == "__main__":
    unittest.main()
