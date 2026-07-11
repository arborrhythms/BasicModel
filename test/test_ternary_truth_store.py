"""Truth / Ideas consolidation -- stage 1: the unified ternary tensor store
(doc/specs/mereological-order-raising.md "Two consolidation questions"; Alec
2026-06-18).

``TernaryTruthStore`` combines the LTM end-state chain and the
RelativeTruthStore into ONE tensor of ``(NP1, VP, NP2)`` rows (full idea
vectors; Null = zero) + a per-row timestamp + a per-row scalar trust:

  * NP . .   = an IDEA (absolute truth)
  * NP VP .  = a unary predication
  * NP VP NP = an IDEA-RELATION-IDEA (relative truth)

Storage-only foundation: nothing constructs it yet, so the class is verified
in isolation here (registered buffers => persists across a state_dict
round-trip; trust stays a SEPARATE column, never baked into the magnitude).
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_D = 4


def _store(cap=8, **kw):
    from Layers import TernaryTruthStore
    return TernaryTruthStore(_D, capacity=cap, **kw)


def _v(*vals):
    v = torch.zeros(_D)
    for i, x in enumerate(vals):
        v[i] = x
    return v


class TestSchema(unittest.TestCase):
    def test_absolute_idea_is_np_null_null(self):
        s = _store()
        from Layers import TernaryTruthStore as T
        r = s.append_idea(_v(1), trust=0.6)
        self.assertEqual(r, 0)
        row = s.row(0)
        self.assertTrue(torch.equal(row['np1'], _v(1)))
        self.assertTrue(torch.equal(row['vp'], torch.zeros(_D)))   # Null
        self.assertTrue(torch.equal(row['np2'], torch.zeros(_D)))  # Null
        self.assertEqual(row['rel_type'], T.REL_NONE)
        self.assertAlmostEqual(row['trust'], 0.6, places=6)

    def test_relation_is_np_vp_np(self):
        s = _store()
        from Layers import TernaryTruthStore as T
        s.append_relation(_v(1), _v(0, 1), _v(0, 0, 1),
                          rel_type=T.REL_IMPLIES, trust=0.9)
        row = s.row(0)
        self.assertTrue(torch.equal(row['np1'], _v(1)))
        self.assertTrue(torch.equal(row['vp'], _v(0, 1)))
        self.assertTrue(torch.equal(row['np2'], _v(0, 0, 1)))
        self.assertEqual(row['rel_type'], T.REL_IMPLIES)

    def test_np_vp_null_unary(self):
        s = _store()
        from Layers import TernaryTruthStore as T
        s.append_relation(_v(1), _v(0, 1), None, rel_type=T.REL_OTHER)
        row = s.row(0)
        self.assertTrue(torch.equal(row['np2'], torch.zeros(_D)))  # Null tail

    def test_unscaled_storage_trust_is_separate(self):
        # trust is NOT baked into the magnitude (unlike RelativeTruthStore).
        s = _store()
        from Layers import TernaryTruthStore as T
        s.append_relation(_v(2), _v(0, 3), _v(0, 0, 4),
                          rel_type=T.REL_PARTOF, trust=0.5)
        row = s.row(0)
        self.assertTrue(torch.equal(row['np1'], _v(2)), "stored unscaled")
        self.assertTrue(torch.equal(row['np2'], _v(0, 0, 4)), "stored unscaled")
        self.assertAlmostEqual(row['trust'], 0.5, places=6)


class TestTimestampAndTrust(unittest.TestCase):
    def test_timestamp_monotonic(self):
        s = _store()
        s.append_idea(_v(1))
        s.append_idea(_v(2))
        s.append_idea(_v(3))
        self.assertEqual([s.row(i)['timestamp'] for i in range(3)],
                         [0.0, 1.0, 2.0])

    def test_explicit_timestamp_advances_clock(self):
        s = _store()
        s.append_idea(_v(1), timestamp=10.0)   # provisioned early/explicit
        r = s.append_idea(_v(2))               # next tick must be > 10
        self.assertEqual(s.row(r)['timestamp'], 11.0)

    def test_trust_clamped(self):
        s = _store()
        s.append_idea(_v(1), trust=5.0)
        s.append_idea(_v(2), trust=-5.0)
        self.assertEqual(s.row(0)['trust'], 1.0)
        self.assertEqual(s.row(1)['trust'], -1.0)

    def test_set_trust_overwrites(self):
        s = _store()
        s.append_idea(_v(1), trust=0.1)
        s.set_trust(0, 0.8)
        self.assertAlmostEqual(s.row(0)['trust'], 0.8, places=6)
        s.set_trust(0, 9.0)
        self.assertEqual(s.row(0)['trust'], 1.0, "set_trust clamps")


class TestQueries(unittest.TestCase):
    def test_recent_by_timestamp_desc(self):
        s = _store()
        s.append_idea(_v(1))   # ts 0
        s.append_idea(_v(2))   # ts 1
        s.append_idea(_v(3))   # ts 2
        self.assertEqual(s.recent(2).tolist(), [2, 1])
        self.assertEqual(s.recent(10).tolist(), [2, 1, 0])
        self.assertEqual(s.recent(0).tolist(), [])

    def test_relations_and_ideas_filters(self):
        s = _store()
        from Layers import TernaryTruthStore as T
        s.append_idea(_v(1))                                    # 0: idea
        s.append_relation(_v(1), _v(2), _v(3), rel_type=T.REL_PARTOF)   # 1
        s.append_relation(_v(4), _v(5), _v(6), rel_type=T.REL_IMPLIES)  # 2
        self.assertEqual(s.ideas().tolist(), [0])
        self.assertEqual(sorted(s.relations().tolist()), [1, 2])
        self.assertEqual(s.relations(T.REL_PARTOF).tolist(), [1])
        self.assertEqual(s.relations(T.REL_IMPLIES).tolist(), [2])

    def test_empty_queries(self):
        s = _store()
        self.assertEqual(s.recent(3).tolist(), [])
        self.assertEqual(s.relations().tolist(), [])
        self.assertEqual(s.ideas().tolist(), [])


class TestCapacityAndWidth(unittest.TestCase):
    def test_full_returns_minus_one(self):
        s = _store(cap=2)
        self.assertEqual(s.append_idea(_v(1)), 0)
        self.assertEqual(s.append_idea(_v(2)), 1)
        self.assertEqual(s.append_idea(_v(3)), -1, "full -> -1")
        self.assertEqual(len(s), 2)

    def test_width_conform(self):
        s = _store()
        long = torch.arange(_D + 3, dtype=torch.float32)   # longer than nDim
        s.append_idea(long)
        self.assertTrue(torch.equal(s.row(0)['np1'], long[:_D]))
        short = torch.tensor([7.0])                        # shorter than nDim
        s.append_idea(short)
        exp = torch.zeros(_D)
        exp[0] = 7.0
        self.assertTrue(torch.equal(s.row(1)['np1'], exp))


class TestPersistence(unittest.TestCase):
    def test_state_dict_roundtrip(self):
        from Layers import TernaryTruthStore as T
        s = _store()
        s.append_idea(_v(1), trust=0.3)
        s.append_relation(_v(1), _v(0, 1), _v(0, 0, 1),
                          rel_type=T.REL_IMPLIES, trust=-0.4)
        sd = s.state_dict()
        # every piece of state is a registered buffer -> rides state_dict
        for k in ('slots', 'rel_type', 'timestamp', 'trust', 'count',
                  '_next_ts'):
            self.assertIn(k, sd, f"{k} must be in state_dict")
        s2 = _store()
        s2.load_state_dict(sd)
        self.assertEqual(len(s2), 2)
        self.assertEqual(s2.row(0)['rel_type'], T.REL_NONE)
        self.assertAlmostEqual(s2.row(0)['trust'], 0.3, places=6)
        self.assertEqual(s2.row(1)['rel_type'], T.REL_IMPLIES)
        self.assertAlmostEqual(s2.row(1)['trust'], -0.4, places=6)
        self.assertTrue(torch.equal(s2.row(1)['np2'], _v(0, 0, 1)))
        # the clock survives so further appends keep increasing
        r = s2.append_idea(_v(9))
        self.assertEqual(s2.row(r)['timestamp'], 2.0)

    def test_reset_clears(self):
        s = _store()
        s.append_idea(_v(1), trust=0.5)
        s.reset()
        self.assertEqual(len(s), 0)
        self.assertEqual(s.recent(3).tolist(), [])
        r = s.append_idea(_v(2))
        self.assertEqual(s.row(r)['timestamp'], 0.0, "clock reset")


class TestOriginProvenance(unittest.TestCase):
    """Per-row writer provenance: the ``origin`` column + host-side source
    text distinguish conversation / provisioned / user rows sharing the
    store, and ``clear_origin`` gives runtime user rows replace-on-resubmit
    semantics without touching the other writers' rows."""

    def test_append_defaults_conversation_origin(self):
        from Layers import TernaryTruthStore as T
        s = _store()
        s.append_idea(_v(1))
        r = s.row(0)
        self.assertEqual(r['origin'], T.ORIGIN_CONVERSATION)
        self.assertIsNone(r['text'])

    def test_set_origin_tags_row_and_text(self):
        from Layers import TernaryTruthStore as T
        s = _store()
        s.append_idea(_v(1), trust=0.4)
        s.set_origin(0, T.ORIGIN_USER, text="the cat sat")
        r = s.row(0)
        self.assertEqual(r['origin'], T.ORIGIN_USER)
        self.assertEqual(r['text'], "the cat sat")
        # trust untouched by the origin tag
        self.assertAlmostEqual(r['trust'], 0.4, places=6)

    def test_set_origin_out_of_range_raises(self):
        from Layers import TernaryTruthStore as T
        s = _store()
        with self.assertRaises(IndexError):
            s.set_origin(0, T.ORIGIN_USER)

    def test_rows_of_origin_filters_in_row_order(self):
        from Layers import TernaryTruthStore as T
        s = _store()
        s.append_idea(_v(1))                       # 0: conversation
        s.append_idea(_v(2))                       # 1: -> provisioned
        s.append_idea(_v(3))                       # 2: -> user
        s.append_idea(_v(4))                       # 3: -> user
        s.set_origin(1, T.ORIGIN_PROVISIONED, text="p")
        s.set_origin(2, T.ORIGIN_USER, text="u1")
        s.set_origin(3, T.ORIGIN_USER, text="u2")
        self.assertEqual(
            s.rows_of_origin(T.ORIGIN_CONVERSATION).tolist(), [0])
        self.assertEqual(
            s.rows_of_origin(T.ORIGIN_USER).tolist(), [2, 3])
        # multi-origin select (the TruthLayer view's read) in row order
        self.assertEqual(
            s.rows_of_origin(T.ORIGIN_PROVISIONED,
                             T.ORIGIN_USER).tolist(), [1, 2, 3])
        self.assertEqual(s.rows_of_origin().tolist(), [])

    def test_clear_origin_compacts_survivors_in_place(self):
        from Layers import TernaryTruthStore as T
        s = _store()
        s.append_idea(_v(1), trust=0.1)            # 0: conversation
        s.append_idea(_v(2), trust=0.2)            # 1: user
        s.append_relation(_v(3), _v(0, 3), _v(0, 0, 3),
                          rel_type=T.REL_IMPLIES, trust=0.3)   # 2: prov
        s.append_idea(_v(4), trust=0.4)            # 3: user
        s.set_origin(1, T.ORIGIN_USER, text="u1")
        s.set_origin(2, T.ORIGIN_PROVISIONED, text="p")
        s.set_origin(3, T.ORIGIN_USER, text="u2")
        removed = s.clear_origin(T.ORIGIN_USER)
        self.assertEqual(removed, 2)
        self.assertEqual(len(s), 2)
        # survivors keep order, vectors, trust, rel_type, timestamp, text
        r0, r1 = s.row(0), s.row(1)
        self.assertTrue(torch.equal(r0['np1'], _v(1)))
        self.assertEqual(r0['origin'], T.ORIGIN_CONVERSATION)
        self.assertEqual(r0['timestamp'], 0.0)
        self.assertTrue(torch.equal(r1['np1'], _v(3)))
        self.assertEqual(r1['rel_type'], T.REL_IMPLIES)
        self.assertAlmostEqual(r1['trust'], 0.3, places=6)
        self.assertEqual(r1['timestamp'], 2.0)
        self.assertEqual(r1['text'], "p")
        # vacated tail rows are zeroed
        self.assertEqual(float(s.slots[2:4].abs().sum()), 0.0)
        self.assertEqual(int(s.origin[2:4].abs().sum()), 0)
        # the clock is NOT rewound: the next append lands after row 3's tick
        r = s.append_idea(_v(9))
        self.assertEqual(s.row(r)['timestamp'], 4.0)
        # idempotent: nothing left of that origin
        self.assertEqual(s.clear_origin(T.ORIGIN_USER), 0)

    def test_reset_clears_origin_and_texts(self):
        from Layers import TernaryTruthStore as T
        s = _store()
        s.append_idea(_v(1))
        s.set_origin(0, T.ORIGIN_USER, text="u")
        s.reset()
        self.assertEqual(s._texts, [])
        s.append_idea(_v(2))
        r = s.row(0)
        self.assertEqual(r['origin'], T.ORIGIN_CONVERSATION)
        self.assertIsNone(r['text'])

    def test_origin_rides_state_dict_texts_do_not(self):
        from Layers import TernaryTruthStore as T
        s = _store()
        s.append_idea(_v(1))
        s.set_origin(0, T.ORIGIN_USER, text="u")
        sd = s.state_dict()
        self.assertIn('origin', sd)
        s2 = _store()
        s2.load_state_dict(sd)
        r = s2.row(0)
        self.assertEqual(r['origin'], T.ORIGIN_USER)
        # source text is host-side transient metadata: absent after load,
        # readers fall back to None.
        self.assertIsNone(r['text'])


if __name__ == "__main__":
    unittest.main()
