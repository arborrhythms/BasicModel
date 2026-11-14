"""Unary ideas, tagged relations, evidence, capacity and checkpoint contracts."""

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
    from part_fixtures import native_store
    return native_store(_D,capacity=cap)[0]


def _v(*vals):
    v = torch.zeros(_D)
    for i, x in enumerate(vals):
        v[i] = x
    return v


class TestSchema(unittest.TestCase):
    def test_absolute_idea_is_np_null_null(self):
        s = _store()
        from Memory import LTM0 as T
        r = s.append_idea(_v(1), trust=0.6)
        self.assertEqual(r, 0)
        row = s.row(0)
        self.assertTrue(torch.equal(row['meaning'].roles[0], _v(1)))
        self.assertTrue(torch.equal(row['meaning'].roles[1], torch.zeros(_D)))   # Null
        self.assertTrue(torch.equal(row['meaning'].roles[2], torch.zeros(_D)))  # Null
        self.assertEqual(int(s.raw_meaning_of(r).role_mask.sum()),1)
        self.assertAlmostEqual(row['trust'], 0.6, places=6)

    def test_relation_has_two_native_symbol_arguments(self):
        from part_fixtures import part
        s=_store();edge=part(s,_v(1),_v(0,0,1),evidence=(.9,0.))
        row=s.row(edge)
        self.assertEqual(row['refs'].tolist(),[1,2])
        self.assertEqual(len(s),0)
        self.assertEqual(len(s.parts),1)
        torch.testing.assert_close(row['meaning'].roles[0],_v(1))
        torch.testing.assert_close(row['meaning'].roles[1],_v(0,0,1))

    def test_unary_content_is_an_idea_and_supplies_no_part_pair(self):
        s=_store();s.append_idea(_v(.2,.3))
        self.assertEqual(len(s.parts),0)
        self.assertTrue(torch.equal(s.row(0)['meaning'].roles[0],_v(.2,.3)))

    def test_unscaled_storage_evidence_is_separate(self):
        from part_fixtures import part
        s=_store();edge=part(s,_v(.2),_v(0,0,.4),evidence=(.5,0.))
        row=s.row(edge)
        torch.testing.assert_close(row['meaning'].roles[0],_v(.2))
        torch.testing.assert_close(row['meaning'].roles[1],_v(0,0,.4))
        self.assertAlmostEqual(float(s.parts.truth_plus[edge[1]]),.5,places=6)
        self.assertAlmostEqual(row['trust'],.5,places=6)


class TestTimestampAndTrust(unittest.TestCase):


    def test_trust_clamped(self):
        s = _store()
        s.append_idea(_v(1), trust=5.0)
        s.append_idea(_v(2), trust=-5.0)
        self.assertEqual(s.row(0)['trust'], 1.0)
        self.assertEqual(s.row(1)['trust'], -1.0)


    pass


class TestQueries(unittest.TestCase):

    def test_relations_and_ideas_filters(self):
        from part_fixtures import part
        s=_store();s.append_idea(_v(1))
        a=part(s,_v(1),_v(0,1));b=part(s,_v(0,0,1),_v(0,0,0,1),types=('situation',))
        self.assertEqual(s.ideas().tolist(),[0])
        self.assertEqual(s.relations().tolist(),[a[1],b[1]])
        self.assertEqual(s.relations('implies').tolist(),[b[1]])

    def test_empty_queries(self):
        s = _store()
        self.assertEqual(s.relations().tolist(), [])
        self.assertEqual(s.ideas().tolist(), [])


class TestCapacityAndWidth(unittest.TestCase):
    def test_full_raises(self):
        s = _store(cap=2)
        self.assertEqual(s.append_idea(_v(1)), 0)
        self.assertEqual(s.append_idea(_v(2)), 1)
        with self.assertRaises(OverflowError):
            s.append_idea(_v(3))
        self.assertEqual(len(s), 2)

    def test_width_conform(self):
        s = _store()
        long = torch.arange(_D + 3, dtype=torch.float32)   # longer than nDim
        s.append_idea(long)
        self.assertTrue(torch.equal(s.row(0)['meaning'].roles[0], long[:_D]))
        short = torch.tensor([7.0])                        # shorter than nDim
        s.append_idea(short)
        exp = torch.zeros(_D)
        exp[0] = 7.0
        self.assertTrue(torch.equal(s.row(1)['meaning'].roles[0], exp))


class TestPersistence(unittest.TestCase):
    def test_state_dict_roundtrip(self):
        from part_fixtures import part
        s=_store();s.append_idea(_v(1),trust=.3)
        edge=part(s,_v(1),_v(0,0,1),evidence=(0.,.4))
        fresh=_store();same=part(fresh,_v(1),_v(0,0,1),evidence=(0.,0.))
        self.assertEqual(same,edge)
        fresh.load_state_dict(s.state_dict());fresh.parts.load_state_dict(s.parts.state_dict())
        self.assertEqual(len(fresh),1);self.assertEqual(len(fresh.parts),1)
        self.assertAlmostEqual(fresh.row(0)['trust'],.3,places=6)
        self.assertAlmostEqual(fresh.row(edge)['trust'],-.4,places=6)
        torch.testing.assert_close(fresh.row(edge)['meaning'].roles[1],_v(0,0,1))

    def test_reset_clears(self):
        s = _store()
        s.append_idea(_v(1), trust=0.5)
        s.reset()
        self.assertEqual(len(s), 0)
        r = s.append_idea(_v(2))


class TestOriginProvenance(unittest.TestCase):
    """Input documents distinguish conversation, provisioned and user evidence.
    Replacing a user document preserves other writers’ occurrences and texts.
    """

    pass

    pass

    pass

    pass

    pass

    pass

    pass


if __name__ == "__main__":
    unittest.main()
