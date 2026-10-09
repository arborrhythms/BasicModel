"""Native percept/property insertion; word-object META is owned by CS."""

from __future__ import annotations

import os
import sys
import unittest

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
    """Build the MM_xor radix-chunking model for end-to-end tests."""
    import warnings
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


def _word_concept(model, raw):
    store = model.perceptualSpace.percept_store
    parts = store.identity.admit(raw)
    owner = model._concept_owner()
    return owner.interpret.lookup_word(parts, (), form=raw), store.identity.word_rows[raw]


def _idea_store(model):
    from ClauseRow import attach_clause_index
    from Layers import TernaryTruthStore
    owner = model._concept_owner()
    store = TernaryTruthStore(int(owner.outputShape[-1]), capacity=8)
    attach_clause_index(model, store, owner)
    return store


def _write_point(store, point, **address):
    from ClauseRow import Clause
    from Meaning import ConceptualMeaning
    return store.write_clause(Clause(ConceptualMeaning.from_description(point), point=point), **address)


class TestInsertPercept(unittest.TestCase):
    def test_word_concept_has_positive_id_and_owned_percept_inverse(self):
        m = _make_radix_model()
        ps = m.perceptualSpace.percept_store
        self.assertIsNotNone(ps)
        starting_size = len(ps)
        pos, row = _word_concept(m, b"hello")
        self.assertIsInstance(pos, int)
        self.assertGreater(pos, 0)
        self.assertEqual(set(m._concept_owner().concept_parts(pos)), set(ps.identity.admit(b"hello")))
        self.assertGreaterEqual(row, starting_size)
        self.assertEqual(ps.bytes_for(row), b"hello")
        again, _ = _word_concept(m, b"hello")
        self.assertEqual(again, pos)


class TestInsertIdea(unittest.TestCase):
    def test_completed_point_gets_a_native_address_without_changing_earlier_rows(self):
        m = _make_radix_model()
        store = _idea_store(m)
        _write_point(store, torch.ones(store.nDim))
        before = store.slots.detach().clone()
        point = torch.zeros(store.nDim)
        point[0], point[1] = .7, -.3
        row = _write_point(store, point)
        pos = int(store.row_ids[row])
        self.assertIsInstance(pos, int)
        self.assertNotIn(pos, (-1, 0))
        self.assertEqual(int(store.rel_type[row]), store.REL_NONE)
        self.assertGreaterEqual(row, 0)
        self.assertLess(row, store.capacity)
        self.assertTrue(torch.allclose(store.slots[row, 0], point, atol=1e-5))
        for previous in range(row):
            self.assertTrue(torch.allclose(before[previous], store.slots[previous], atol=1e-6))

    def test_distinct_occurrences_get_distinct_addresses(self):
        m = _make_radix_model()
        store = _idea_store(m)
        first = _write_point(store, torch.ones(store.nDim))
        second = _write_point(store, torch.ones(store.nDim), document_key='another-occurrence')
        self.assertNotEqual(int(store.row_ids[first]), int(store.row_ids[second]))
