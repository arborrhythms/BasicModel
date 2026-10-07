"""Positive native addresses and retirement of position-keyed taxonomy."""

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


class TestNativeAddresses(unittest.TestCase):
    def test_word_concept_returns_a_positive_stable_address(self):
        from test_two_codebook_meta_taxonomy import _word_concept
        m = _make_radix_model()
        pos, row = _word_concept(m, b"alpha")
        self.assertIsInstance(pos, int)
        self.assertGreater(pos, 0)
        again, _ = _word_concept(m, b"alpha")
        self.assertEqual(again, pos)
        self.assertEqual(set(m._concept_owner().concept_parts(pos)), set(m.perceptualSpace.percept_store.identity.admit(b"alpha")))

    def test_idea_address_resolves_to_its_native_row(self):
        from test_two_codebook_meta_taxonomy import _idea_store, _write_point
        m = _make_radix_model()
        store = _idea_store(m)
        point = torch.zeros(store.nDim)
        point[0] = .5
        row = _write_point(store, point)
        pos = int(store.row_ids[row])
        self.assertIsInstance(pos, int)
        self.assertGreater(pos, 0)
        self.assertEqual(int(store.rel_type[row]), store.REL_NONE)
        native_row = store.index_of_row(pos)
        self.assertIsInstance(native_row, int)
        self.assertGreaterEqual(native_row, 0)
        self.assertEqual(int(store.row_ids[native_row]), pos)
