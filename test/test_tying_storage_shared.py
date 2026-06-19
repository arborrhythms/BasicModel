"""UNTIED storage contract (Step 3, 2026-06-10 symbolic-iteration plan).

This file used to pin the strict-tying contract of the 2026-05-27
tied-storage refactor (PS.vocabulary rows aliasing SS.codebook rows
through one shared nn.Parameter). The tie is RETIRED: the lexicon keeps
PS-LOCAL storage permanently, and word insertion no longer reaches
across to the SS codebook at all. This file now pins the word-flow side
of the untied contract on the same fixture:

  * inserting a word grows ONLY the PS-side lexicon (the SS codebook
    prototype is bit-identical before/after);
  * the freshly inserted row is readable through ``wv._vectors`` (the
    local Parameter) and carries the inserted values;
  * ``key_to_index`` keeps the identity-style PS-local row mapping the
    decode's inverse map relies on (no SS orth_idx remapping).
"""

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
_CONFIG = os.path.join(_DATA_DIR, "MM_xor_loopback.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")

import Models  # noqa: E402
import Language  # noqa: E402
from Spaces import Embedding  # noqa: E402
from util import init_config  # noqa: E402


def _build_model():
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model, _ = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    model.eval()
    return model


class TestUntiedWordFlow(unittest.TestCase):
    """Word insertion is PS-local; the SS codebook never hears about it."""

    def test_insert_grows_ps_only(self):
        model = _build_model()
        emb = model.perceptualSpace.vocabulary
        self.assertIsInstance(emb, Embedding)
        ws = model.wholeSpace
        W = ws.subspace.what.getW()
        ws_before = None if W is None else W.detach().clone()
        rows_before = int(emb.wv._vectors.shape[0])

        vec = torch.zeros(int(emb.wv._vectors.shape[1]))
        vec[0] = 0.7
        emb.insert("untiedword", vector=vec)

        self.assertEqual(int(emb.wv._vectors.shape[0]), rows_before + 1,
                         "insert must grow the PS-side lexicon by one row")
        if ws_before is not None:
            self.assertTrue(
                torch.equal(ws_before, ws.subspace.what.getW().detach()),
                "the SS codebook prototype must be bit-identical across a "
                "PS-side word insert (the paired-row reach-across is "
                "retired)")

    def test_inserted_row_reads_back_through_local_parameter(self):
        model = _build_model()
        emb = model.perceptualSpace.vocabulary
        wv = emb.wv
        vec = torch.zeros(int(wv._vectors.shape[1]))
        vec[0] = 0.25
        emb.insert("localrow", vector=vec)
        idx = wv.key_to_index["localrow"]
        row = wv._vectors[idx].detach()
        self.assertAlmostEqual(float(row[0]), 0.25, places=5)
        self.assertEqual(
            wv._vectors.data_ptr(), wv._local_vectors.data_ptr(),
            "the readable storage must be the LOCAL Parameter")

    def test_key_to_index_stays_ps_local(self):
        model = _build_model()
        emb = model.perceptualSpace.vocabulary
        wv = emb.wv
        n = int(wv._vectors.shape[0])
        for key, idx in list(wv.key_to_index.items())[:64]:
            self.assertTrue(0 <= int(idx) < n, (
                f"key {key!r} maps to row {idx}, outside the PS-local "
                f"storage [0, {n}) -- the SS orth_idx remapping is "
                "retired; the decode inverse map expects PS-local rows"))


if __name__ == "__main__":
    unittest.main()
