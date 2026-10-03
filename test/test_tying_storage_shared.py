"""UNTIED storage contract (Step 3, 2026-06-10 symbolic-iteration plan).

This file used to pin the strict-tying contract of the 2026-05-27
tied-storage refactor (PS.vocabulary rows aliasing SS.codebook rows
through one shared nn.Parameter). The tie is RETIRED: the lexicon keeps
PS-LOCAL storage permanently, and word insertion no longer reaches
across to the SS codebook at all. This file now pins the word-flow side
of the untied contract on the same fixture:

  * inserting a word grows ONLY the PS-side lexicon (the SS codebook
    prototype is bit-identical before/after);
  * the freshly inserted row is readable through ``weight`` (the
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
from Layers import RadixLayer  # noqa: E402
from util import init_config  # noqa: E402


def _build_model():
    Models.TheData.load("xor")
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model, _ = Models.BasicModel.from_config(_CONFIG)
    model.eval()
    return model


class TestUntiedWordFlow(unittest.TestCase):
    """Word insertion is PS-local; the SS codebook never hears about it."""

    def test_insert_grows_ps_only(self):
        model = _build_model()
        emb = model.perceptualSpace.vocabulary
        self.assertIsInstance(emb, RadixLayer)
        ws = model.wholeSpace
        W = ws.subspace.what.getW()
        ws_before = None if W is None else W.detach().clone()
        rows_before = int(emb._basis.W.shape[0])
        active_before = len(emb)
        parameter = emb._basis.W

        vec = torch.zeros(int(emb._basis.W.shape[1]))
        vec[0] = 0.7
        emb.insert(b"untiedword", init_vector=vec)

        self.assertEqual(len(emb), active_before + 1)
        self.assertEqual(int(emb._basis.W.shape[0]), rows_before)
        self.assertIs(emb._basis.W, parameter)
        if ws_before is not None:
            self.assertTrue(
                torch.equal(ws_before, ws.subspace.what.getW().detach()),
                "the SS codebook prototype must be bit-identical across a "
                "PS-side word insert (the paired-row reach-across is "
                "retired)")

    def test_inserted_row_reads_back_through_local_parameter(self):
        model = _build_model()
        emb = model.perceptualSpace.vocabulary
        weight = emb._basis.W
        vec = torch.zeros(int(weight.shape[1]))
        vec[0] = 0.25
        emb.insert(b"localrow", init_vector=vec)
        idx = emb.get_id(b"localrow")
        row = weight[idx].detach()
        self.assertAlmostEqual(float(row[0]), 0.25, places=5)
        self.assertEqual(
            weight.data_ptr(), emb._basis.W.data_ptr(),
            "the readable storage must be the LOCAL Parameter")

    def test_key_to_index_stays_ps_local(self):
        model = _build_model()
        emb = model.perceptualSpace.vocabulary
        weight = emb._basis.W
        n = int(weight.shape[0])
        for key, idx in list(emb.hash_map.items())[:64]:
            self.assertTrue(0 <= int(idx) < n, (
                f"key {key!r} maps to row {idx}, outside the PS-local "
                f"storage [0, {n}) -- the SS orth_idx remapping is "
                "retired; the decode inverse map expects PS-local rows"))


if __name__ == "__main__":
    unittest.main()
