"""PS-LOCAL orth storage (Step 3, 2026-06-10 symbolic-iteration plan).

This file used to pin the 2026-05-27 TIED-storage contract (orth rows
living on the SS codebook; ``weight`` routed through ``cb.getW()``).
That tie -- ``insert_paired_word`` / ``tie_to_codebook`` / the
``_tie_lexicon_to_codebook`` migration -- is RETIRED: the Step-1 symbol
codebook on the CS leg captures the code-as-written vs
code-for-the-concept correspondence in place, and the lexicon keeps
PS-LOCAL storage permanently. The same file now pins the inverse:

  * ``weight`` resolves to the locally-owned ``_local_vectors``
    Parameter (registered on the WordVectors module).
  * PS lexicon storage and the SS codebook prototype are SEPARATE
    memory; in-place SS writes do not leak into PS rows.
  * ``WholeSpace.insert_paired_word`` no longer exists.
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


def _make_plain_model():
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model, _ = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    model.eval()
    return model


class TestPsLocalOrthStorage(unittest.TestCase):
    """The lexicon Parameter lives on PS permanently; no SS aliasing."""


    def test_local_parameter_is_registered(self):
        model = _make_plain_model()
        store = model.perceptualSpace.vocabulary
        basis = model.perceptualSpace.subspace.what
        from Layers import RadixLayer
        self.assertIsInstance(store, RadixLayer)
        self.assertIn("W", basis._parameters)
        self.assertIs(store._basis, basis)
        self.assertIs(store._basis.W, basis.W)

    def test_ps_storage_is_separate_from_ws_codebook(self):
        model = _make_plain_model()
        ws = model.wholeSpace
        weight = model.perceptualSpace.subspace.what.W
        W = ws.subspace.what.getW()
        if W is None:
            self.skipTest("SS codebook carries no prototype matrix")
        self.assertNotEqual(
            weight.data_ptr(), W.data_ptr(),
            "PS lexicon storage and the SS codebook prototype must be "
            "SEPARATE memory (the tie is retired).")

    def test_ws_write_does_not_leak_into_ps_rows(self):
        model = _make_plain_model()
        ws = model.wholeSpace
        weight = model.perceptualSpace.subspace.what.W
        W = ws.subspace.what.getW()
        if W is None:
            self.skipTest("SS codebook carries no prototype matrix")
        n = min(int(weight.shape[0]), int(W.shape[0]))
        if n == 0:
            self.skipTest("no overlapping rows to compare")
        before = weight[:n].detach().clone()
        with torch.no_grad():
            W.data[:n, :] += 0.1
        after = weight[:n].detach()
        self.assertTrue(
            torch.equal(before, after),
            "an in-place write to SS.codebook.W must NOT be visible "
            "through the PS lexicon rows (untied storage).")


if __name__ == "__main__":
    unittest.main()
