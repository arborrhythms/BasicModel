"""Stage 1.D+1.B follow-up: tied orth storage.

The orthographic vector for a word should live in ONE place only --
``SymbolicSpace.subspace.what`` (the SS codebook). The legacy
``PerceptualSpace.vocabulary.wv._vectors`` ``nn.Parameter`` is GONE; what
remains on ``wv`` is a property / view that reads through SS.

Contract:
  * ``ps.vocabulary.wv._vectors[orth_idx]`` and
    ``ss.codebook[orth_idx]`` share ``.data_ptr()`` (same memory).
  * ``ps.vocabulary.wv._vectors`` is NOT an ``nn.Parameter`` on the
    PS-side WordVectors module any more (i.e. it is not in
    ``wv._parameters``).
  * An optimizer step on the joint model updates BOTH views (since they
    are the same tensor).
"""

import os
import sys
import tempfile
import unittest
import warnings
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn

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
from Spaces import Codebook, Embedding  # noqa: E402
from util import init_config  # noqa: E402


def _make_plain_model():
    """Cheap-boot MM_xor_loopback model; mirrors the Stage-1.A / 1.C
    test fixtures.
    """
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model, _ = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    model.eval()
    return model


class TestTiedOrthStorage(unittest.TestCase):
    """``PS.vocabulary.wv._vectors`` and ``SS.codebook.W`` are the same
    underlying storage. The PS side carries no separate Parameter.
    """

    def test_ps_vocabulary_vectors_share_dataptr_with_ss_codebook(self):
        """A row in PS.vocabulary[word_idx] and SS.codebook[orth_idx] for
        the same word share ``.data_ptr()``: there is no copy.
        """
        model = _make_plain_model()
        ss = model.symbolicSpace
        ps = model.perceptualSpace
        cb = ss.subspace.what

        ps_vec = torch.zeros(int(ss.nDim))
        ps_vec[0] = 0.7
        orth_idx, _ = ss.insert_paired_word("dataptrword", ps_vec)

        # Both must resolve to identical memory.
        emb = ps.vocabulary
        self.assertIsInstance(emb, Embedding)
        ps_row = emb.wv._vectors[orth_idx]
        ss_row = cb.getW()[orth_idx]
        self.assertEqual(
            ps_row.data_ptr(), ss_row.data_ptr(),
            "PS.vocabulary.wv._vectors[orth_idx] and SS.codebook[orth_idx] "
            "must share data_ptr (single tied storage); got "
            f"ps={ps_row.data_ptr()} vs ss={ss_row.data_ptr()}.",
        )

    def test_no_separate_vocabulary_vectors_parameter(self):
        """``wv._vectors`` is not registered as an ``nn.Parameter`` on the
        WordVectors module any more. The single Parameter for the orth
        rows lives on the SS codebook.
        """
        model = _make_plain_model()
        ps = model.perceptualSpace
        emb = ps.vocabulary
        wv = emb.wv
        # No nn.Parameter under the name _vectors in wv's own parameter
        # registry.
        self.assertNotIn(
            "_vectors", wv._parameters,
            "wv._parameters must NOT contain '_vectors' after the tied-"
            "storage refactor; the only Parameter for orth rows lives "
            "on SS.codebook.W (single trainable storage).",
        )

    def test_sgd_step_propagates_to_both_views(self):
        """A gradient step on the underlying SS parameter is visible from
        both ``ps.vocabulary[k]`` and ``ss.codebook[k]`` -- because they
        are the same tensor.
        """
        model = _make_plain_model()
        ss = model.symbolicSpace
        ps = model.perceptualSpace
        cb = ss.subspace.what

        ps_vec = torch.zeros(int(ss.nDim))
        ps_vec[0] = 0.25
        orth_idx, _ = ss.insert_paired_word("sgdtest", ps_vec)

        W = cb.getW()
        before_ps = ps.vocabulary.wv._vectors[orth_idx].detach().clone()
        before_ss = W[orth_idx].detach().clone()
        self.assertTrue(torch.allclose(before_ps, before_ss),
                        "Pre-step PS and SS rows must already match.")

        # Apply a synthetic delta to the SS parameter (no real optimizer
        # needed; the test is about the alias holding).
        with torch.no_grad():
            W.data[orth_idx, :] += 0.1

        after_ps = ps.vocabulary.wv._vectors[orth_idx].detach().clone()
        after_ss = W[orth_idx].detach().clone()
        self.assertTrue(
            torch.allclose(after_ps, after_ss),
            "After mutating SS.codebook.W in place, PS.vocabulary row "
            "must reflect the same delta (single tied tensor).",
        )
        self.assertTrue(
            torch.allclose(after_ps, before_ps + 0.1, atol=1e-6),
            "PS row must show the +0.1 delta applied via SS write.",
        )

    def test_optimizer_holds_only_one_parameter_for_orth_storage(self):
        """The training optimizer must not double-count the orth storage:
        there is exactly ONE ``data_ptr`` covering the orth Parameter (on
        SS), not two.
        """
        model = _make_plain_model()
        # Insert a couple of words so the orth storage is populated.
        ss = model.symbolicSpace
        ps_vec = torch.zeros(int(ss.nDim))
        ss.insert_paired_word("optword1", ps_vec)
        ss.insert_paired_word("optword2", ps_vec + 0.01)

        ps = model.perceptualSpace
        emb = ps.vocabulary
        wv = emb.wv

        # The SS codebook prototype Parameter ptr.
        ss_W = ss.subspace.what.getW()
        ss_ptr = ss_W.data_ptr()
        # The PS-side _vectors view's underlying storage ptr must match.
        wv_v = wv._vectors
        self.assertEqual(
            wv_v.data_ptr(), ss_ptr,
            "wv._vectors must share underlying storage with SS.codebook.W "
            "(same data_ptr at the matrix level); got "
            f"wv={wv_v.data_ptr()} vs ss={ss_ptr}.",
        )


if __name__ == "__main__":
    unittest.main()
