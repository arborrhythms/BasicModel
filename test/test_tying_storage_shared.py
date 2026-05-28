"""Strict-tying contract: PS.vocabulary[word_idx] and SS.codebook[orth_idx]
share the *same* memory (single trainable nn.Parameter on SS).

This is the user-driven contract from the 2026-05-27 tied-storage refactor:
no divergent copies; one Parameter; mutations through either view are
visible to the other.

The fixture must size SS such that the PS bootstrap (ASCII chars + the
NULL_PERCEPT key) can be migrated onto SS without overflow. The
``test_tied_orth_storage.py`` companion file exercises the same invariant
on the loopback config; this file adds a direct word-flow check using a
freshly inserted word.
"""

import os
import sys
import unittest
import warnings

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
from Spaces import Embedding  # noqa: E402
from util import init_config  # noqa: E402


def _build_model():
    """Cheap-boot MM_xor_loopback model; mirrors the Stage-1 fixtures."""
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model, _ = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    model.eval()
    return model


class TestTyingStorageShared(unittest.TestCase):
    """``ps.vocabulary.lookup(word)`` and ``ss.codebook.W[orth_idx]`` share
    the same underlying tensor storage after the tied-storage refactor.
    """

    def test_insert_paired_word_then_lookup_shares_dataptr(self):
        """After ``insert_paired_word(word, ps_vec)``, the row stored on
        ``SS.codebook.W[orth_idx]`` and the row retrieved from
        ``ps.vocabulary.wv._vectors[orth_idx]`` are the same tensor
        (identical ``data_ptr``).
        """
        model = _build_model()
        ss = model.symbolicSpace
        ps = model.perceptualSpace
        cb = ss.subspace.what
        emb = ps.vocabulary
        self.assertIsInstance(emb, Embedding)

        ps_vec = torch.zeros(int(ss.nDim))
        ps_vec[0] = 0.5
        orth_idx, _sem_idx = ss.insert_paired_word("tiedword", ps_vec)

        ss_row = cb.getW()[orth_idx]
        ps_row = emb.wv._vectors[orth_idx]
        self.assertEqual(
            ps_row.data_ptr(), ss_row.data_ptr(),
            "wv._vectors[orth_idx] and cb.getW()[orth_idx] must share "
            "storage (single tied Parameter); got "
            f"ps={ps_row.data_ptr()} vs ss={ss_row.data_ptr()}",
        )

    def test_no_separate_ps_vectors_parameter_after_tie(self):
        """After tying, ``wv._parameters`` does NOT contain a local
        ``_vectors`` / ``_local_vectors`` -- there is exactly one
        Parameter for the orth storage, on the SS codebook.
        """
        model = _build_model()
        ss = model.symbolicSpace
        ps = model.perceptualSpace
        cb = ss.subspace.what
        emb = ps.vocabulary

        # Insert at least one word so tying is forced.
        ps_vec = torch.zeros(int(ss.nDim))
        ss.insert_paired_word("ensure_tie", ps_vec)

        wv = emb.wv
        # No ``_vectors`` Parameter on wv.
        self.assertNotIn("_vectors", wv._parameters,
                         "wv._parameters must not contain '_vectors' "
                         "after tying.")
        # No ``_local_vectors`` Parameter either (the canonical Parameter
        # is on the SS codebook).
        self.assertNotIn("_local_vectors", wv._parameters,
                         "wv._parameters must not contain "
                         "'_local_vectors' after tying.")

    def test_in_place_write_via_ss_visible_through_ps_view(self):
        """A direct write to ``cb.getW().data[orth_idx]`` is reflected by
        a subsequent ``emb.wv._vectors[orth_idx]`` read, because both
        views index the same underlying tensor.
        """
        model = _build_model()
        ss = model.symbolicSpace
        ps = model.perceptualSpace
        cb = ss.subspace.what
        emb = ps.vocabulary

        ps_vec = torch.zeros(int(ss.nDim))
        ps_vec[0] = 0.25
        orth_idx, _ = ss.insert_paired_word("aliasword", ps_vec)

        W = cb.getW()
        delta = 0.7
        with torch.no_grad():
            W.data[orth_idx, :] += delta

        after_ps = emb.wv._vectors[orth_idx].detach().clone()
        after_ss = W[orth_idx].detach().clone()
        self.assertTrue(
            torch.allclose(after_ps, after_ss),
            "PS-side row must reflect direct SS write (alias contract).")


if __name__ == "__main__":
    unittest.main()
