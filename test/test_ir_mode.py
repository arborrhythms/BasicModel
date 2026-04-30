"""Tests for IR (Input Reconstruction) mode building blocks.

See doc/specs/2026-04-29-ir-mode-design.md.

These tests cover the unit-level building blocks of IR. Integration
verification (end-to-end forward+reverse with a real model) is
deferred to the smoke run in step 10 of the design plan and lives
outside the unit-test suite because it requires a built lexicon
artifact.
"""
import unittest
import warnings
from types import SimpleNamespace

import torch

warnings.filterwarnings("ignore", message=".*script_method.*",
                        category=DeprecationWarning)

from util import parse
from Spaces import Embedding, NULL_PERCEPT_KEY


class TestWordLexerRegex(unittest.TestCase):
    """Step 2: word-mode regex tokenizes non-letters individually.

    ``parse`` returns the regex-matched runs. The downstream lex_batch
    appends a trailing ``\\x00`` end-of-content terminator, so the
    InputSpace sees 6 tokens for ``"Hello, world!"`` (5 + NULL); these
    tests verify the parse-level slice (5 tokens, no NULL).
    """

    def test_hello_world_lex_runs(self):
        toks = parse("Hello, world!", lex="words")
        texts = [t for t, _ in toks]
        self.assertEqual(texts, ["Hello", ",", " ", "world", "!"])

    def test_letter_runs_grouped(self):
        toks = parse("foobar", lex="words")
        self.assertEqual([t for t, _ in toks], ["foobar"])

    def test_digits_individual(self):
        toks = parse("123", lex="words")
        self.assertEqual([t for t, _ in toks], ["1", "2", "3"])

    def test_whitespace_individual(self):
        toks = parse("a  b", lex="words")
        # 'a', ' ', ' ', 'b'
        self.assertEqual([t for t, _ in toks], ["a", " ", " ", "b"])


class TestNullPerceptSlot(unittest.TestCase):
    """Step 1: NULL-percept slot at end of Embedding codebook."""

    def _make_embedding(self):
        emb = Embedding()
        # Construct without an embedding file so the create() path uses the
        # dynamic-vocab fallback (puts \x00 at index 0, then ASCII 1..126).
        emb.create(nInput=8, nVectors=8, nDim=4, passThrough=True,
                   embedding_path=None)
        return emb

    def test_null_percept_idx_is_at_tail(self):
        emb = self._make_embedding()
        # Tail slot lives at the last index; the WordVectors size includes it.
        self.assertEqual(emb.null_percept_idx, len(emb.wv) - 1)

    def test_null_percept_key_in_vocab(self):
        emb = self._make_embedding()
        self.assertIn(NULL_PERCEPT_KEY, emb.wv.key_to_index)
        self.assertEqual(
            emb.wv.key_to_index[NULL_PERCEPT_KEY], emb.null_percept_idx)

    def test_null_percept_vector_in_unit_ball(self):
        emb = self._make_embedding()
        vec = emb.getW()[emb.null_percept_idx]
        # Wrapped values live in [-1, 1).
        self.assertTrue(bool((vec >= -1.0).all()))
        self.assertTrue(bool((vec < 1.0).all()))

    def test_null_percept_vector_is_learnable(self):
        emb = self._make_embedding()
        # The trailing slot is part of the same learnable Parameter.
        self.assertTrue(emb.wv._vectors.requires_grad)


class TestIRInjectMask(unittest.TestCase):
    """Step 5: mask injection replaces WHAT slice with NULL_PERCEPT."""

    def _build_minimal_subspace(self, B=2, K=4, nWhat=3, nWhere=2):
        """Build the slice of state ``create_ir_mask`` actually reads.

        Avoids spinning up the full model factory: the helper only needs
        ``percept_subspace.event.getW/setW``, ``.what.null_percept_idx``,
        ``.what.getW()``, and ``._active``.
        """
        D = nWhat + nWhere
        event_data = torch.randn(B, K, D)
        # Pretend slot 0 of every row is padding (codebook index 0).
        active = torch.zeros(B, K, 1, dtype=torch.long)
        active[:, 1:, 0] = 5  # any non-zero index marks "real" positions

        codebook_W = torch.randn(8, nWhat)
        codebook_W[7] = torch.tensor([0.5, -0.5, 0.25])  # null-percept vec
        codebook = SimpleNamespace(
            null_percept_idx=7,
            getW=lambda: codebook_W,
        )
        event_basis = SimpleNamespace(
            getW=lambda: event_data,
            setW=lambda v: setattr(event_basis, '_W', v),
            _W=event_data,
        )
        # Re-wire getW/setW to the mutable backing _W.
        event_basis.getW = lambda: event_basis._W
        event_basis.setW = lambda v: setattr(event_basis, '_W', v)
        event_basis._W = event_data
        subspace = SimpleNamespace(
            event=event_basis,
            what=codebook,
            _active=active,
            is_empty=lambda: False,
        )
        return subspace, codebook_W[7], nWhat

    def test_mask_excludes_padding(self):
        """Padding slots (active index 0) never get masked."""
        from Models import BasicModel
        m = BasicModel()
        m.mask_rate = 1.0  # try to mask everything; only non-padding should pass
        subspace, _, _ = self._build_minimal_subspace(B=2, K=4)
        m.create_ir_mask(subspace)
        mask = m._ir_mask_positions
        self.assertIsNotNone(mask)
        # Slot 0 is padding, so it must not be masked anywhere.
        self.assertFalse(bool(mask[:, 0].any()))
        # Non-padding slots should all be masked at rate=1.0.
        self.assertTrue(bool(mask[:, 1:].all()))

    def test_mask_preserves_where_when(self):
        """Mask edits only the WHAT slice; WHERE/WHEN remain untouched."""
        from Models import BasicModel
        m = BasicModel()
        m.mask_rate = 1.0
        subspace, null_vec, nWhat = self._build_minimal_subspace(
            B=1, K=3, nWhat=3, nWhere=2)
        pre = subspace.event.getW().clone()
        m.create_ir_mask(subspace)
        post = subspace.event.getW()
        mask = m._ir_mask_positions
        # WHERE slice (last nWhere dims) is unchanged everywhere.
        self.assertTrue(torch.allclose(pre[..., nWhat:], post[..., nWhat:]))
        # WHAT slice at masked positions is the null vector.
        for b in range(mask.shape[0]):
            for k in range(mask.shape[1]):
                if bool(mask[b, k]):
                    self.assertTrue(torch.allclose(
                        post[b, k, :nWhat], null_vec))

    def test_zero_rate_no_mask(self):
        """mask_rate=0 -> nothing masked, but a snapshot is still stored."""
        from Models import BasicModel
        m = BasicModel()
        m.mask_rate = 0.0
        subspace, _, _ = self._build_minimal_subspace()
        m.create_ir_mask(subspace)
        # Per the helper's contract, rate=0 short-circuits before sampling.
        self.assertTrue(
            m._ir_mask_positions is None
            or not bool(m._ir_mask_positions.any()))


if __name__ == '__main__':
    unittest.main()
