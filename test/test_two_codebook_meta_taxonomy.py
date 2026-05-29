"""Stage 8: Two-codebook split + META taxonomy.

Tests the new SymbolicSpace API surface:

  * ``insert_percept(canonical_bytes) -> int (positive)`` delegates to
    ``PerceptualSpace.percept_store``.
  * ``insert_symbol(init_vec=None) -> int (negative)`` allocates a new
    SS.codebook row.
  * ``insert_meta(ps_idx, ss_idx, fused_vec=None) -> int (negative)``
    allocates a META node binding ``(ps_idx, ss_idx)``; idempotent on
    the (ps_idx, ss_idx) pair (subsequent calls return the same META
    idx and EMA-update the stored fused vec).
  * ``taxonomy``, ``taxonomy_parent`` (dicts), ``taxonomy_children``,
    ``is_meta`` helpers.
  * Reverse decode: terminal CS state -> SS nearest match -> walk META
    taxonomy -> PS percept_id -> ``inverse_table`` -> canonical bytes.
  * Persistence roundtrip: taxonomy + parent + meta-key map survive
    ``vocab_extras`` save/load.

The sign convention (locked in the plan §Architectural decisions #3):

  * positive ``i`` -> ``PS.percept_store.codebook[i]``,
    ``PS.percept_store.inverse_table[i]`` for surface bytes.
  * negative ``i`` -> ``SS.codebook[-i - 1]``.

Stage 8 of doc/plans/2026-05-27-perceptstore-meta-taxonomy-reentrancy.md.
"""

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


# ---------------------------------------------------------------------------
# Sign-convention helpers (mirror the implementation contract).
# ---------------------------------------------------------------------------


def _ss_row_from_signed(signed_idx):
    """``signed_idx`` -> SS.codebook row index. Raises if positive."""
    if signed_idx >= 0:
        raise AssertionError(
            f"SS row expected from negative signed idx, got {signed_idx}")
    return -signed_idx - 1


def _ps_row_from_signed(signed_idx):
    """``signed_idx`` -> PS.percept_store row index. Raises if negative."""
    if signed_idx < 0:
        raise AssertionError(
            f"PS row expected from positive signed idx, got {signed_idx}")
    return int(signed_idx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInsertPercept(unittest.TestCase):
    """``insert_percept`` delegates to PerceptStore; returns positive idx."""

    def test_insert_percept_returns_positive_idx_and_inserts(self):
        m = _make_radix_model()
        ss = m.symbolicSpace
        # PerceptStore starts empty in a freshly built model
        ps = m.perceptualSpace.percept_store
        self.assertIsNotNone(ps,
                             "MM_xor radix-mode model must have a "
                             "percept_store on PerceptualSpace")
        starting_size = len(ps)
        signed_idx = ss.insert_percept(b"hello")
        self.assertIsInstance(signed_idx, int)
        self.assertGreaterEqual(signed_idx, 0,
                                "insert_percept must return a positive "
                                f"signed idx; got {signed_idx}")
        ps_row = _ps_row_from_signed(signed_idx)
        self.assertEqual(ps_row, starting_size,
                         "the new percept id should be the next slot")
        self.assertEqual(ps.bytes_for(ps_row), b"hello",
                         "inverse_table lookup must match insertion")
        # Repeat insert of the same canonical bytes returns the same id.
        again = ss.insert_percept(b"hello")
        self.assertEqual(again, signed_idx,
                         "re-inserting same bytes must return the same idx")


class TestInsertSymbol(unittest.TestCase):
    """``insert_symbol`` allocates a fresh SS row; returns negative idx."""

    def test_insert_symbol_returns_negative_idx_and_writes_row(self):
        m = _make_radix_model()
        ss = m.symbolicSpace
        cb = ss.subspace.what
        W_before = cb.getW().detach().clone()
        # Use a custom init vector so the test can check exact write.
        init_vec = torch.zeros(int(ss.nDim))
        init_vec[0] = 0.7
        init_vec[1] = -0.3
        signed_idx = ss.insert_symbol(init_vec=init_vec)
        self.assertIsInstance(signed_idx, int)
        self.assertLess(signed_idx, 0,
                        "insert_symbol must return a negative signed idx; "
                        f"got {signed_idx}")
        ss_row = _ss_row_from_signed(signed_idx)
        self.assertGreaterEqual(ss_row, 0)
        self.assertLess(ss_row, cb.nVectors)
        W_after = cb.getW().detach()
        # Compare on the codebook's dtype/device.
        expected = init_vec.to(device=W_after.device, dtype=W_after.dtype)
        self.assertTrue(
            torch.allclose(W_after[ss_row], expected, atol=1e-5),
            f"SS row {ss_row} should match init_vec; got "
            f"{W_after[ss_row].tolist()} vs {expected.tolist()}")
        # Other pre-existing rows untouched.
        for r in range(min(W_before.shape[0], W_after.shape[0])):
            if r == ss_row:
                continue
            self.assertTrue(
                torch.allclose(W_before[r], W_after[r], atol=1e-6),
                f"row {r} changed unexpectedly after insert_symbol")

    def test_insert_symbol_default_init_random(self):
        """init_vec=None falls back to a random-ish init."""
        m = _make_radix_model()
        ss = m.symbolicSpace
        idx_a = ss.insert_symbol()
        idx_b = ss.insert_symbol()
        self.assertNotEqual(idx_a, idx_b,
                            "Distinct insert_symbol calls must return "
                            "distinct signed idxs")


class TestInsertMeta(unittest.TestCase):
    """``insert_meta(ps_idx, ss_idx, fused_vec)`` allocates a META node."""

    def test_insert_meta_basic_creates_taxonomy_entry(self):
        m = _make_radix_model()
        ss = m.symbolicSpace
        ps_idx = ss.insert_percept(b"meta_a")
        ss_idx = ss.insert_symbol()
        meta_idx = ss.insert_meta(ps_idx, ss_idx)
        self.assertLess(meta_idx, 0,
                        "insert_meta must return a negative signed idx (a "
                        f"new SS row hosting the META vector); got {meta_idx}")
        # taxonomy_children: meta -> [ps, ss]
        children = ss.taxonomy_children(meta_idx)
        self.assertEqual(set(children), {ps_idx, ss_idx},
                         f"META node children should be {{ps_idx, ss_idx}}; "
                         f"got {children!r}")
        # taxonomy_parent: ps -> meta, ss -> meta
        self.assertEqual(ss.taxonomy_parent(ps_idx), meta_idx)
        self.assertEqual(ss.taxonomy_parent(ss_idx), meta_idx)
        # is_meta predicate
        self.assertTrue(ss.is_meta(meta_idx),
                        "is_meta must return True for a META node")
        self.assertFalse(ss.is_meta(ps_idx),
                         "is_meta must return False for a pure percept idx")
        self.assertFalse(ss.is_meta(ss_idx),
                         "is_meta must return False for a pure symbol idx")

    def test_insert_meta_default_fused_vec_is_average(self):
        """When fused_vec is None, the META row defaults to the average of
        PS.codebook[ps_idx] and SS.codebook[ss_idx]."""
        m = _make_radix_model()
        ss = m.symbolicSpace
        ps = m.perceptualSpace.percept_store
        ps_idx = ss.insert_percept(b"meta_b")
        # Pin PS row to a known value so we can predict the average.
        with torch.no_grad():
            ps.codebook.data[ps_idx, :].zero_()
            ps.codebook.data[ps_idx, 0] = 1.0
        sym_init = torch.zeros(int(ss.nDim))
        sym_init[1] = 1.0
        ss_idx = ss.insert_symbol(init_vec=sym_init)
        ss_row_of_input = _ss_row_from_signed(ss_idx)

        meta_idx = ss.insert_meta(ps_idx, ss_idx)
        meta_row = _ss_row_from_signed(meta_idx)
        W = ss.subspace.what.getW()
        expected = (ps.codebook[ps_idx] + W[ss_row_of_input]) / 2.0
        actual = W[meta_row]
        # Tolerance handles device/dtype variation.
        self.assertTrue(
            torch.allclose(actual.detach(),
                           expected.detach().to(actual.device, actual.dtype),
                           atol=1e-5),
            f"META row {meta_row} should default to average of PS+SS; "
            f"got {actual.tolist()} vs expected {expected.tolist()}")

    def test_insert_meta_is_idempotent_with_ema_update(self):
        m = _make_radix_model()
        ss = m.symbolicSpace
        ps_idx = ss.insert_percept(b"meta_c")
        ss_idx = ss.insert_symbol()
        # First insert with an explicit fused_vec.
        fused1 = torch.zeros(int(ss.nDim))
        fused1[0] = 1.0
        meta_idx_1 = ss.insert_meta(ps_idx, ss_idx, fused_vec=fused1)
        meta_row = _ss_row_from_signed(meta_idx_1)
        W = ss.subspace.what.getW()
        first_stored = W[meta_row].detach().clone()
        # Second call with the same (ps_idx, ss_idx) MUST return the same
        # META idx AND ema-update the stored vec.
        fused2 = torch.zeros(int(ss.nDim))
        fused2[1] = 1.0
        meta_idx_2 = ss.insert_meta(ps_idx, ss_idx, fused_vec=fused2)
        self.assertEqual(meta_idx_1, meta_idx_2,
                         "insert_meta on the same (ps, ss) pair must return "
                         "the same meta idx")
        # New stored vec is somewhere between first and fused2 (EMA).
        second_stored = ss.subspace.what.getW()[meta_row].detach()
        self.assertFalse(
            torch.allclose(second_stored, first_stored, atol=1e-6),
            "Second insert_meta call should EMA-update the stored vec")
        # Children list AND parent map must remain consistent.
        self.assertEqual(set(ss.taxonomy_children(meta_idx_1)),
                         {ps_idx, ss_idx})
        self.assertEqual(ss.taxonomy_parent(ps_idx), meta_idx_1)
        self.assertEqual(ss.taxonomy_parent(ss_idx), meta_idx_1)


class TestReverseDecodeStructural(unittest.TestCase):
    """Reverse decode: terminal CS state -> SS nearest match -> META
    taxonomy walk -> PS percept -> inverse_table -> bytes.

    This test bypasses the full model forward and exercises the decode
    surface directly: build a tiny SS codebook with META nodes, set up
    a synthetic terminal CS state that's closest to one of them, and
    verify the structural decode produces the registered PS bytes
    exactly.
    """

    def test_reverse_decode_walks_meta_taxonomy_to_bytes(self):
        m = _make_radix_model()
        ss = m.symbolicSpace
        ps = m.perceptualSpace.percept_store
        # Insert three (word, symbol, meta) bundles. Pin the META rows
        # so we can synthesize a terminal CS state that is closest to
        # one of them.
        words = [b"alpha", b"beta", b"gamma"]
        meta_idxs = []
        for w in words:
            pid = ss.insert_percept(w)
            sid = ss.insert_symbol()
            mid = ss.insert_meta(pid, sid)
            meta_idxs.append(mid)
        # Pin the META rows to deterministic, well-separated vectors so
        # nearest-neighbour search is unambiguous.
        cb = ss.subspace.what
        D = int(ss.nDim)
        pinned = {}
        for i, mid in enumerate(meta_idxs):
            row = _ss_row_from_signed(mid)
            vec = torch.zeros(D, device=cb.getW().device, dtype=cb.getW().dtype)
            vec[i] = 1.0
            with torch.no_grad():
                cb.getW().data[row, :] = vec
            pinned[mid] = vec
        # Pick a target word -> META -> pinned vector. Synthesize a
        # terminal CS state matching that vector (the nearest-match
        # should snap to it).
        target_idx = 1  # "beta"
        target_meta = meta_idxs[target_idx]
        target_vec = pinned[target_meta]
        # Now invoke the structural decode helper. We expect the path:
        #  target_vec -> nearest META on SS -> taxonomy children ->
        #  positive PS child -> percept_store.bytes_for() == b"beta".
        bytes_out = m._reverse_decode_one(target_vec)
        self.assertEqual(bytes_out, words[target_idx],
                         f"Expected reverse decode to recover "
                         f"{words[target_idx]!r}, got {bytes_out!r}")


class TestStructuralReverseDecodeIntegration(unittest.TestCase):
    """End-to-end integration: with the radix path active and a populated
    META taxonomy, ``_decode_reconstructed_inputs`` invokes the structural
    decode (SS nearest -> META children -> PS percept id -> inverse_table
    bytes), producing the canonical bytes (not the legacy
    ``\\x01 \\x01 \\x01`` failure mode).
    """

    def test_decode_reconstructed_inputs_uses_structural_path(self):
        """Build a small synthetic META taxonomy, hand the matching SS
        codebook vectors to ``_decode_reconstructed_inputs`` as the
        "reconstructed input", and assert it returns the registered
        surface bytes.
        """
        m = _make_radix_model()
        ss = m.symbolicSpace
        ps = m.perceptualSpace.percept_store
        words = [b"hello", b"world", b"foo"]
        meta_idxs = []
        for w in words:
            pid = ss.insert_percept(w)
            sid = ss.insert_symbol()
            mid = ss.insert_meta(pid, sid)
            meta_idxs.append(mid)
        # Pin META rows to deterministic vectors.
        cb = ss.subspace.what
        W = cb.getW()
        D = int(ss.nDim)
        for i, mid in enumerate(meta_idxs):
            row = _ss_row_from_signed(mid)
            vec = torch.zeros(D, device=W.device, dtype=W.dtype)
            vec[i] = 1.0
            with torch.no_grad():
                W.data[row, :].copy_(vec)
        # Synthesize a "reconstructed input" tensor: one batch row, N
        # slots, each slot pinned to one META vector. Shape [B, N, D].
        n_slots = len(words)
        recon = torch.zeros(1, n_slots, D, device=W.device, dtype=W.dtype)
        for i, mid in enumerate(meta_idxs):
            row = _ss_row_from_signed(mid)
            recon[0, i, :] = W[row].detach()
        # Originals only used for the per-row token-count clip; for the
        # decode call it's a single sentence with N tokens.
        originals = [" ".join(w.decode("utf-8") for w in words)]
        rendered = m._decode_reconstructed_inputs(recon, originals)
        self.assertEqual(len(rendered), 1)
        out = rendered[0]
        # Every input word should be present somewhere in the rendered
        # string (the order is the slot order: words[0], words[1], ...).
        self.assertIn("hello", out,
                      f"structural decode dropped 'hello': out={out!r}")
        self.assertIn("world", out)
        self.assertIn("foo", out)
        # No \x01-padding from a legacy near-init snap.
        self.assertNotIn("\x01", out,
                         f"structural decode produced \\x01 padding -- "
                         f"the legacy nearest-init fallback is firing: "
                         f"out={out!r}")


class TestPersistenceRoundtrip(unittest.TestCase):
    """The taxonomy and parent dicts and the (ps,ss) -> meta lookup must
    survive a ``vocab_extras`` save/load cycle.
    """

    def test_taxonomy_survives_vocab_extras_roundtrip(self):
        m = _make_radix_model()
        ss = m.symbolicSpace
        # Insert a few META nodes.
        pid_a = ss.insert_percept(b"persist_a")
        pid_b = ss.insert_percept(b"persist_b")
        sid_a = ss.insert_symbol()
        sid_b = ss.insert_symbol()
        mid_a = ss.insert_meta(pid_a, sid_a)
        mid_b = ss.insert_meta(pid_b, sid_b)
        # Dump.
        extras = ss.vocab_extras()
        self.assertIn("taxonomy", extras)
        self.assertIn("taxonomy_parent", extras)
        self.assertIn("meta_pair_to_idx", extras)
        # Fresh SymbolicSpace, load extras.
        from Spaces import SymbolicSpace
        ss2 = SymbolicSpace(
            list(ss.inputShape), list(ss.spaceShape), list(ss.outputShape))
        # Grow ss2's codebook to match capacity (needed if grow_to had to
        # extend the original).
        cb2 = ss2.subspace.what
        cb1 = ss.subspace.what
        if cb2.nVectors < cb1.nVectors:
            cb2.grow_to(int(cb1.nVectors))
        # Restore.
        ss2.load_vocab_extras(extras)
        # taxonomy / taxonomy_parent / meta_pair_to_idx must match.
        self.assertEqual(ss2.taxonomy_children(mid_a),
                         ss.taxonomy_children(mid_a))
        self.assertEqual(ss2.taxonomy_children(mid_b),
                         ss.taxonomy_children(mid_b))
        self.assertEqual(ss2.taxonomy_parent(pid_a), mid_a)
        self.assertEqual(ss2.taxonomy_parent(sid_a), mid_a)
        self.assertEqual(ss2.taxonomy_parent(pid_b), mid_b)
        self.assertEqual(ss2.taxonomy_parent(sid_b), mid_b)
        # Deep equality on the meta-pair lookup: catches any
        # serialization bug (e.g., lost negative-number parse on
        # the stringified tuple keys) that would otherwise leave
        # the dict shaped right but mis-keyed.
        self.assertEqual(ss2.meta_pair_to_idx, ss.meta_pair_to_idx)
        # Re-insertion proves the cache was restored correctly:
        # insert_meta on an existing pair must return the SAME
        # meta idx via the idempotency hit, not allocate a new
        # SS row. Pass a fresh fused_vec so the code path doesn't
        # need a wired perceptualSpace_ref on ss2.
        fresh_vec = torch.zeros(int(ss2.nDim))
        re_mid_a = ss2.insert_meta(pid_a, sid_a, fused_vec=fresh_vec)
        self.assertEqual(re_mid_a, mid_a,
                         f"Re-inserting (pid_a, sid_a) after reload "
                         f"returned {re_mid_a}, expected cached "
                         f"{mid_a}; meta_pair_to_idx was not "
                         f"restored correctly.")


class TestReverseDecodeGuards(unittest.TestCase):
    """Stage 8 code-quality guards on ``_reverse_decode_one``:

      * NaN/Inf input must raise (numerical divergence must surface,
        not be silently masked by argmin returning row 0).
      * Empty SS codebook must return ``b""`` gracefully (not crash
        argmin on a [0, D] tensor).
    """

    def test_reverse_decode_raises_on_nan_input(self):
        m = _make_radix_model()
        ss = m.symbolicSpace
        # Ensure W has at least one row so the NaN check is reached
        # before the empty-codebook short-circuit.
        _pid = ss.insert_percept(b"x")
        _sid = ss.insert_symbol()
        _mid = ss.insert_meta(_pid, _sid)
        D = int(ss.nDim)
        vec = torch.full((D,), float("nan"))
        with self.assertRaises(RuntimeError) as ctx:
            m._reverse_decode_one(vec)
        self.assertIn("NaN/Inf", str(ctx.exception))

    def test_reverse_decode_raises_on_inf_input(self):
        m = _make_radix_model()
        ss = m.symbolicSpace
        _pid = ss.insert_percept(b"y")
        _sid = ss.insert_symbol()
        _mid = ss.insert_meta(_pid, _sid)
        D = int(ss.nDim)
        vec = torch.zeros(D)
        vec[0] = float("inf")
        with self.assertRaises(RuntimeError):
            m._reverse_decode_one(vec)

    def test_reverse_decode_empty_ss_codebook_returns_empty_bytes(self):
        """[0, D]-shape SS codebook -> b"", not a crash.

        Monkeypatches ``cb.getW`` on the SS codebook instance so the
        decode sees a [0, D] W without having to swap the codebook
        out of the (torch.nn.Module) subspace.
        """
        m = _make_radix_model()
        ss = m.symbolicSpace
        cb = ss.subspace.what
        W = cb.getW()
        D = int(W.shape[1])
        empty_W = torch.empty(0, D, device=W.device, dtype=W.dtype)
        saved_getW = cb.getW
        try:
            cb.getW = (lambda _W=empty_W: _W)
            vec = torch.zeros(D, device=W.device, dtype=W.dtype)
            out = m._reverse_decode_one(vec)
            self.assertEqual(out, b"")
        finally:
            # Drop the instance attribute so subsequent calls use the
            # class-level getW again.
            del cb.getW


class TestInsertMetaGuards(unittest.TestCase):
    """Stage 8 code-quality guards on ``SymbolicSpace.insert_meta``:

      * ``ema`` outside ``[0.0, 1.0]`` must raise ``ValueError``.
      * ``fused_vec`` containing NaN/Inf must raise ``RuntimeError``
        on first insert AND on idempotent EMA-update.
    """

    def test_insert_meta_rejects_negative_ema(self):
        m = _make_radix_model()
        ss = m.symbolicSpace
        pid = ss.insert_percept(b"neg_ema")
        sid = ss.insert_symbol()
        # First insert with valid args, so the second call hits the
        # EMA-update branch.
        D = int(ss.nDim)
        v = torch.zeros(D)
        ss.insert_meta(pid, sid, fused_vec=v)
        with self.assertRaises(ValueError) as ctx:
            ss.insert_meta(pid, sid, fused_vec=v, ema=-0.1)
        self.assertIn("ema", str(ctx.exception).lower())

    def test_insert_meta_rejects_ema_above_one(self):
        m = _make_radix_model()
        ss = m.symbolicSpace
        pid = ss.insert_percept(b"big_ema")
        sid = ss.insert_symbol()
        D = int(ss.nDim)
        v = torch.zeros(D)
        ss.insert_meta(pid, sid, fused_vec=v)
        with self.assertRaises(ValueError):
            ss.insert_meta(pid, sid, fused_vec=v, ema=1.5)

    def test_insert_meta_rejects_nan_fused_vec_on_first_insert(self):
        m = _make_radix_model()
        ss = m.symbolicSpace
        pid = ss.insert_percept(b"nan_first")
        sid = ss.insert_symbol()
        D = int(ss.nDim)
        bad = torch.full((D,), float("nan"))
        with self.assertRaises(RuntimeError) as ctx:
            ss.insert_meta(pid, sid, fused_vec=bad)
        self.assertIn("NaN/Inf", str(ctx.exception))

    def test_insert_meta_rejects_inf_fused_vec_on_first_insert(self):
        m = _make_radix_model()
        ss = m.symbolicSpace
        pid = ss.insert_percept(b"inf_first")
        sid = ss.insert_symbol()
        D = int(ss.nDim)
        bad = torch.zeros(D)
        bad[0] = float("inf")
        with self.assertRaises(RuntimeError):
            ss.insert_meta(pid, sid, fused_vec=bad)

    def test_insert_meta_rejects_nan_fused_vec_on_ema_update(self):
        m = _make_radix_model()
        ss = m.symbolicSpace
        pid = ss.insert_percept(b"nan_update")
        sid = ss.insert_symbol()
        D = int(ss.nDim)
        good = torch.zeros(D)
        ss.insert_meta(pid, sid, fused_vec=good)
        bad = torch.full((D,), float("nan"))
        with self.assertRaises(RuntimeError) as ctx:
            ss.insert_meta(pid, sid, fused_vec=bad, ema=0.5)
        self.assertIn("NaN/Inf", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
