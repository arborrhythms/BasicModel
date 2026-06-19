"""Stage 8 + Stage 3+4: Two-codebook split + META taxonomy (positive-int keyed).

Tests the WholeSpace API surface as it exists post-`.where`-keyed
taxonomy refactor (doc/plans/2026-05-28-where-keyed-taxonomy.md):

  * ``insert_percept(canonical_bytes) -> int (position)`` delegates to
    ``PartSpace.percept_store`` and binds a position.
  * ``insert_symbol(init_vec=None) -> int (position)`` allocates a new
    SS.codebook row + position; tagged ``"ws"`` in ``_pos_kind``.
  * ``insert_meta(ps_pos, ws_pos, fused_vec=None) -> int (position)``
    allocates a META node binding ``(ps_pos, ws_pos)``; idempotent on
    the pair (subsequent calls return the same META position and
    EMA-update the stored fused vec). Tagged ``"meta"`` in
    ``_pos_kind``.
  * ``taxonomy``, ``taxonomy_parent`` (dicts), ``taxonomy_children``,
    ``is_meta`` helpers (positive-int keys / values).
  * ``_pos_kind[pos]`` / ``_{ps,ws}_pos_to_row`` lookup tables resolve
    positions back to their underlying codebook rows.
  * Reverse decode: terminal CS state -> SS nearest match -> walk META
    taxonomy -> PS percept_id -> ``inverse_table`` -> canonical bytes.
  * Persistence roundtrip: taxonomy + parent + meta-key map +
    lookup-tables survive ``vocab_extras`` save/load.

Sign convention retired 2026-05-29 (positive-int positions throughout).
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
# Position → row helpers (mirror the implementation lookup tables).
# ---------------------------------------------------------------------------


def _ws_row_from_pos(ws, pos):
    """``pos`` -> SS.codebook row index via ``WholeSpace._ws_pos_to_row``."""
    row = ws._ws_pos_to_row.get(int(pos))
    if row is None:
        raise AssertionError(
            f"position {pos} has no SS-side row binding; expected an "
            f"SS or META position")
    return int(row)


def _ps_row_from_pos(ws, pos):
    """``pos`` -> PerceptStore row index via ``WholeSpace._ps_pos_to_row``."""
    row = ws._ps_pos_to_row.get(int(pos))
    if row is None:
        raise AssertionError(
            f"position {pos} has no PS-side row binding; expected a "
            f"PS position")
    return int(row)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInsertPercept(unittest.TestCase):
    """``insert_percept`` delegates to PerceptStore; returns positive idx."""

    def test_insert_percept_returns_positive_idx_and_inserts(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        # PerceptStore starts empty in a freshly built model
        ps = m.perceptualSpace.percept_store
        self.assertIsNotNone(ps,
                             "MM_xor radix-mode model must have a "
                             "percept_store on PartSpace")
        starting_size = len(ps)
        pos = ws.insert_percept(b"hello")
        self.assertIsInstance(pos, int)
        self.assertGreater(pos, 0,
                           f"insert_percept must return a positive position; "
                           f"got {pos}")
        self.assertEqual(ws._pos_kind.get(pos), "ps",
                         "PS position must be tagged 'ps' in _pos_kind")
        ps_row = _ps_row_from_pos(ws, pos)
        self.assertEqual(ps_row, starting_size,
                         "the new percept id should be the next slot")
        self.assertEqual(ps.bytes_for(ps_row), b"hello",
                         "inverse_table lookup must match insertion")
        # Repeat insert of the same canonical bytes returns the same position.
        again = ws.insert_percept(b"hello")
        self.assertEqual(again, pos,
                         "re-inserting same bytes must return the same position")


class TestInsertSymbol(unittest.TestCase):
    """``insert_symbol`` allocates a fresh SS row + position (positive int)."""

    def test_insert_symbol_returns_positive_position_and_writes_row(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        cb = ws.subspace.what
        W_before = cb.getW().detach().clone()
        # Use a custom init vector so the test can check exact write.
        init_vec = torch.zeros(int(ws.nDim))
        init_vec[0] = 0.7
        init_vec[1] = -0.3
        pos = ws.insert_symbol(init_vec=init_vec)
        self.assertIsInstance(pos, int)
        self.assertGreater(pos, 0,
                           f"insert_symbol must return a positive position; "
                           f"got {pos}")
        self.assertEqual(ws._pos_kind.get(pos), "ws",
                         "fresh SS symbol must be tagged 'ws' in _pos_kind")
        ws_row = _ws_row_from_pos(ws, pos)
        self.assertGreaterEqual(ws_row, 0)
        self.assertLess(ws_row, cb.nVectors)
        W_after = cb.getW().detach()
        # Compare on the codebook's dtype/device.
        expected = init_vec.to(device=W_after.device, dtype=W_after.dtype)
        self.assertTrue(
            torch.allclose(W_after[ws_row], expected, atol=1e-5),
            f"SS row {ws_row} should match init_vec; got "
            f"{W_after[ws_row].tolist()} vs {expected.tolist()}")
        # Other pre-existing rows untouched.
        for r in range(min(W_before.shape[0], W_after.shape[0])):
            if r == ws_row:
                continue
            self.assertTrue(
                torch.allclose(W_before[r], W_after[r], atol=1e-6),
                f"row {r} changed unexpectedly after insert_symbol")

    def test_insert_symbol_default_init_random(self):
        """init_vec=None falls back to a random-ish init."""
        m = _make_radix_model()
        ws = m.wholeSpace
        pos_a = ws.insert_symbol()
        pos_b = ws.insert_symbol()
        self.assertNotEqual(pos_a, pos_b,
                            "Distinct insert_symbol calls must return "
                            "distinct positions")


class TestInsertMeta(unittest.TestCase):
    """``insert_meta(ps_idx, ws_idx, fused_vec)`` allocates a META node."""

    def test_insert_meta_basic_creates_taxonomy_entry(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_pos = ws.insert_percept(b"meta_a")
        ws_pos = ws.insert_symbol()
        meta_pos = ws.insert_meta(ps_pos, ws_pos)
        self.assertGreater(meta_pos, 0,
                           f"insert_meta must return a positive position; "
                           f"got {meta_pos}")
        self.assertEqual(ws._pos_kind.get(meta_pos), "meta",
                         "META position must be tagged 'meta' in _pos_kind")
        # taxonomy_children: meta -> [ps, ws]
        children = ws.taxonomy_children(meta_pos)
        self.assertEqual(set(children), {ps_pos, ws_pos},
                         f"META node children should be {{ps_pos, ws_pos}}; "
                         f"got {children!r}")
        # taxonomy_parent: ps -> meta, ws -> meta
        self.assertEqual(ws.taxonomy_parent(ps_pos), meta_pos)
        self.assertEqual(ws.taxonomy_parent(ws_pos), meta_pos)
        # is_meta predicate
        self.assertTrue(ws.is_meta(meta_pos),
                        "is_meta must return True for a META node")
        self.assertFalse(ws.is_meta(ps_pos),
                         "is_meta must return False for a PS position")
        self.assertFalse(ws.is_meta(ws_pos),
                         "is_meta must return False for a plain SS position")

    def test_insert_meta_default_fused_vec_is_average(self):
        """When fused_vec is None, the META row defaults to the average of
        the PS-row vector and SS-row vector for the bound positions."""
        m = _make_radix_model()
        ws = m.wholeSpace
        ps = m.perceptualSpace.percept_store
        ps_pos = ws.insert_percept(b"meta_b")
        ps_row = _ps_row_from_pos(ws, ps_pos)
        # Pin PS row to a known value so we can predict the average.
        with torch.no_grad():
            ps.codebook.data[ps_row, :].zero_()
            ps.codebook.data[ps_row, 0] = 1.0
        sym_init = torch.zeros(int(ws.nDim))
        sym_init[1] = 1.0
        ws_pos = ws.insert_symbol(init_vec=sym_init)
        ws_row_of_input = _ws_row_from_pos(ws, ws_pos)

        meta_pos = ws.insert_meta(ps_pos, ws_pos)
        meta_row = _ws_row_from_pos(ws, meta_pos)
        W = ws.subspace.what.getW()
        expected = (ps.codebook[ps_row] + W[ws_row_of_input]) / 2.0
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
        ws = m.wholeSpace
        ps_pos = ws.insert_percept(b"meta_c")
        ws_pos = ws.insert_symbol()
        # First insert with an explicit fused_vec.
        fused1 = torch.zeros(int(ws.nDim))
        fused1[0] = 1.0
        meta_pos_1 = ws.insert_meta(ps_pos, ws_pos, fused_vec=fused1)
        meta_row = _ws_row_from_pos(ws, meta_pos_1)
        W = ws.subspace.what.getW()
        first_stored = W[meta_row].detach().clone()
        # Second call with the same (ps_pos, ws_pos) MUST return the same
        # META position AND ema-update the stored vec.
        fused2 = torch.zeros(int(ws.nDim))
        fused2[1] = 1.0
        meta_pos_2 = ws.insert_meta(ps_pos, ws_pos, fused_vec=fused2)
        self.assertEqual(meta_pos_1, meta_pos_2,
                         "insert_meta on the same (ps_pos, ws_pos) pair "
                         "must return the same meta position")
        # New stored vec is somewhere between first and fused2 (EMA).
        second_stored = ws.subspace.what.getW()[meta_row].detach()
        self.assertFalse(
            torch.allclose(second_stored, first_stored, atol=1e-6),
            "Second insert_meta call should EMA-update the stored vec")
        # Children list AND parent map must remain consistent.
        self.assertEqual(set(ws.taxonomy_children(meta_pos_1)),
                         {ps_pos, ws_pos})
        self.assertEqual(ws.taxonomy_parent(ps_pos), meta_pos_1)
        self.assertEqual(ws.taxonomy_parent(ws_pos), meta_pos_1)


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
        ws = m.wholeSpace
        ps = m.perceptualSpace.percept_store
        # Insert three (word, symbol, meta) bundles. Pin the META rows
        # so we can synthesize a terminal CS state that is closest to
        # one of them.
        words = [b"alpha", b"beta", b"gamma"]
        meta_idxs = []
        for w in words:
            pid = ws.insert_percept(w)
            sid = ws.insert_symbol()
            mid = ws.insert_meta(pid, sid)
            meta_idxs.append(mid)
        # Pin the META rows to deterministic, well-separated vectors so
        # nearest-neighbour search is unambiguous.
        cb = ws.subspace.what
        D = int(ws.nDim)
        pinned = {}
        for i, mid in enumerate(meta_idxs):
            row = _ws_row_from_pos(ws,mid)
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
        ws = m.wholeSpace
        ps = m.perceptualSpace.percept_store
        words = [b"hello", b"world", b"foo"]
        meta_idxs = []
        for w in words:
            pid = ws.insert_percept(w)
            sid = ws.insert_symbol()
            mid = ws.insert_meta(pid, sid)
            meta_idxs.append(mid)
        # Pin META rows to deterministic vectors.
        cb = ws.subspace.what
        W = cb.getW()
        D = int(ws.nDim)
        for i, mid in enumerate(meta_idxs):
            row = _ws_row_from_pos(ws,mid)
            vec = torch.zeros(D, device=W.device, dtype=W.dtype)
            vec[i] = 1.0
            with torch.no_grad():
                W.data[row, :].copy_(vec)
        # Synthesize a "reconstructed input" tensor: one batch row, N
        # slots, each slot pinned to one META vector. Shape [B, N, D].
        n_slots = len(words)
        recon = torch.zeros(1, n_slots, D, device=W.device, dtype=W.dtype)
        for i, mid in enumerate(meta_idxs):
            row = _ws_row_from_pos(ws,mid)
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
    """The taxonomy and parent dicts and the (ps,ws) -> meta lookup must
    survive a ``vocab_extras`` save/load cycle.
    """

    def test_taxonomy_survives_vocab_extras_roundtrip(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        # Insert a few META nodes.
        pid_a = ws.insert_percept(b"persist_a")
        pid_b = ws.insert_percept(b"persist_b")
        sid_a = ws.insert_symbol()
        sid_b = ws.insert_symbol()
        mid_a = ws.insert_meta(pid_a, sid_a)
        mid_b = ws.insert_meta(pid_b, sid_b)
        # Dump.
        extras = ws.vocab_extras()
        self.assertIn("taxonomy", extras)
        self.assertIn("taxonomy_parent", extras)
        self.assertIn("meta_pair_to_idx", extras)
        # Fresh WholeSpace, load extras.
        from Spaces import WholeSpace
        ss2 = WholeSpace(
            list(ws.inputShape), list(ws.spaceShape), list(ws.outputShape))
        # Grow ss2's codebook to match capacity (needed if grow_to had to
        # extend the original).
        cb2 = ss2.subspace.what
        cb1 = ws.subspace.what
        if cb2.nVectors < cb1.nVectors:
            cb2.grow_to(int(cb1.nVectors))
        # Restore.
        ss2.load_vocab_extras(extras)
        # taxonomy / taxonomy_parent / meta_pair_to_idx must match.
        self.assertEqual(ss2.taxonomy_children(mid_a),
                         ws.taxonomy_children(mid_a))
        self.assertEqual(ss2.taxonomy_children(mid_b),
                         ws.taxonomy_children(mid_b))
        self.assertEqual(ss2.taxonomy_parent(pid_a), mid_a)
        self.assertEqual(ss2.taxonomy_parent(sid_a), mid_a)
        self.assertEqual(ss2.taxonomy_parent(pid_b), mid_b)
        self.assertEqual(ss2.taxonomy_parent(sid_b), mid_b)
        # Deep equality on the meta-pair lookup: catches any
        # serialization bug (e.g., lost negative-number parse on
        # the stringified tuple keys) that would otherwise leave
        # the dict shaped right but mis-keyed.
        self.assertEqual(ss2.meta_pair_to_idx, ws.meta_pair_to_idx)
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
        ws = m.wholeSpace
        # Ensure W has at least one row so the NaN check is reached
        # before the empty-codebook short-circuit.
        _pid = ws.insert_percept(b"x")
        _sid = ws.insert_symbol()
        _mid = ws.insert_meta(_pid, _sid)
        D = int(ws.nDim)
        vec = torch.full((D,), float("nan"))
        with self.assertRaises(RuntimeError) as ctx:
            m._reverse_decode_one(vec)
        self.assertIn("NaN/Inf", str(ctx.exception))

    def test_reverse_decode_raises_on_inf_input(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        _pid = ws.insert_percept(b"y")
        _sid = ws.insert_symbol()
        _mid = ws.insert_meta(_pid, _sid)
        D = int(ws.nDim)
        vec = torch.zeros(D)
        vec[0] = float("inf")
        with self.assertRaises(RuntimeError):
            m._reverse_decode_one(vec)

    def test_reverse_decode_empty_ws_codebook_returns_empty_bytes(self):
        """[0, D]-shape SS codebook -> b"", not a crash.

        Monkeypatches ``cb.getW`` on the SS codebook instance so the
        decode sees a [0, D] W without having to swap the codebook
        out of the (torch.nn.Module) subspace.
        """
        m = _make_radix_model()
        ws = m.wholeSpace
        cb = ws.subspace.what
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
    """Stage 8 code-quality guards on ``WholeSpace.insert_meta``:

      * ``ema`` outside ``[0.0, 1.0]`` must raise ``ValueError``.
      * ``fused_vec`` containing NaN/Inf must raise ``RuntimeError``
        on first insert AND on idempotent EMA-update.
    """

    def test_insert_meta_rejects_negative_ema(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        pid = ws.insert_percept(b"neg_ema")
        sid = ws.insert_symbol()
        # First insert with valid args, so the second call hits the
        # EMA-update branch.
        D = int(ws.nDim)
        v = torch.zeros(D)
        ws.insert_meta(pid, sid, fused_vec=v)
        with self.assertRaises(ValueError) as ctx:
            ws.insert_meta(pid, sid, fused_vec=v, ema=-0.1)
        self.assertIn("ema", str(ctx.exception).lower())

    def test_insert_meta_rejects_ema_above_one(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        pid = ws.insert_percept(b"big_ema")
        sid = ws.insert_symbol()
        D = int(ws.nDim)
        v = torch.zeros(D)
        ws.insert_meta(pid, sid, fused_vec=v)
        with self.assertRaises(ValueError):
            ws.insert_meta(pid, sid, fused_vec=v, ema=1.5)

    def test_insert_meta_rejects_nan_fused_vec_on_first_insert(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        pid = ws.insert_percept(b"nan_first")
        sid = ws.insert_symbol()
        D = int(ws.nDim)
        bad = torch.full((D,), float("nan"))
        with self.assertRaises(RuntimeError) as ctx:
            ws.insert_meta(pid, sid, fused_vec=bad)
        self.assertIn("NaN/Inf", str(ctx.exception))

    def test_insert_meta_rejects_inf_fused_vec_on_first_insert(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        pid = ws.insert_percept(b"inf_first")
        sid = ws.insert_symbol()
        D = int(ws.nDim)
        bad = torch.zeros(D)
        bad[0] = float("inf")
        with self.assertRaises(RuntimeError):
            ws.insert_meta(pid, sid, fused_vec=bad)

    def test_insert_meta_rejects_nan_fused_vec_on_ema_update(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        pid = ws.insert_percept(b"nan_update")
        sid = ws.insert_symbol()
        D = int(ws.nDim)
        good = torch.zeros(D)
        ws.insert_meta(pid, sid, fused_vec=good)
        bad = torch.full((D,), float("nan"))
        with self.assertRaises(RuntimeError) as ctx:
            ws.insert_meta(pid, sid, fused_vec=bad, ema=0.5)
        self.assertIn("NaN/Inf", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
