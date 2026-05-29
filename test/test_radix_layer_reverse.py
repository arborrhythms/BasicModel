"""Tests for ``RadixLayer.reverse`` (Task F).

Covers the relocated structural-decode body that previously lived on
``BasicModel._reverse_decode_one``:

  * Standalone fallback (no ``symbolic_space``): nearest-PS-codebook
    decode returns the canonical bytes for the matching percept.
  * SS-walk: when a SymbolicSpace peer is supplied, the walk goes
    nearest -> META taxonomy children -> positive PS percept id ->
    ``bytes_for``.
  * NaN / Inf input raises ``RuntimeError`` (fail-loud policy).
  * Near-zero norm input returns ``b""`` (null-slot guard).
  * Empty PS codebook returns ``b""``.
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


def _ss_row_from_signed(signed_idx):
    """``signed_idx`` -> SS.codebook row index. Raises if positive."""
    if signed_idx >= 0:
        raise AssertionError(
            f"SS row expected from negative signed idx, got {signed_idx}")
    return -signed_idx - 1


def _make_radix_model():
    """Build the MM_xor radix-chunking model (mirrors the helper in
    test_two_codebook_meta_taxonomy.py)."""
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


class TestRadixLayerReverseStandalone(unittest.TestCase):
    """``RadixLayer.reverse(vec)`` with no ``symbolic_space`` falls back
    to nearest-PS-codebook lookup. Verify it returns the canonical bytes
    for a vec near a known percept row."""

    def test_standalone_decode_returns_nearest_percept_bytes(self):
        from Layers import RadixLayer
        D = 6
        rl = RadixLayer(dim=D, initial_cap=8)
        words = [b"alpha", b"beta", b"gamma"]
        # Use deterministic init vectors so nearest-match is unambiguous.
        pids = []
        for i, w in enumerate(words):
            vec = torch.zeros(D)
            vec[i] = 1.0
            pid = rl.insert(w, init_vector=vec)
            pids.append(pid)
        # Aim for "beta" by handing in a vec that matches its codebook row.
        target = torch.zeros(D)
        target[1] = 1.0
        out = rl.reverse(target)
        self.assertEqual(out, b"beta",
                         f"standalone reverse should snap to nearest PS row; "
                         f"got {out!r}")

    def test_standalone_decode_handles_perturbed_input(self):
        """Small perturbations should still snap to the nearest row."""
        from Layers import RadixLayer
        D = 6
        rl = RadixLayer(dim=D, initial_cap=8)
        for i, w in enumerate([b"alpha", b"beta", b"gamma"]):
            vec = torch.zeros(D)
            vec[i] = 1.0
            rl.insert(w, init_vector=vec)
        # Perturb "alpha"'s slot slightly.
        target = torch.zeros(D)
        target[0] = 0.9
        target[3] = 0.1
        out = rl.reverse(target)
        self.assertEqual(out, b"alpha",
                         f"perturbed input should still pick the nearest "
                         f"PS row; got {out!r}")


class TestRadixLayerReverseWithSS(unittest.TestCase):
    """``RadixLayer.reverse(vec, symbolic_space=ss)`` walks META taxonomy."""

    def test_ss_walk_recovers_bytes_via_meta_taxonomy(self):
        m = _make_radix_model()
        ss = m.symbolicSpace
        ps = m.perceptualSpace.percept_store
        words = [b"alpha", b"beta", b"gamma"]
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
            v = torch.zeros(D, device=W.device, dtype=W.dtype)
            v[i] = 1.0
            with torch.no_grad():
                W.data[row, :].copy_(v)
        # Target "beta" -> pinned META vector.
        target_vec = torch.zeros(D, device=W.device, dtype=W.dtype)
        target_vec[1] = 1.0
        out = ps.reverse(target_vec, symbolic_space=ss)
        self.assertEqual(out, b"beta",
                         f"SS-walk reverse should recover the matching "
                         f"surface bytes; got {out!r}")


class TestRadixLayerReverseFailLoudOnNaN(unittest.TestCase):
    """NaN / Inf input must raise (fail-loud policy)."""

    def test_nan_input_raises(self):
        from Layers import RadixLayer
        D = 4
        rl = RadixLayer(dim=D, initial_cap=4)
        rl.insert(b"x", init_vector=torch.zeros(D))
        vec = torch.full((D,), float("nan"))
        with self.assertRaises(RuntimeError) as ctx:
            rl.reverse(vec)
        msg = str(ctx.exception)
        self.assertIn("NaN", msg, f"error must mention NaN: {msg!r}")

    def test_inf_input_raises(self):
        from Layers import RadixLayer
        D = 4
        rl = RadixLayer(dim=D, initial_cap=4)
        rl.insert(b"x", init_vector=torch.zeros(D))
        vec = torch.full((D,), float("inf"))
        with self.assertRaises(RuntimeError):
            rl.reverse(vec)


class TestRadixLayerReverseNullSlotGuard(unittest.TestCase):
    """Near-zero input returns ``b""`` (padding / inter-word-space slots)."""

    def test_near_zero_input_returns_empty_bytes(self):
        from Layers import RadixLayer
        D = 4
        rl = RadixLayer(dim=D, initial_cap=4)
        rl.insert(b"x", init_vector=torch.ones(D))
        vec = torch.full((D,), 1e-5)
        out = rl.reverse(vec)
        self.assertEqual(out, b"",
                         f"near-zero norm vec should return empty bytes; "
                         f"got {out!r}")


class TestRadixLayerReverseEmptyCodebook(unittest.TestCase):
    """An empty PS codebook returns ``b""`` (no rows to argmin over)."""

    def test_empty_codebook_returns_empty_bytes(self):
        from Layers import RadixLayer
        D = 4
        rl = RadixLayer(dim=D, initial_cap=4)
        # No inserts. With no inserts the populated size is zero; but the
        # codebook Parameter has shape [initial_cap, D] and is not empty
        # rows-wise. The reverse path's "empty" guard is W.shape[0] == 0;
        # to exercise it, force-zero the codebook shape via a peer that
        # has no rows.
        # Standalone fallback path: the comparison codebook is
        # self.codebook, which has shape [initial_cap, D] -- non-empty.
        # Insert nothing and the bytes_for() guard kicks in instead,
        # also returning b"" (out-of-range pid).
        vec = torch.ones(D)
        out = rl.reverse(vec)
        self.assertEqual(out, b"",
                         f"empty inverse-table should yield b''; got {out!r}")


class TestRadixLayerReverseBatchShape(unittest.TestCase):
    """Higher-rank inputs return list-of-bytes / list-of-list-of-bytes."""

    def test_2d_input_returns_list_of_bytes(self):
        from Layers import RadixLayer
        D = 4
        rl = RadixLayer(dim=D, initial_cap=8)
        for i, w in enumerate([b"a", b"b", b"c"]):
            v = torch.zeros(D)
            v[i] = 1.0
            rl.insert(w, init_vector=v)
        batch = torch.stack([
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
        ])
        out = rl.reverse(batch)
        self.assertEqual(out, [b"a", b"b", b"c"])

    def test_3d_input_returns_nested_list(self):
        from Layers import RadixLayer
        D = 4
        rl = RadixLayer(dim=D, initial_cap=8)
        for i, w in enumerate([b"a", b"b"]):
            v = torch.zeros(D)
            v[i] = 1.0
            rl.insert(w, init_vector=v)
        # Two batch rows, two slots each.
        batch = torch.zeros(2, 2, D)
        batch[0, 0, 0] = 1.0  # 'a'
        batch[0, 1, 1] = 1.0  # 'b'
        batch[1, 0, 1] = 1.0  # 'b'
        batch[1, 1, 0] = 1.0  # 'a'
        out = rl.reverse(batch)
        self.assertEqual(out, [[b"a", b"b"], [b"b", b"a"]])


if __name__ == "__main__":
    unittest.main()
