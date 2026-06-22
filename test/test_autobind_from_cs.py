"""Task G regression: auto-META binding fires from ConceptualSpace, not
PartSpace.

After Task G the cross-space PS<->SS allocation that previously lived on
``PartSpace._maybe_autobind_meta`` (and called from
``_embed_radix``) moved to ``ConceptualSpace._maybe_autobind_meta``
(called from ``cs.forward`` at stage 0). Verify:

  * PartSpace no longer holds a back-ref to WholeSpace
    (``wholeSpace_ref`` attribute absent).
  * PartSpace no longer defines ``_maybe_autobind_meta``.
  * ConceptualSpace defines ``_maybe_autobind_meta``.
  * Each per-stage ConceptualSpace has a ``perceptualSpace_ref`` and a
    ``terminalSymbolSpace_ref`` after BasicModel construction.
  * The autobind on CS at stage 0 fires when invoked with a synthetic
    pid grid + matching vec tensor.
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
    """Build the MM_xor radix-chunking model."""
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


class TestAutobindMovedFromPSToCS(unittest.TestCase):
    """``_maybe_autobind_meta`` lives on ConceptualSpace, not
    PartSpace."""

    def test_perceptualspace_has_no_symbolicspace_ref(self):
        m = _make_radix_model()
        ps = m.perceptualSpace
        self.assertFalse(
            hasattr(ps, 'wholeSpace_ref'),
            "PartSpace must NOT hold a back-ref to WholeSpace "
            "after Task G; the auto-bind cross-space hop moved to "
            "ConceptualSpace.",
        )

    def test_perceptualspace_does_not_define_autobind(self):
        from Spaces import PartSpace
        self.assertFalse(
            hasattr(PartSpace, '_maybe_autobind_meta'),
            "PartSpace must NOT define _maybe_autobind_meta after "
            "Task G; the body moved to ConceptualSpace.",
        )

    def test_conceptualspace_defines_autobind(self):
        from Spaces import ConceptualSpace
        self.assertTrue(
            hasattr(ConceptualSpace, '_maybe_autobind_meta'),
            "ConceptualSpace must define _maybe_autobind_meta after "
            "Task G.",
        )

    def test_each_cs_has_perceptual_and_terminal_ws_refs(self):
        m = _make_radix_model()
        terminal_ss = m.wholeSpace
        ps = m.perceptualSpace
        for i, cs in enumerate(m.conceptualSpaces):
            self.assertTrue(
                hasattr(cs, 'perceptualSpace_ref'),
                f"ConceptualSpace[{i}] missing perceptualSpace_ref")
            self.assertIs(cs.perceptualSpace_ref, ps,
                          f"ConceptualSpace[{i}].perceptualSpace_ref "
                          f"must point at the canonical PartSpace")
            self.assertTrue(
                hasattr(cs, 'terminalSymbolSpace_ref'),
                f"ConceptualSpace[{i}] missing "
                f"terminalSymbolSpace_ref")
            self.assertIs(
                cs.terminalSymbolSpace_ref, terminal_ss,
                f"ConceptualSpace[{i}].terminalSymbolSpace_ref must "
                f"point at the terminal WholeSpace "
                f"(wholeSpaces[-1])",
            )


class TestAutobindFiresFromCS(unittest.TestCase):
    """Calling ``cs._maybe_autobind_meta`` directly creates an SS row +
    META taxonomy entry for the supplied percept ids."""

    def test_autobind_creates_meta_taxonomy_entry(self):
        m = _make_radix_model()
        ps = m.perceptualSpace
        ps_store = ps.percept_store
        ws = m.wholeSpace
        # Use the FIRST conceptualSpace stage (stage_idx == 0) where the
        # production autobind fires. NB: since the 2026-06-03 fullgraph
        # refactor it fires from cs.Reset (sentence boundary), not mid-
        # forward; this test calls _maybe_autobind_meta directly, so it is
        # unaffected by that relocation.
        cs = m.conceptualSpaces[0]
        # Insert a fresh percept on the PS store and grab its vector.
        D = int(ps_store.dim)
        seed = torch.zeros(D)
        seed[0] = 0.7
        pid = ps_store.insert(b"taskg_unique", init_vector=seed)
        # Synthesize a [B=1, N=1] pid grid + [B=1, N=1, D] vec tensor.
        pid_grid = torch.tensor([[pid]], dtype=torch.long)
        vec_event = seed.view(1, 1, -1)
        # Snapshot the SS taxonomy size before the call.
        size_before = len(ws.taxonomy)
        cs._maybe_autobind_meta(pid_grid, vec_event)
        size_after = len(ws.taxonomy)
        self.assertGreater(
            size_after, size_before,
            "autobind from CS should have grown the SS META taxonomy",
        )
        # The autobind set on CS should track the pid.
        bound = getattr(cs, '_autobound_percept_ids', None)
        self.assertIsNotNone(bound)
        self.assertIn(pid, bound,
                      "autobind set on CS should track the newly-bound "
                      "percept id")

    def test_autobind_skips_negative_pids(self):
        m = _make_radix_model()
        cs = m.conceptualSpaces[0]
        # All-sentinel pid grid should be a no-op (no SS rows added).
        D = int(m.perceptualSpace.percept_store.dim)
        pid_grid = torch.tensor([[-1, -1]], dtype=torch.long)
        vec_event = torch.zeros(1, 2, D)
        ws = m.wholeSpace
        size_before = len(ws.taxonomy)
        cs._maybe_autobind_meta(pid_grid, vec_event)
        size_after = len(ws.taxonomy)
        self.assertEqual(size_before, size_after,
                         "all-sentinel pid grid should be a no-op")


if __name__ == "__main__":
    unittest.main()
