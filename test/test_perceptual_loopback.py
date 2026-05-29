"""PerceptualSpace conceptual loopback (C→P) — explicit two-input
recurrent-cell contract (post-2026-05 reconciliation).

The cross-space combination that used to live in
``PerceptualSpace._sourced_input`` (reading ``conceptualSpace_ref``)
is now an explicit ``forward`` argument supplied by the recurrent cell:

  * ``PerceptualSpace.forward(IS_subspace, CS_subspaceForPS=None)``
  * ``ConceptualSpace.forward(PS_subspace, SS_subspace=None)`` exposing
    ``_subspaceForPS`` / ``_subspaceForSS`` for the next pass
  * ``SymbolicSpace.forward(CS_subspaceForSS)``

The ``_sourced_input`` / ``_read_event`` / ``_get_active_input_sibling``
helpers and the ``conceptualSpace_ref`` / ``symbolicSpace_ref`` /
ConceptualSpace.``perceptualSpace_ref`` forward-input refs are deleted.
``SymbolicSpace.perceptualSpace_ref`` is KEPT (structural lexicon
ownership) and is still covered by ``TestLexiconOnSymbolicSpace`` below.

Tests in this file:
  * The new explicit ``forward`` arities; the optional second arg
    defaults to ``None`` (standalone single-arg callers still work).
  * The deleted helpers / forward-input refs are gone.
  * A full model forward runs the recurrent cell and the terminal
    ConceptualSpace exposes ``_subspaceForPS`` / ``_subspaceForSS``.
  * Cold start (``CS_subspaceForPS`` None / empty) degrades to the
    primary ``pi_input`` path.
  * ``SubsymbolicSpace`` and its mode selector remain retired.
  * Lexicon ownership on SymbolicSpace (structural ref) unchanged.
"""
import os
import sys
import unittest

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_CONFIG = os.path.join(_PROJECT, "data", "MM_xor_loopback.xml")
_DEFAULTS = os.path.join(_PROJECT, "data", "model.xml")


def _fresh_model():
    """Build a fresh BasicModel from MM_xor_loopback.xml with XOR data."""
    import Models
    import Language
    from util import init_config
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    m, cfg = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    return m


class TestForwardArityContract(unittest.TestCase):
    """Explicit two-input forward signatures replace _sourced_input/refs."""

    # Post-2026-05-21 SentenceState dissolution: the per-sentence
    # ``work`` carrier was retired. Spaces forwards now take only their
    # data SubSpace argument(s); grammar / serial-processing state lives
    # on ``subspace.wordSubSpace`` (the back-reference threaded by
    # ``copy_context``). The arity guards below assert the new
    # carrier-free signatures.

    def test_perceptual_forward_arity(self):
        # Post-Stage-1.A substrate refactor (doc/plans/
        # 2026-05-26-two-loop-pi-sigma-substrate.md): PS.forward is
        # single-arg. Body composes ``pi(x) + sigma(x)`` on the same
        # materialized input; the legacy ``CS_subspaceForPS`` second
        # arg is gone (CS feedback no longer enters PS at this level —
        # it re-enters via chart / signal-router over STM in later
        # stages).
        import inspect
        import Spaces
        sig = inspect.signature(Spaces.PerceptualSpace.forward)
        params = [n for n in sig.parameters if n != "self"]
        self.assertEqual(len(params), 1,
                         f"PerceptualSpace.forward must be "
                         f"(x_subspace); got {params}")
        self.assertNotIn("work", sig.parameters,
                         "PerceptualSpace.forward must not carry the "
                         "retired Phase-1 'work' carrier.")

    def test_conceptual_forward_arity(self):
        import inspect
        import Spaces
        sig = inspect.signature(Spaces.ConceptualSpace.forward)
        params = [n for n in sig.parameters if n != "self"]
        self.assertEqual(len(params), 2,
                         f"ConceptualSpace.forward must be "
                         f"(PS_subspace, SS_subspace=None); got {params}")
        self.assertIsNot(sig.parameters[params[1]].default,
                         inspect.Parameter.empty,
                         "SS_subspace must default to None.")
        self.assertNotIn("work", sig.parameters,
                         "ConceptualSpace.forward must not carry the "
                         "retired Phase-1 'work' carrier.")

    def test_symbolic_forward_single_arg(self):
        import inspect
        import Spaces
        sig = inspect.signature(Spaces.SymbolicSpace.forward)
        params = [n for n in sig.parameters if n != "self"]
        self.assertEqual(len(params), 1,
                         f"SymbolicSpace.forward must be "
                         f"(CS_subspaceForSS); got {params}")
        self.assertNotIn("work", sig.parameters,
                         "SymbolicSpace.forward must not carry the "
                         "retired Phase-1 'work' carrier.")

    def test_sourced_input_and_read_event_removed(self):
        import Spaces
        for cls in (Spaces.PerceptualSpace, Spaces.ConceptualSpace):
            self.assertFalse(
                hasattr(cls, '_sourced_input'),
                f"{cls.__name__}._sourced_input must be folded into "
                f"forward and removed.")
            self.assertFalse(
                hasattr(cls, '_read_event'),
                f"{cls.__name__}._read_event must be removed.")
        self.assertFalse(
            hasattr(Spaces.ConceptualSpace, '_get_active_input_sibling'),
            "ConceptualSpace._get_active_input_sibling must be removed.")

    def test_forward_input_refs_removed(self):
        """conceptualSpace_ref / symbolicSpace_ref / ConceptualSpace's
        perceptualSpace_ref are no longer initialised in __init__."""
        import inspect
        import Spaces
        p_src = inspect.getsource(Spaces.PerceptualSpace.__init__)
        self.assertNotIn("self.conceptualSpace_ref = None", p_src,
                         "PerceptualSpace must not init conceptualSpace_ref "
                         "(C→P feedback is now an explicit forward arg).")
        c_src = inspect.getsource(Spaces.ConceptualSpace.__init__)
        self.assertNotIn("self.symbolicSpace_ref = None", c_src)
        self.assertNotIn("self.perceptualSpace_ref = None", c_src,
                         "ConceptualSpace must not init the forward-input "
                         "refs (PS/SS are explicit forward args).")


class TestSubsymbolicSpaceRetired(unittest.TestCase):
    """The standalone SubsymbolicSpace class and its mode selector are gone."""

    def test_subsymbolic_space_class_removed(self):
        import Spaces
        self.assertFalse(hasattr(Spaces, 'SubsymbolicSpace'),
                         "SubsymbolicSpace should be removed; "
                         "PerceptualSpace is the subsymbolic substrate "
                         "via the explicit C→P forward arg.")

    def test_subsymbolicSpace_ref_not_on_conceptual(self):
        import Spaces
        import inspect
        src = inspect.getsource(Spaces.ConceptualSpace.__init__)
        self.assertNotIn("self.subsymbolicSpace_ref = None", src,
                         "ConceptualSpace must not retain a "
                         "subsymbolicSpace_ref attribute.")

    def test_subsymbolic_widen_dim_param_removed(self):
        import Spaces
        import inspect
        sig = inspect.signature(Spaces.ConceptualSpace.__init__)
        self.assertNotIn("subsymbolic_widen_dim", sig.parameters,
                         "ConceptualSpace.__init__ must not accept "
                         "subsymbolic_widen_dim.")


class TestRecurrentCellAndOutputViews(unittest.TestCase):
    """A full forward runs the recurrent cell; terminal CS exposes the
    two consumer views; cold start degrades to the primary path."""

    @classmethod
    def setUpClass(cls):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            cls.model = _fresh_model()
        cls.percep = cls.model.perceptualSpace

    def _one_input(self):
        m = self.model
        loader = m.inputSpace.data.data_loader(split="train", num_streams=1)
        inp_items, _ = next(iter(loader))
        return m.inputSpace.prepInput(inp_items)

    def test_full_forward_runs_recurrent_cell(self):
        import warnings
        m = self.model
        m.eval()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                out = m.forward(self._one_input())
        self.assertEqual(len(out), 4,
                         "_forward_per_stage must return its 4-tuple.")

    def test_terminal_conceptual_exposes_views(self):
        """Post-SentenceState-dissolution (2026-05-21): the cross-stage
        CS→PS and CS→SS feedback is exposed directly as
        ``ConceptualSpace._subspaceForPS`` / ``._subspaceForSS`` (the
        persistent SubSpace objects that ``ConceptualSpace.forward``
        mutates in place). The reverse path generates its own
        reconstructed estimates and never reads these forward caches.
        """
        import warnings
        m = self.model
        m.eval()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                m.forward(self._one_input())
        cs = m.conceptualSpaces[-1]
        self.assertTrue(hasattr(cs, '_subspaceForPS'),
                        "ConceptualSpace._subspaceForPS must exist after "
                        "forward.")
        self.assertTrue(hasattr(cs, '_subspaceForSS'),
                        "ConceptualSpace._subspaceForSS must exist after "
                        "forward.")

    def test_forward_produces_percept_event(self):
        """Post-Stage-1.A: PS.forward is single-arg
        (``pi(x) + sigma(x)`` on the same input). Two successive calls
        with the same upstream IS subspace produce a non-None percept
        event with the same shape -- the in-call composition is
        deterministic, no CS-feedback branching."""
        import warnings
        m = self.model
        m.eval()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                in_sub = m.inputSpace.forward(self._one_input())
                a = self.percep.forward(in_sub)
                ev_a = a.materialize() if a is not None else None
                b = self.percep.forward(in_sub)
                ev_b = b.materialize() if b is not None else None
        self.assertIsNotNone(ev_a,
                             "forward(IS) must produce a percept event.")
        self.assertIsNotNone(ev_b,
                             "second forward(IS) must produce a percept "
                             "event.")
        self.assertEqual(tuple(ev_a.shape), tuple(ev_b.shape),
                         "Two successive forward(IS) calls must produce "
                         "events of the same shape.")


class TestLexiconOnSymbolicSpace(unittest.TestCase):
    """SymbolicSpace is the logical owner of the orthographic Lexicon.

    Post-lexicon-migration: ``S.vocabulary`` returns the Embedding, and
    the orthographic-API methods delegate to the physical Embedding via
    the KEPT structural ``perceptualSpace_ref``. This ref is NOT a
    forward-input plumbing ref and is preserved by the reconciliation.
    """

    @classmethod
    def setUpClass(cls):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            cls.model = _fresh_model()

    def test_symbolic_perceptualSpace_ref_wired(self):
        sym = self.model.symbolicSpace
        self.assertIsNotNone(sym.perceptualSpace_ref,
                             "Model.__init__ must keep wiring "
                             "symbolicSpace.perceptualSpace_ref (structural "
                             "lexicon ownership, NOT forward input).")
        import Spaces
        self.assertIsInstance(sym.perceptualSpace_ref,
                              Spaces.PerceptualSpace)

    def test_symbolic_vocabulary_returns_lexicon(self):
        sym = self.model.symbolicSpace
        self.assertIsNotNone(sym.vocabulary,
                             "S.vocabulary must return the Embedding from "
                             "P (text mode) or fall back to S's own .what "
                             "(numeric / non-text mode).")

    def test_symbolic_lexicon_methods_exist(self):
        sym = self.model.symbolicSpace
        for name in (
            'train_embeddings', 'sbow_loss', '_snapshot_embeddings',
            'set_embedding_sigma', 'reconstruct_data',
            'reconstruct_to_buffer', 'get_recovered_word',
        ):
            self.assertTrue(hasattr(sym, name),
                            f"SymbolicSpace must expose {name!r} after "
                            f"the lexicon-API migration.")


class TestSubsymbolicSymbolicSplit(unittest.TestCase):
    """Post-2026-05-12 split: grammar's canonical home is S. The per-tier
    SyntacticLayer dispatchers at P and C are retained as backward-compat
    no-ops for grammars that omit P/C rules.
    """

    @classmethod
    def setUpClass(cls):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            cls.model = _fresh_model()

    def test_s_has_syntactic_layer(self):
        for s in self.model.symbolicSpaces:
            layer = getattr(s, 'syntacticLayer', None)
            self.assertIsNotNone(
                layer,
                f"SymbolicSpace must retain its SyntacticLayer "
                f"(the grammar's canonical dispatch host); missing on {s}")

    def test_no_p_tier_rules_in_grammar(self):
        import Language
        p_rules = [r for r in Language.TheGrammar.rules
                   if getattr(r, 'tier', None) == 'P']
        self.assertEqual(len(p_rules), 0,
                         f"Production grammar must list no P-tier "
                         f"rules; found {p_rules}")

    def test_no_c_tier_rules_in_grammar(self):
        import Language
        c_rules = [r for r in Language.TheGrammar.rules
                   if getattr(r, 'tier', None) == 'C']
        self.assertEqual(len(c_rules), 0,
                         f"Production grammar must list no C-tier "
                         f"rules; found {c_rules}")


class TestPerceptStoreIntegration(unittest.TestCase):
    """Stage 7 (doc/plans/2026-05-27-perceptstore-meta-taxonomy-
    reentrancy.md): PerceptualSpace exposes ``self.percept_store`` when
    ``<chunking>radix</chunking>`` is selected; legacy chunking modes
    (``lexicon|bpe|mphf|none``) keep their existing ``ChunkLayer`` /
    Embedding-based wiring with ``self.percept_store is None``.
    """

    def test_radix_config_builds_percept_store(self):
        """MM_xor.xml selects ``<chunking>radix</chunking>`` post-Stage-7
        and the constructed PerceptualSpace must expose a PerceptStore.
        """
        import warnings
        import Models
        import Language
        from util import init_config
        cfg_path = os.path.join(_PROJECT, "data", "MM_xor.xml")
        init_config(path=cfg_path, defaults_path=_DEFAULTS)
        Language.TheGrammar._configured = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m, _ = Models.BasicModel.from_config(cfg_path)
        ps_space = m.perceptualSpace
        self.assertEqual(ps_space.chunking_mode, "radix",
                         "MM_xor.xml should now use radix chunking")
        self.assertIsNotNone(ps_space.percept_store,
                             "radix-mode PerceptualSpace must expose "
                             "self.percept_store")
        # Importing PerceptStore directly verifies the class identity.
        from PerceptStore import PerceptStore
        self.assertIsInstance(ps_space.percept_store, PerceptStore)
        # vocabulary property returns the percept store in radix mode.
        self.assertIs(ps_space.vocabulary, ps_space.percept_store)

    def test_legacy_lexicon_mode_keeps_chunklayer_path(self):
        """The MM_xor_loopback config doesn't set chunking; the default
        path remains ``lexicon`` for backward compatibility, with
        ``percept_store`` set to ``None``."""
        m = _fresh_model()
        ps_space = m.perceptualSpace
        # MM_xor_loopback doesn't set chunking, so the default lexicon
        # path stays active and percept_store should be None.
        self.assertEqual(ps_space.chunking_mode, "lexicon")
        self.assertIsNone(ps_space.percept_store,
                          "lexicon mode must leave percept_store unset "
                          "so the legacy ChunkLayer / Embedding path "
                          "stays authoritative")
        # vocabulary property falls back to subspace.vocabulary in
        # non-radix mode (the base Space behaviour).
        self.assertIs(ps_space.vocabulary, ps_space.subspace.vocabulary)

    def test_radix_percept_store_roundtrips_inserted_words(self):
        """Stage 7 acceptance: 'words are inserted into the PerceptStore;
        inverse table reproduces the surface bytes exactly'.

        Post-review fix (Issue 1): _embed_radix now routes through
        ``ps.lookup_with_id`` so promotion is gated by
        ``<chunkPromotionThreshold>`` / ``<chunkPromotionMinLength>``.
        Words become permanent percepts only after the threshold is
        reached. We force promotion by feeding the same batch
        ``promotion_threshold`` times, then verify the inverse table
        roundtrip on the resulting permanent percept IDs.

        We bypass the full model forward (which would also exercise CS
        / SS / OS layers and isn't required for the Stage 7 contract).
        Instead we directly drive ``_embed_radix`` with a hand-built
        upstream SubSpace whose ``_host_tokens`` carry a small sentence.
        """
        import warnings
        import Models
        import Language
        from util import init_config
        cfg_path = os.path.join(_PROJECT, "data", "MM_xor.xml")
        init_config(path=cfg_path, defaults_path=_DEFAULTS)
        Language.TheGrammar._configured = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m, _ = Models.BasicModel.from_config(cfg_path)
            Models.TheData.load("xor")
        ps_space = m.perceptualSpace
        ps = ps_space.percept_store
        self.assertEqual(len(ps), 0,
                         "PerceptStore should start empty")
        # Build a fake upstream SubSpace with host tokens stamped on.
        # The _embed_radix path uses _host_tokens when present and
        # only touches what_buf to decide non-emptiness, so we just
        # need a non-None what_buf and a list-of-lists host_tokens.
        words_per_row = [
            ["hello", "world"],
            ["the", "quick", "brown"],
        ]
        # All words are >= promotion_min_length=2, so each one will
        # promote after ``promotion_threshold`` (default 4) hits.
        what_buf = torch.zeros(2, 8, 1, dtype=torch.long)
        upstream = ps_space.subspace
        upstream._host_tokens = words_per_row
        upstream.set_what(what_buf)
        upstream.batch = 2
        # Drive the radix-mode embedding path ``promotion_threshold``
        # times so every distinct word promotes into the store.
        threshold = ps.promotion_threshold
        for _ in range(threshold):
            ps_space._embed_radix(upstream)
        # Every distinct word should now have landed in the store.
        seen_words = set()
        for row in words_per_row:
            for w in row:
                seen_words.add(w)
        for w in seen_words:
            pid = ps.get_id(w.encode("utf-8"))
            self.assertIsNotNone(
                pid,
                f"word {w!r} should have been promoted into the "
                f"PerceptStore by _embed_radix after {threshold} hits")
        # Roundtrip every percept_id -> bytes -> percept_id.
        for pid in range(len(ps)):
            recovered = ps.bytes_for(pid)
            self.assertEqual(ps.get_id(recovered), pid,
                             f"inverse table roundtrip failed for "
                             f"percept_id {pid}: bytes={recovered!r}")

    def test_embed_radix_respects_promotion_threshold(self):
        """Issue 1 regression guard: ``_embed_radix`` must route
        through ``lookup_with_id`` so the
        ``<chunkPromotionThreshold>`` / ``<chunkPromotionMinLength>``
        knobs gate permanent installation.

        Contract:
          * A word at or above ``promotion_min_length`` does NOT get a
            permanent percept_id until ``promotion_threshold`` calls
            have happened.
          * Exactly at ``promotion_threshold`` hits the word is in the
            store.
          * A word below ``promotion_min_length`` NEVER promotes
            regardless of how many times we see it.
        """
        import warnings
        import Models
        import Language
        from util import init_config
        cfg_path = os.path.join(_PROJECT, "data", "MM_xor.xml")
        init_config(path=cfg_path, defaults_path=_DEFAULTS)
        Language.TheGrammar._configured = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m, _ = Models.BasicModel.from_config(cfg_path)
            Models.TheData.load("xor")
        ps_space = m.perceptualSpace
        ps = ps_space.percept_store
        # Override the knobs to make the assertions sharper without
        # touching the production config.
        ps.promotion_threshold = 3
        ps.promotion_min_length = 3
        threshold = ps.promotion_threshold
        min_length = ps.promotion_min_length
        # Pick one word that promotes (len >= min_length) and one
        # that never does (len < min_length).
        promoting_word = "promotable"   # len 10, >= 3
        short_word = "ab"               # len 2, < 3
        words_per_row = [
            [promoting_word, short_word],
        ]
        self.assertGreaterEqual(len(promoting_word), min_length)
        self.assertLess(len(short_word), min_length)
        what_buf = torch.zeros(1, 4, 1, dtype=torch.long)
        upstream = ps_space.subspace
        upstream._host_tokens = words_per_row
        upstream.set_what(what_buf)
        upstream.batch = 1
        # Before the threshold is reached, the promoting word stays
        # transient -- get_id must return None.
        for i in range(threshold - 1):
            ps_space._embed_radix(upstream)
            self.assertIsNone(
                ps.get_id(promoting_word.encode("utf-8")),
                f"{promoting_word!r} promoted prematurely on hit "
                f"{i + 1} (threshold={threshold})")
            self.assertIsNone(
                ps.get_id(short_word.encode("utf-8")),
                f"{short_word!r} (below min_length) promoted on hit "
                f"{i + 1} -- min_length gate ignored")
        # The threshold-th call should promote the promoting word.
        ps_space._embed_radix(upstream)
        self.assertIsNotNone(
            ps.get_id(promoting_word.encode("utf-8")),
            f"{promoting_word!r} should have promoted on hit "
            f"{threshold} (threshold reached)")
        # The short word never promotes regardless of how many hits.
        for _ in range(threshold * 3):
            ps_space._embed_radix(upstream)
        self.assertIsNone(
            ps.get_id(short_word.encode("utf-8")),
            f"{short_word!r} (below min_length={min_length}) must "
            f"never promote regardless of hit count")


if __name__ == "__main__":
    unittest.main()
