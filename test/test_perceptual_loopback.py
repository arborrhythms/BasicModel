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

    def test_perceptual_forward_arity(self):
        import inspect
        import Spaces
        sig = inspect.signature(Spaces.PerceptualSpace.forward)
        params = [n for n in sig.parameters if n != "self"]
        self.assertEqual(len(params), 2,
                         f"PerceptualSpace.forward must be "
                         f"(IS_subspace, CS_subspaceForPS=None); got {params}")
        # Second arg optional so standalone single-arg callers still work.
        second = sig.parameters[params[1]]
        self.assertIsNot(second.default, inspect.Parameter.empty,
                         "CS_subspaceForPS must default to None.")

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

    def test_symbolic_forward_single_arg(self):
        import inspect
        import Spaces
        sig = inspect.signature(Spaces.SymbolicSpace.forward)
        params = [n for n in sig.parameters if n != "self"]
        self.assertEqual(len(params), 1,
                         f"SymbolicSpace.forward must be "
                         f"(CS_subspaceForSS); got {params}")

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
        import warnings
        m = self.model
        m.eval()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                m.forward(self._one_input())
        cs = m.conceptualSpaces[-1]
        self.assertTrue(hasattr(cs, '_subspaceForPS'),
                        "Terminal ConceptualSpace must expose "
                        "_subspaceForPS after forward.")
        self.assertTrue(hasattr(cs, '_subspaceForSS'),
                        "Terminal ConceptualSpace must expose "
                        "_subspaceForSS after forward.")

    def test_cold_start_none_equals_default(self):
        """forward(IS) and forward(IS, None) are the same call; an empty
        CS_subspaceForPS also degrades to the primary pi_input path
        (no crash, produces a percept event)."""
        import warnings
        m = self.model
        m.eval()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                in_sub = m.inputSpace.forward(self._one_input())
                empty = m._empty_subspace()
                a = self.percep.forward(in_sub)
                ev_a = a.materialize() if a is not None else None
                b = self.percep.forward(in_sub, empty)
                ev_b = b.materialize() if b is not None else None
        self.assertIsNotNone(ev_a,
                             "forward(IS) must produce a percept event.")
        self.assertIsNotNone(ev_b,
                             "forward(IS, empty) must produce a percept "
                             "event (cold-start degrades to pi_input).")
        self.assertEqual(tuple(ev_a.shape), tuple(ev_b.shape),
                         "None vs empty CS feedback must not change the "
                         "percept shape (both take the primary path).")


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


if __name__ == "__main__":
    unittest.main()
