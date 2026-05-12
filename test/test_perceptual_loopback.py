"""PerceptualSpace conceptual loopback (C→P) — parallel percept-concept
subsymbolic loop acceptance.

Mirrors ConceptualSpace's symbolic loopback one tier down: the
``conceptualSpace_ref`` attribute carries a reference to the model's
ConceptualSpace; ``PerceptualSpace._sourced_input`` reads that
sibling's prior event, lifts the bivector through the C-tier codebook's
SVD pseudo-inverse when applicable, and averages it into the primary
input (forwardBegin reshape).

Tests in this file:
  * The reference and helper method exist on PerceptualSpace.
  * When the ref is None or the C event is unset, ``_sourced_input``
    returns the primary input unchanged (legacy single-source
    semantics; sentence-start / standalone-construction compatibility).
  * After a full model forward, ``perceptualSpace.conceptualSpace_ref``
    is wired to a non-None ConceptualSpace.
  * ``SubsymbolicSpace`` and the ``<architecture><mode>grammar|parallel
    </mode>`` selector have been retired.
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

_CONFIG = os.path.join(_PROJECT, "data", "MM_xor_bivector.xml")
_DEFAULTS = os.path.join(_PROJECT, "data", "model.xml")


def _fresh_model():
    """Build a fresh BasicModel from MM_xor_bivector.xml with XOR data."""
    import Models
    import Language
    from util import init_config
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    m, cfg = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    return m


class TestPerceptualLoopbackContract(unittest.TestCase):
    """The new C→P attribute / method surface lives on PerceptualSpace."""

    def test_conceptualSpace_ref_attribute_exists(self):
        import Spaces
        # The attribute is initialised in __init__ to None for
        # standalone construction; the Model wires it post-construction.
        # Verify the class attribute path exists at construction time
        # by checking the default after a bare instantiation isn't
        # cheap (Space needs XML config); instead, assert via the
        # source that __init__ sets it.
        import inspect
        src = inspect.getsource(Spaces.PerceptualSpace.__init__)
        self.assertIn("self.conceptualSpace_ref = None", src,
                      "PerceptualSpace.__init__ must initialise "
                      "conceptualSpace_ref so attribute access is safe "
                      "before the Model wires the sibling reference.")

    def test_sourced_input_method_exists(self):
        import Spaces
        self.assertTrue(hasattr(Spaces.PerceptualSpace, '_sourced_input'),
                        "PerceptualSpace must expose _sourced_input "
                        "for the C→P loopback (mirrors "
                        "ConceptualSpace._sourced_input one tier down).")

    def test_read_event_helper_exists(self):
        import Spaces
        self.assertTrue(hasattr(Spaces.PerceptualSpace, '_read_event'),
                        "PerceptualSpace must expose _read_event "
                        "(materialised sibling event reader).")


class TestSubsymbolicSpaceRetired(unittest.TestCase):
    """The standalone SubsymbolicSpace class and its mode selector are gone."""

    def test_subsymbolic_space_class_removed(self):
        import Spaces
        self.assertFalse(hasattr(Spaces, 'SubsymbolicSpace'),
                         "SubsymbolicSpace should be removed; "
                         "PerceptualSpace is the subsymbolic substrate "
                         "via the new C→P conceptualSpace_ref loopback.")

    def test_subsymbolicSpace_ref_not_on_conceptual(self):
        import Spaces
        import inspect
        src = inspect.getsource(Spaces.ConceptualSpace.__init__)
        self.assertNotIn("self.subsymbolicSpace_ref = None", src,
                         "ConceptualSpace must not retain a "
                         "subsymbolicSpace_ref attribute after the "
                         "SubsymbolicSpace retirement.")

    def test_subsymbolic_widen_dim_param_removed(self):
        import Spaces
        import inspect
        sig = inspect.signature(Spaces.ConceptualSpace.__init__)
        self.assertNotIn("subsymbolic_widen_dim", sig.parameters,
                         "ConceptualSpace.__init__ must not accept "
                         "subsymbolic_widen_dim; the right-half "
                         "loopback widening was retired together with "
                         "SubsymbolicSpace.")

    def test_get_active_input_sibling_returns_symbolic(self):
        """_get_active_input_sibling no longer toggles by <mode>."""
        import Spaces
        import inspect
        src = inspect.getsource(Spaces.ConceptualSpace._get_active_input_sibling)
        # The retired path read architecture.mode; the new path
        # returns symbolicSpace_ref unconditionally.
        self.assertNotIn("architecture.mode", src,
                         "_get_active_input_sibling must no longer "
                         "read the retired <architecture><mode> knob.")


class TestSourcedInputCold(unittest.TestCase):
    """``_sourced_input`` degrades cleanly when no C ref / no C event."""

    @classmethod
    def setUpClass(cls):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            cls.model = _fresh_model()
        cls.percep = cls.model.perceptualSpace

    def test_ref_wired_after_model_build(self):
        """Model.__init__ post-wires conceptualSpace_ref on PerceptualSpace."""
        self.assertIsNotNone(self.percep.conceptualSpace_ref,
                             "Model.__init__ must wire "
                             "perceptualSpace.conceptualSpace_ref to a "
                             "ConceptualSpace instance (parallel "
                             "percept-concept subsymbolic loop).")
        # Should point at a ConceptualSpace (or the terminal stage in
        # the staged BasicModel path).
        import Spaces
        self.assertIsInstance(self.percep.conceptualSpace_ref,
                              Spaces.ConceptualSpace)

    def test_ref_cleared_falls_back_to_forwardBegin(self):
        """Setting conceptualSpace_ref=None makes _sourced_input
        equivalent to forwardBegin (legacy single-source path)."""
        m = self.model
        # Run one forward to populate subspace state.
        import warnings
        m.eval()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            loader = m.inputSpace.data.data_loader(split="train", num_streams=1)
            inp_items, out_items = next(iter(loader))
            inputTensor = m.inputSpace.prepInput(inp_items)
            with torch.no_grad():
                m.forward(inputTensor)

        # Save the live ref then temporarily clear it.
        saved_ref = self.percep.conceptualSpace_ref
        try:
            object.__setattr__(self.percep, 'conceptualSpace_ref', None)
            # _sourced_input with ref=None must equal forwardBegin
            # (since there is no sibling to merge in). Build a fresh
            # vspace context for the comparison.
            vspace = self.percep.subspace
            with torch.no_grad():
                only_primary = self.percep._sourced_input(vspace)
                expected = self.percep.forwardBegin(vspace, returnVectors=True)
            self.assertTrue(torch.equal(only_primary, expected),
                            "With conceptualSpace_ref=None, "
                            "_sourced_input must return the primary "
                            "forwardBegin output unchanged.")
        finally:
            object.__setattr__(self.percep, 'conceptualSpace_ref',
                               saved_ref)


class TestLexiconOnSymbolicSpace(unittest.TestCase):
    """SymbolicSpace is the logical owner of the orthographic Lexicon.

    Post-lexicon-migration: ``S.vocabulary`` returns the Embedding,
    and the orthographic-API methods (``train_embeddings``,
    ``sbow_loss``, ``reconstruct_data``, ``reconstruct_to_buffer``,
    ``get_recovered_word``, ``_snapshot_embeddings``,
    ``set_embedding_sigma``) live on ``SymbolicSpace`` and delegate
    to the physical Embedding via ``perceptualSpace_ref``. The
    Embedding itself still lives on PerceptualSpace because the input
    pipeline's InputSpace._peer_perceptual.vocabulary wiring requires
    it there -- ownership and access pattern, not physical location.
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
                             "Model.__init__ must wire "
                             "symbolicSpace.perceptualSpace_ref so S "
                             "can reach the orthographic Embedding.")
        import Spaces
        self.assertIsInstance(sym.perceptualSpace_ref,
                              Spaces.PerceptualSpace)

    def test_symbolic_vocabulary_returns_lexicon(self):
        """S.vocabulary delegates to the Embedding on PerceptualSpace."""
        # MM_xor_bivector uses Codebook-based PerceptualSpace, not text.
        # In Codebook mode, S.vocabulary should fall back to S's own
        # ``.what`` codebook (the legacy code path) since P doesn't
        # carry an Embedding. Either way, S.vocabulary returns
        # something non-None.
        sym = self.model.symbolicSpace
        v = sym.vocabulary
        # MM_xor_bivector has a S.what codebook (bivector regime).
        # The fallback path returns it when no Embedding on P.
        self.assertIsNotNone(v,
                             "S.vocabulary must return the Embedding "
                             "from P (text mode) or fall back to S's "
                             "own .what (numeric / non-text mode).")

    def test_symbolic_lexicon_methods_exist(self):
        """All migrated orthographic API methods are exposed on S."""
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
    """Post-2026-05-12 split: grammar's canonical home is S. The
    per-tier SyntacticLayer dispatchers at P and C are retained as
    backward-compat no-ops for grammars that omit P/C rules (which
    is every current production grammar). The "split" is realised at
    the grammar-XML level (current grammars list only S-tier rules)
    rather than by deleting the dispatch mechanism wholesale.
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
        """Current production grammar (MM_xor_bivector) lists no
        P-tier rules -- the split's semantic intent.
        """
        import Language
        p_rules = [r for r in Language.TheGrammar.rules
                   if getattr(r, 'tier', None) == 'P']
        self.assertEqual(len(p_rules), 0,
                         f"Production grammar must list no P-tier "
                         f"rules; found {p_rules}")

    def test_no_c_tier_rules_in_grammar(self):
        """Current production grammar (MM_xor_bivector) lists no
        C-tier rules -- the split's semantic intent.
        """
        import Language
        c_rules = [r for r in Language.TheGrammar.rules
                   if getattr(r, 'tier', None) == 'C']
        self.assertEqual(len(c_rules), 0,
                         f"Production grammar must list no C-tier "
                         f"rules; found {c_rules}")


if __name__ == "__main__":
    unittest.main()
