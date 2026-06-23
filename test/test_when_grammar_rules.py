"""PREPOSITION grammar rules + signal-router binding (Phase 1, Tasks 1.3 / 1.4).

doc/plans/2026-06-03-contextual-bind-preposition-when.md "Operation 1:
PREPOSITION". The op fires automatically once it is (a) in
``GRAMMAR_LAYER_CLASSES``, (b) constructible with ``cls()``, and (c)
named by a ``<rule>`` in ``data/complete.grammar``.
These tests assert the grammar carries the role-only rule and that the
production wiring path (``SymbolSubSpace._resolve_rule_layer``) resolves it
to a ``PrepositionLayer``. Hard rule: no global POS inventory.
"""

import math, os, sys, unittest
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from Language import PrepositionLayer


def _load_complete():
    """Load the broad role-only grammar via the standard path."""
    from Language import Grammar
    g = Grammar()
    g.load_from_grammar_file("complete.grammar")
    return g


class TestPrepositionGrammarRules(unittest.TestCase):
    def test_preposition_rule_present(self):
        """The broad role-only grammar declares a ``preposition`` rule."""
        g = _load_complete()
        names = {r.method_name for r in g.rules_upward if r.method_name}
        self.assertIn("preposition", names)

    def test_preposition_forward_reverse_pairing(self):
        """``preposition`` has both a compose (upward) and a generate
        (downward) rule on the collapsed I/O roles."""
        g = _load_complete()
        up = [r for r in g.rules_upward if r.method_name == "preposition"]
        dn = [r for r in g.rules_downward if r.method_name == "preposition"]
        self.assertTrue(up, "no forward preposition rule")
        self.assertTrue(dn, "no reverse preposition rule")
        self.assertEqual(up[0].lhs, "preposition_O1")
        self.assertEqual(up[0].rhs_symbols,
                         ("preposition_I1", "preposition_I2"))


class TestBindGrammarRules(unittest.TestCase):
    """Phase 2, Task 2.3: the broad role-only grammar declares a ``bind`` rule
    (contextual missing-NP marker). Same auto-fire contract as preposition:
    registered class + ``cls()`` + a ``<rule>`` naming it. Role-only (no POS)."""

    def test_bind_rule_present(self):
        g = _load_complete()
        names = {r.method_name for r in g.rules_upward if r.method_name}
        self.assertIn("bind", names)

    def test_bind_forward_reverse_pairing(self):
        g = _load_complete()
        up = [r for r in g.rules_upward if r.method_name == "bind"]
        dn = [r for r in g.rules_downward if r.method_name == "bind"]
        self.assertTrue(up, "no forward bind rule")
        self.assertTrue(dn, "no reverse bind rule")
        self.assertEqual(up[0].lhs, "bind_O1")
        self.assertEqual(up[0].rhs_symbols, ("bind_I1", "bind_I2"))


class TestTenseGrammarRules(unittest.TestCase):
    """The broad role-only grammar declares unary ``tense`` as a live .when op.

    ``aspect`` remains a code-level helper for morphology, but is no longer a
    standalone grammar rule; the complete grammar reserves that slot for a
    future rewrite path.
    """

    def test_tense_rule_present_and_aspect_not_live(self):
        g = _load_complete()
        names = {r.method_name for r in g.rules_upward if r.method_name}
        self.assertIn("tense", names)
        self.assertNotIn("aspect", names)

    def test_tense_forward_reverse_pairing(self):
        g = _load_complete()
        up = [r for r in g.rules_upward if r.method_name == "tense"]
        dn = [r for r in g.rules_downward if r.method_name == "tense"]
        self.assertTrue(up, "no forward tense rule")
        self.assertTrue(dn, "no reverse tense rule")
        self.assertEqual(up[0].lhs, "tense_O1")
        self.assertEqual(up[0].rhs_symbols, ("tense_I1",))   # UNARY: one input

class TestPrepositionSignalRouterBinding(unittest.TestCase):
    """Task 1.4: the auto-wiring contract -- ``_wire_signal_router_grammar
    _ops`` resolves the ``preposition`` rule to a ``PrepositionLayer`` via
    ``SymbolSubSpace._resolve_rule_layer(space_role, name)``.

    Verification path: we call the REAL production method
    ``SymbolSubSpace._resolve_rule_layer('CS', 'preposition')``. That method
    only reads ``self._host_layer_registry`` before falling through to
    ``GRAMMAR_LAYER_CLASSES['preposition']()`` (the ``cls()`` call), so we
    build a bare SymbolSubSpace via ``__new__`` with an empty host registry
    rather than the heavy 3-space constructor. This exercises the genuine
    lookup-order -> ``cls()`` path the wiring uses; preposition is not
    pre-registered in any host registry, so it resolves through the
    fresh-instance branch exactly as a parameter-free op should.
    """

    def test_resolve_rule_layer_yields_preposition_layer(self):
        from Language import SymbolSubSpace
        ss = SymbolSubSpace.__new__(SymbolSubSpace)   # bypass the multi-space ctor
        ss._host_layer_registry = {}              # the only attr the method reads
        layer = ss._resolve_rule_layer('CS', 'preposition')
        self.assertIsInstance(layer, PrepositionLayer)
        self.assertEqual(layer.space_role, 'CS')

    def test_auto_wiring_contract_holds(self):
        """Guard the three conditions that make the op fire automatically:
        registered class, ``cls()`` constructibility, CS-space_role."""
        from Language import GRAMMAR_LAYER_CLASSES
        self.assertIs(GRAMMAR_LAYER_CLASSES['preposition'], PrepositionLayer)
        inst = PrepositionLayer()             # the cls() call _resolve_rule_layer makes
        self.assertEqual(inst.space_role, 'CS')


class TestMentalModelWhenEnabled(unittest.TestCase):
    """Phase 6 (Task 6.1): MentalModel.xml ships ``.when`` ON. Its Input /
    Perceptual subspaces carry a 2-dim WhenRangeEncoding (nWhen == nWhere == 2),
    while Conceptual / Symbolic / Output keep it disabled (nWhen == 0). The
    plan's "turn it on" flip was already committed (nWhen already matched
    nWhere in every space), so this is a guard against a silent later disable
    or a hardcoded-width drift, not a new switch. A finite ``.when`` loss on a
    real forward is already exercised by test_basicmodel.py's nWhen=2 model
    suite; here we lock the structural enablement and the encoding type.
    """

    def test_when_enabled_at_width_two_with_range_encoding(self):
        import os, warnings
        import Models, Language
        from util import init_config
        data = str(Path(__file__).resolve().parent.parent / "data")
        path = os.path.join(data, "MentalModel.xml")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            init_config(path=path, defaults_path=os.path.join(data, "model.xml"))
            Language.TheGrammar._configured = False
            model, _cfg = Models.BasicModel.from_config(path)
            model.eval()
        carriers = [(n, m) for n, m in model.named_modules()
                    if getattr(m, "whenEncoding", None) is not None]
        self.assertTrue(carriers, "no whenEncoding carriers found in MentalModel")
        enabled = []
        for name, mod in carriers:
            we = mod.whenEncoding
            # Phase 3's range encoding is the one in use everywhere.
            self.assertEqual(type(we).__name__, "WhenRangeEncoding", name)
            # nWhen and the encoding width agree (no hardcoded-width drift).
            self.assertEqual(mod.nWhen, we.nDim, f"{name}: nWhen != encoding nDim")
            if mod.nWhen == 2:
                enabled.append(name)
        # .when is genuinely ON at width 2 for the input + perceptual subspaces.
        self.assertTrue(any("inputSpace.subspace" in n for n in enabled), enabled)
        self.assertTrue(any("perceptualSpace" in n for n in enabled), enabled)


if __name__ == "__main__":
    unittest.main()
