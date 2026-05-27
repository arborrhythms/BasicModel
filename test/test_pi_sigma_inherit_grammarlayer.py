"""Stage 2 substrate refactor: PiLayer / SigmaLayer inherit from GrammarLayer.

Post-Stage-2 contract (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md):

  * ``PiLayer`` and ``SigmaLayer`` inherit from ``GrammarLayer`` (not from
    ``Layer`` directly). The class docstring at ``bin/Layers.py:1541``
    already documents this contract; Stage 2 makes the code match.

  * As ``GrammarLayer`` subclasses, ``PiLayer`` / ``SigmaLayer`` carry the
    GrammarLayer class-attr defaults: ``rule_name == ""`` (empty -- they
    are anonymous substrate folds, not chart-dispatched grammar
    operators), ``arity == 1`` (the base default; the binary
    ``compose(left, right)`` API is overridden directly on each subclass
    and is not gated by the ``arity`` flag).

  * Pi / Sigma stay anonymous substrate folds: their empty ``rule_name``
    means ``GrammarLayer.__init__``'s auto-registration with the chart
    authority is a no-op. They are instantiated directly by spaces
    (``PerceptualSpace.pi`` / ``PerceptualSpace.sigma``), not by the
    chart parser's typed-GrammarLayer registry.

  * This Stage 2 inheritance change is a precondition for Stage 5
    (butterfly mode on GrammarLayer base): the butterfly cascade lives
    on GrammarLayer and is inherited uniformly by both substrate
    (Pi / Sigma) and grammar-op layers.
"""

import os
import sys
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Layers import GrammarLayer, Layer, PiLayer, SigmaLayer


class TestPiSigmaInheritGrammarLayer(unittest.TestCase):
    """Pi / Sigma must be GrammarLayer subclasses."""

    def test_pi_is_grammarlayer_subclass(self):
        """PiLayer is a direct subclass of GrammarLayer."""
        self.assertTrue(issubclass(PiLayer, GrammarLayer))

    def test_sigma_is_grammarlayer_subclass(self):
        """SigmaLayer is a direct subclass of GrammarLayer."""
        self.assertTrue(issubclass(SigmaLayer, GrammarLayer))

    def test_pi_mro_direct_parent(self):
        """PiLayer's immediate parent in the MRO is GrammarLayer
        (not Layer). PiLayer.__mro__[1] is the direct base."""
        self.assertIs(PiLayer.__mro__[1], GrammarLayer)

    def test_sigma_mro_direct_parent(self):
        """SigmaLayer's immediate parent in the MRO is GrammarLayer
        (not Layer). SigmaLayer.__mro__[1] is the direct base."""
        self.assertIs(SigmaLayer.__mro__[1], GrammarLayer)

    def test_pi_instance_is_grammarlayer(self):
        """An instantiated PiLayer passes isinstance(.., GrammarLayer)."""
        layer = PiLayer(nInput=3, nOutput=3)
        self.assertIsInstance(layer, GrammarLayer)
        # And of course still a Layer (transitively).
        self.assertIsInstance(layer, Layer)

    def test_sigma_instance_is_grammarlayer(self):
        """An instantiated SigmaLayer passes isinstance(.., GrammarLayer)."""
        layer = SigmaLayer(nInput=3, nOutput=3)
        self.assertIsInstance(layer, GrammarLayer)
        self.assertIsInstance(layer, Layer)

    def test_pi_inherits_empty_rule_name(self):
        """PiLayer inherits GrammarLayer's default ``rule_name == ""``.
        Pi / Sigma stay anonymous substrate folds; the empty rule_name
        causes ``GrammarLayer.__init__``'s chart-authority
        auto-registration to be a no-op."""
        layer = PiLayer(nInput=3, nOutput=3)
        self.assertEqual(layer.rule_name, "")

    def test_sigma_inherits_empty_rule_name(self):
        """SigmaLayer inherits GrammarLayer's default ``rule_name == ""``."""
        layer = SigmaLayer(nInput=3, nOutput=3)
        self.assertEqual(layer.rule_name, "")

    def test_pi_class_attr_rule_name_unchanged(self):
        """PiLayer does not override the class-level rule_name attr."""
        self.assertEqual(PiLayer.rule_name, "")

    def test_sigma_class_attr_rule_name_unchanged(self):
        """SigmaLayer does not override the class-level rule_name attr."""
        self.assertEqual(SigmaLayer.rule_name, "")


if __name__ == '__main__':
    unittest.main()
