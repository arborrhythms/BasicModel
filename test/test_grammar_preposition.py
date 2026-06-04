"""PREPOSITION grammar op (Phase 1).

doc/plans/2026-06-03-contextual-bind-preposition-when.md "Operation 1:
PREPOSITION". A parameter-free, content-transparent binary C-tier
operator that packages a learned surface marker P (that / to / in /
because / when) with a phrase X (NP / VP / S). PREPOSITION does NOT
decide the final relation; that is learned from how the marker-headed
phrase participates downstream. Hard rule: no global POS inventory --
this is a role-only operator, not a part of speech.
"""

import math, os, sys, unittest
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from Language import PrepositionLayer


class TestPrepositionLayer(unittest.TestCase):
    def test_class_contract(self):
        self.assertEqual(PrepositionLayer.rule_name, "preposition")
        self.assertEqual(PrepositionLayer.arity, 2)
        self.assertEqual(PrepositionLayer.tier, "C")

    def test_parameter_free_construction(self):
        self.assertIsInstance(PrepositionLayer(), PrepositionLayer)  # _resolve_rule_layer uses cls()

    def test_forward_is_content_transparent(self):
        layer = PrepositionLayer()
        marker, phrase = torch.randn(2, 3, 6), torch.randn(2, 3, 6)
        out = layer.forward(marker, phrase)
        self.assertEqual(out.shape, phrase.shape)
        self.assertTrue(torch.allclose(out, phrase, atol=1e-6))  # phrase survives unchanged

    def test_compose_matches_forward(self):
        layer = PrepositionLayer(); m, p = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
        self.assertTrue(torch.allclose(layer.compose(m, p), layer.forward(m, p), atol=1e-6))

    def test_reverse_structural_split(self):
        layer = PrepositionLayer(); parent = torch.randn(2, 3, 6)
        left, right = layer.reverse(parent)
        self.assertTrue(torch.allclose(right, parent, atol=1e-6))  # content side recovers phrase
        self.assertEqual(left.shape, parent.shape)

    def test_permissive_arguments(self):  # accepts NP/VP/S-shaped content; gating is a learned hook
        layer = PrepositionLayer()
        for d in (2, 6, 10):
            x = torch.randn(2, 3, d); self.assertEqual(layer.forward(x, x).shape, x.shape)


class TestPrepositionRegistration(unittest.TestCase):
    def test_in_grammar_layer_classes(self):
        from Language import GRAMMAR_LAYER_CLASSES
        self.assertIs(GRAMMAR_LAYER_CLASSES["preposition"], PrepositionLayer)

    def test_surface_schema_assigned(self):
        from Layers import T3_BINARY_DIRECTIONAL   # learned PRE marker, does not select the op
        self.assertEqual(PrepositionLayer.surface_schema, T3_BINARY_DIRECTIONAL)


if __name__ == "__main__":
    unittest.main()
