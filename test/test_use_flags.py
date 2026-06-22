"""Regression: useGrammar reflects config for every live model.xml."""

import os
import sys
import unittest
import warnings

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Models
import Spaces
import Language
from util import init_config


def _load(cfg_name):
    path = os.path.join(_DATA, cfg_name)
    init_config(path=path, defaults_path=os.path.join(_DATA, "model.xml"))
    Language.TheGrammar._configured = False
    model, _cfg = Models.BasicModel.from_config(path)
    model.eval()
    return model


# Locks per-config useGrammar values so later refactors can't silently
# flip them.  Configs that cannot instantiate on the current tree are
# excluded; their loading failures are pre-existing issues tracked
# separately:
#   - model.xml         -- BasicModel template, not a BasicModel
#   - MM_20M.xml         -- reconstruct=concepts fails validate_config
#   - MM_400M.xml       -- relied on butterfly N-halving (post 2026-05-12)
#   - MM_shamatha.xml   -- ConceptualSpace nVectors!=nActive
#   - MM_xor_step4.xml  -- ConceptualSpace nVectors!=nActive
# ``useGrammar`` is derived from the grammar rules
# (``_derive_use_grammar``): ``"none"`` when every operational rule is
# a unary substrate fold (``pi`` / ``sigma``), ``"all"`` when any other
# operator (``intersection`` / ``union`` / ``not`` / ``lift`` / …) appears.
# The configs below all carry such operators at S-space_role (XOR's
# not/intersection/union, MentalModel's full Boolean grammar, etc.),
# so the derived value is ``"all"`` for every entry.
# (MM_xor.xml dropped 2026-06-04: it deliberately sets PartSpace
#  <codebook>none</codebook>, which the converged modality architecture
#  rejects -- PS/SS codebooks are mandatory -- so it no longer instantiates.)
EXPECTED = {
    "RamsifiedModel.xml":    "all",
    "MM_bpe.xml":            "all",
    "MentalModel.xml":       "all",
    "MM_grammar.xml":        "all",
}


class TestOrthogonalFlags(unittest.TestCase):
    def test_flags_match_expected(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for cfg, expect_gr in EXPECTED.items():
                with self.subTest(cfg=cfg):
                    m = _load(cfg)
                    self.assertEqual(
                        m.useGrammar, expect_gr,
                        f"{cfg}: useGrammar expected {expect_gr!r}")


if __name__ == "__main__":
    unittest.main()
