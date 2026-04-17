"""Regression: useButterflies/useGrammar reflect config for every live model.xml."""

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
from util import init_config


def _load(cfg_name):
    path = os.path.join(_DATA, cfg_name)
    init_config(path=path, defaults_path=os.path.join(_DATA, "model.xml"))
    Spaces.TheGrammar._configured = False
    model, _cfg = Models.MentalModel.from_config(path)
    model.eval()
    return model


# Locks per-config (useButterflies, useGrammar) values so later refactors
# can't silently flip them.  Configs that cannot instantiate on the current
# tree are excluded; their loading failures are pre-existing issues
# (unrelated to the ramsified refactor) and are tracked separately:
#   - model.xml         -- BasicModel template, not a MentalModel
#   - MM_5M.xml         -- reconstruct=concepts fails validate_config
#   - MM_shamatha.xml   -- ConceptualSpace nVectors!=nActive
#   - MM_xor_step4.xml  -- ConceptualSpace nVectors!=nActive
EXPECTED = {
    "MM_400M.xml":           (True,  "none"),
    "MM_xor.xml":            (True,  "none"),
    "RamsifiedModel.xml":    (True,  "none"),
    "MM_bpe.xml":            (False, "none"),
    "MentalModel.xml":       (False, "none"),
    "MM_grammar.xml":        (False, "all"),
}


class TestOrthogonalFlags(unittest.TestCase):
    def test_flags_match_expected(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for cfg, (expect_bf, expect_gr) in EXPECTED.items():
                with self.subTest(cfg=cfg):
                    m = _load(cfg)
                    self.assertEqual(
                        bool(m.useButterflies), expect_bf,
                        f"{cfg}: useButterflies expected {expect_bf}")
                    self.assertEqual(
                        m.useGrammar, expect_gr,
                        f"{cfg}: useGrammar expected {expect_gr!r}")
                    # The exclusion invariant must hold.
                    self.assertFalse(
                        m.useButterflies and m.useGrammar == "all",
                        f"{cfg}: butterflies+grammar excluded")


if __name__ == "__main__":
    unittest.main()
