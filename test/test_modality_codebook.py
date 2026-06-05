"""PerceptualSpace / SymbolicSpace codebook defaults + opt-out.

The model.xml defaults for the PS/SS tiers are <codebook>quantize</codebook>,
so configs that don't override inherit a codebook. 2026-06-04 the *mandatory*
constraint (architecture.MANDATORY_CODEBOOK_TIERS) was REVERTED: a config may
now explicitly resolve PS/SS to <codebook>none</codebook> to build a full-width
INVERTIBLE PASSTHROUGH (no VQ snap) -- required by the exact-XOR reconstruction
fixture, where an exactly invertible forward<->reverse chain is the point.
"""

import os, sys, tempfile, unittest, warnings
from pathlib import Path
import xml.etree.ElementTree as ET
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

import Models, Language
from util import init_config

_DATA = str(Path(__file__).resolve().parent.parent / "data")
_DEFAULTS = os.path.join(_DATA, "model.xml")


def _build(cfg):
    init_config(path=cfg, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    model, _ = Models.BasicModel.from_config(cfg)
    model.eval()
    return model


class TestMandatoryCodebooks(unittest.TestCase):

    def test_ps_ss_codebooks_present(self):
        """MentalModel's PerceptualSpace and SymbolicSpace resolve to a
        non-"none" codebook mode (inheriting the quantize default)."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = _build(os.path.join(_DATA, "MentalModel.xml"))
        self.assertNotEqual(model.perceptualSpace.codebook_mode, "none",
                            "PerceptualSpace codebook is mandatory")
        self.assertNotEqual(model.symbolicSpace.codebook_mode, "none",
                            "SymbolicSpace codebook is mandatory")

    def test_codebook_none_allowed(self):
        """2026-06-04: an explicit <codebook>none</codebook> on PerceptualSpace
        / SymbolicSpace is now ALLOWED (the mandatory-codebook constraint was
        reverted). data/XOR_exact.xml resolves BOTH to "none" -- a full-width
        invertible passthrough -- and must build without raising."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = _build(os.path.join(_DATA, "XOR_exact.xml"))
        self.assertEqual(model.perceptualSpace.codebook_mode, "none",
                         "PerceptualSpace codebook=none should be honored")
        self.assertEqual(model.symbolicSpace.codebook_mode, "none",
                         "SymbolicSpace codebook=none should be honored")


if __name__ == "__main__":
    unittest.main()
